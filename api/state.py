"""
App State
=========
Loads the full ML pipeline once at startup.
All API endpoints read from this shared state — no reloading per request.
"""

import os
import sys
import logging
import numpy as np
import torch

# Add project root so imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.utils import setup_logging, set_seed, get_device
from src.data_ingestion import ingest_all
from src.preprocessing import preprocess_pipeline
from src.clustering import find_optimal_clusters, create_zones, build_adjacency_matrix
from src.aggregation import create_time_windows, aggregate_by_zone_window, fill_missing_windows
from src.features import feature_pipeline
from src.dataset import create_sequences, LAST_MSI_COMPONENTS
from src.model import SpatioTemporalModel, MultiTaskSpatioTemporalModel
from src.risk_engine import RiskEngine
from src.hotspot_dbscan import run_hotspot_pipeline

logger = logging.getLogger(__name__)


class AppState:
    """
    Singleton container for all pipeline artifacts.
    Populated once during FastAPI lifespan startup.
    """

    def __init__(self):
        self.ready = False

        # Pipeline artifacts
        self.model = None
        self.adj_matrix: np.ndarray = None
        self.adj_tensor: torch.Tensor = None
        self.feature_tensor: np.ndarray = None
        self.feature_names: list = []
        self.agg_df = None
        self.df_complaints = None
        self.centroids: np.ndarray = None
        self.num_zones: int = 0
        self.num_features: int = 0
        self.device: torch.device = None

        # Latest inference results (updated on each /predict call)
        self.last_predictions: np.ndarray = None        # [N] predicted MSI
        self.last_risk_results: list = []
        self.last_hotspot_result: dict = {}

        # MSI targets for the full dataset (used by /msi endpoint)
        self.y_msi: np.ndarray = None                   # [samples, N]
        self.X_sequences: np.ndarray = None             # [samples, seq_len, N, F]

        # Risk engine (stateful EMA)
        self.risk_engine: RiskEngine = None

    def load(self):
        """Run the full pipeline and populate all state fields."""
        setup_logging()
        set_seed(config.RANDOM_SEED)
        self.device = get_device()
        logger.info("AppState.load() — starting pipeline...")

        # ── Data Ingestion ──
        weather_path = config.WEATHER_CSV if os.path.exists(config.WEATHER_CSV) else None
        festival_path = config.FESTIVALS_CSV if os.path.exists(config.FESTIVALS_CSV) else None
        df = ingest_all(config.COMPLAINTS_CSV, weather_path, festival_path,
                        encoding=config.CSV_ENCODING)
        df = preprocess_pipeline(df)
        self.df_complaints = df

        # ── Spatial Clustering ──
        coords = df[["latitude", "longitude"]].values
        optimal_k = find_optimal_clusters(coords, config.CLUSTER_K_RANGE)
        df, centroids = create_zones(df, optimal_k)
        adj_matrix = build_adjacency_matrix(centroids, k=config.KNN_NEIGHBORS,
                                             epsilon=config.EDGE_EPSILON)
        self.num_zones = optimal_k
        self.centroids = centroids
        self.adj_matrix = adj_matrix
        self.adj_tensor = torch.FloatTensor(adj_matrix).to(self.device)

        # ── Temporal Aggregation ──
        df = create_time_windows(df, config.TIME_WINDOW_HOURS)
        agg_df = aggregate_by_zone_window(df)
        agg_df = fill_missing_windows(agg_df, self.num_zones)
        self.agg_df = agg_df

        # ── Feature Engineering ──
        feature_tensor, feature_names, _, agg_df_featured = feature_pipeline(
            agg_df, self.num_zones, adj_matrix
        )
        self.feature_tensor = feature_tensor
        self.feature_names = feature_names
        self.num_features = feature_tensor.shape[2]
        self.agg_df_featured = agg_df_featured

        # Rainfall feature index for W (weather anomaly)
        self._rainfall_idx = (feature_names.index("rainfall")
                              if "rainfall" in feature_names else None)

        # Road quality per zone from static features
        static_path = os.path.join(config.DATA_DIR, "zone_static_features.csv")
        if os.path.exists(static_path):
            import pandas as pd
            static_df = pd.read_csv(static_path).sort_values("Zone_ID")
            if "hist_resolution_rate" in static_df.columns:
                rq = static_df["hist_resolution_rate"].values[:self.num_zones]
                self._road_quality = np.clip(rq.astype(float), 0.0, 1.0)
            else:
                self._road_quality = np.full(self.num_zones, 0.5)
        else:
            self._road_quality = np.full(self.num_zones, 0.5)

        # ── Sequence Dataset ──
        X, y_msi = create_sequences(
            feature_tensor,
            seq_len=config.DEFAULT_SEQ_LEN,
            adjacency_matrix=adj_matrix,
            scaling_method=getattr(config, "SCALING_METHOD", "robust"),
            horizon=1,
            predict_delta=False,
            rainfall_feature_idx=self._rainfall_idx,
            road_quality=self._road_quality.tolist(),
        )
        self.X_sequences = X
        self.y_msi = y_msi

        # ── Model ──
        model_path = os.path.join(config.MODEL_DIR, "best_model.pt")
        base = SpatioTemporalModel(
            num_features=self.num_features,
            num_zones=self.num_zones,
            gcn_hidden=config.GCN_HIDDEN_DIM,
            lstm_hidden=config.LSTM_HIDDEN_DIM,
            lstm_layers=config.LSTM_NUM_LAYERS,
            dropout=config.DROPOUT_RATE,
            use_sigmoid=getattr(config, "USE_SIGMOID", False),
        )
        model_type = getattr(config, "MODEL_TYPE", "multi_task")
        if model_type == "multi_task":
            self.model = MultiTaskSpatioTemporalModel(base).to(self.device)
        else:
            self.model = base.to(self.device)

        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=self.device)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                self.model.load_state_dict(ckpt["state_dict"])
            elif isinstance(ckpt, dict):
                self.model.load_state_dict(ckpt)
            else:
                self.model = ckpt
            logger.info(f"Loaded model weights from {model_path}")
        else:
            logger.warning(f"No model checkpoint at {model_path} — using untrained weights")

        self.model.eval()

        # ── Risk Engine ──
        self.risk_engine = RiskEngine(
            num_zones=self.num_zones,
            alpha=config.EMA_ALPHA,
            thresholds=config.RISK_THRESHOLDS,
            weighting_method=getattr(config, "RISK_WEIGHTING_METHOD", "dynamic"),
        )

        # ── Initial inference to populate last_predictions ──
        self._run_inference()

        # ── Hotspot detection ──
        self._run_hotspots()

        self.ready = True
        logger.info(f"AppState ready: {self.num_zones} zones, {self.num_features} features")

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _run_inference(self, horizon: int = 1):
        """Run model forward pass on the latest sequence and update risk results."""
        import pandas as pd

        X_input = torch.FloatTensor(self.X_sequences[-1:]).to(self.device)
        is_multitask = isinstance(self.model, MultiTaskSpatioTemporalModel)

        with torch.no_grad():
            if is_multitask:
                p_msi, p_cnt, p_unres = self.model(X_input, self.adj_tensor)
                self._last_cnt   = p_cnt.cpu().numpy().flatten()
                self._last_unres = p_unres.cpu().numpy().flatten()
            else:
                p_msi = self.model(X_input, self.adj_tensor)
                self._last_cnt   = None
                self._last_unres = None

        pred_delta = p_msi.cpu().numpy().flatten()

        # Reconstruct absolute MSI
        if getattr(config, "PREDICT_DELTA", True):
            prev_msi = self.y_msi[-2] if len(self.y_msi) > 1 else np.zeros(self.num_zones)
            self.last_predictions = pred_delta + prev_msi
        else:
            self.last_predictions = pred_delta

        # W and V values
        last_window = self.agg_df_featured["time_window"].max()
        last_data = (self.agg_df_featured[self.agg_df_featured["time_window"] == last_window]
                     .sort_values("zone_id"))

        U_values = last_data["U"].values if "U" in last_data.columns else np.zeros(self.num_zones)
        D_values = last_data["D"].values if "D" in last_data.columns else np.zeros(self.num_zones)

        if self._rainfall_idx is not None and "rainfall" in last_data.columns:
            rain = last_data["rainfall"].values.astype(float)
            hist_mean = self.agg_df_featured["rainfall"].mean()
            hist_std  = self.agg_df_featured["rainfall"].std() + 1e-6
            W_values  = np.clip((rain - hist_mean) / hist_std, 0.0, 1.0)
        else:
            W_values = np.zeros(self.num_zones)

        road_vuln = 1.0 - self._road_quality
        V_values  = np.clip(0.6 * road_vuln + 0.4 * U_values, 0.0, 1.0)

        self.last_risk_results = self.risk_engine.compute_all_zones(
            U_values, D_values, self.last_predictions, W_values, V_values
        )

        # store U/D/W/V for MSI endpoint
        self._last_U = U_values
        self._last_D = D_values
        self._last_W = W_values
        self._last_V = V_values

    def _run_hotspots(self):
        """Run DBSCAN hotspot detection on the complaint DataFrame."""
        if self.df_complaints is not None:
            self.last_hotspot_result = run_hotspot_pipeline(
                self.df_complaints,
                eps_meters=500.0,
                min_samples=5,
            )


# Global singleton
app_state = AppState()
