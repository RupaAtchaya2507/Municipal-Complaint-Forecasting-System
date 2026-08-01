"""
Sequence Dataset Module
=======================
Create sliding-window sequences and PyTorch DataLoaders
with strict chronological train/val/test split.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)


# Global dictionary to store calculated MSI components for diagnostics
LAST_MSI_COMPONENTS = {}

def create_sequences(
    feature_tensor: np.ndarray,
    seq_len: int,
    adjacency_matrix: np.ndarray = None,
    scaling_method: str = "robust",
    horizon: int = 1,
    predict_delta: bool = False,
    **kwargs
) -> tuple:
    """
    Create input-target sequences using a sliding window.

    For each time step t (where seq_len <= t <= T - horizon):
      Input:  feature_tensor[t-seq_len : t]  → shape [seq_len, N, F]
      Target: MSI at step t + horizon - 1    → shape [N] (continuous index)
              or Delta MSI if predict_delta=True

    Args:
        feature_tensor: shape [T, N, F]
        seq_len: number of past time steps to use as input
        adjacency_matrix: adjacency matrix of shape [N, N] to calculate neighbor pressure
        scaling_method: "minmax" or "robust" (default: "robust")
        horizon: prediction horizon step size (1 = next day, 3 = 3 days ahead, etc.)
        predict_delta: if True, returns Delta MSI (future_msi - current_msi) as y targets (default: True)


    Returns:
        X: np.ndarray of shape [samples, seq_len, N, F]
        y: np.ndarray of shape [samples, N] (continuous MSI targets)
    """
    T, N, F = feature_tensor.shape
    h = horizon

    if seq_len + h - 1 >= T:
        raise ValueError(
            f"seq_len ({seq_len}) + horizon - 1 ({h-1}) must be less than T ({T})"
        )

    # 1. Define index slices for aligned targets and inputs
    target_indices = np.arange(seq_len + h - 1, T)
    current_indices = np.arange(seq_len - 1, T - h)

    # 2. Extract raw components
    C = feature_tensor[target_indices, :, 0]  # Future Complaint Count

    # Future Unresolved Ratio: unresolved_count / max(complaint_count, 1)
    unresolved = feature_tensor[target_indices, :, 1]
    total = feature_tensor[target_indices, :, 0]
    U = unresolved / np.maximum(total, 1.0)

    # Complaint Trend / Growth Rate
    C_current = feature_tensor[current_indices, :, 0]
    G = (C - C_current) / np.maximum(C_current, 1.0)

    # Neighbor Pressure — distance-weighted using adjacency matrix edge weights
    N_pressure = np.zeros_like(C)
    if adjacency_matrix is not None:
        # Normalize adjacency rows so weights sum to 1 per zone
        adj_norm = adjacency_matrix.copy().astype(np.float64)
        row_sums = adj_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        adj_norm = adj_norm / row_sums
        for z in range(N):
            neighbors = [j for j in range(N) if j != z and adjacency_matrix[z, j] > 0]
            if len(neighbors) > 0:
                weights = adj_norm[z, neighbors]           # distance-based weights
                N_pressure[:, z] = (C[:, neighbors] * weights).sum(axis=1)

    # --- MODULE 5 ADDITIONS ---

    # Weather Anomaly Score (W): rainfall deviation above historical mean
    # Feature index for 'rainfall' — find dynamically, default to 0 if absent
    _rainfall_idx = kwargs.get("rainfall_feature_idx", None)
    if _rainfall_idx is not None and _rainfall_idx < feature_tensor.shape[2]:
        rainfall = feature_tensor[target_indices, :, _rainfall_idx]          # [samples, N]
        hist_mean = feature_tensor[:seq_len, :, _rainfall_idx].mean(axis=0)  # [N] historical mean
        hist_std  = feature_tensor[:seq_len, :, _rainfall_idx].std(axis=0) + 1e-6
        W = np.clip((rainfall - hist_mean) / hist_std, 0.0, None)            # only positive anomalies
    else:
        # Fallback: zero weather anomaly (no rainfall feature available)
        W = np.zeros_like(C)

    # Road Vulnerability Score (V): combines road quality + unresolved backlog
    # Road quality is a static feature per zone — passed via kwargs or defaults to 0.5
    road_quality = kwargs.get("road_quality", None)   # expected shape [N] in [0,1]
    if road_quality is not None:
        road_quality = np.array(road_quality, dtype=np.float32)
        # Invert: higher road quality index → lower vulnerability
        road_vuln = 1.0 - np.clip(road_quality, 0.0, 1.0)                   # [N]
        # Broadcast to [samples, N]
        road_vuln_broadcast = np.broadcast_to(road_vuln, C.shape).copy()
    else:
        road_vuln_broadcast = np.full_like(C, 0.5)   # neutral vulnerability if no road data

    # V = 0.6 * road_vulnerability + 0.4 * unresolved_ratio
    V = 0.6 * road_vuln_broadcast + 0.4 * U

    # 3. Normalize
    if scaling_method == "robust":
        from sklearn.preprocessing import RobustScaler

        def robust_scale(arr):
            scaler = RobustScaler()
            shape = arr.shape
            return scaler.fit_transform(arr.reshape(-1, 1)).reshape(shape)

        C_norm = robust_scale(np.log1p(C))
        U_norm = U                                        # already a ratio [0,1]
        G_norm = robust_scale(np.clip(G, -3.0, 3.0))
        N_norm = robust_scale(np.log1p(N_pressure))
        W_norm = robust_scale(np.clip(W, 0.0, 5.0))      # cap at 5-sigma anomaly
        V_norm = np.clip(V, 0.0, 1.0)                    # already in [0,1]
        logger.info("Using Robust scaling (log1p & clipped)")
    else:
        def minmax_scale(arr):
            lo, hi = arr.min(), arr.max()
            return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)

        C_norm = minmax_scale(C)
        U_norm = minmax_scale(U)
        G_norm = minmax_scale(G)
        N_norm = minmax_scale(N_pressure)
        W_norm = minmax_scale(W)
        V_norm = minmax_scale(V)
        logger.info("Using MinMax scaling")

    # 4. Assemble MSI target
    #
    #   MSI = 0.30*C  (Complaint Deviation    — strongest absolute predictor)
    #       + 0.25*U  (Unresolved Ratio        — high positive correlation)
    #       + 0.20*N  (Neighbor Pressure       — spatial spillover signal)
    #       + 0.15*G  (Complaint Trend Score   — growth direction)
    #       + 0.05*W  (Weather Anomaly Score   — external signal)
    #       + 0.05*V  (Road Vulnerability Score — static infrastructure)
    #
    y = (0.30 * C_norm
         + 0.25 * U_norm
         + 0.20 * N_norm
         + 0.15 * G_norm
         + 0.05 * W_norm
         + 0.05 * V_norm)

    # Save components globally for diagnostic reports
    global LAST_MSI_COMPONENTS
    LAST_MSI_COMPONENTS = {
        "C_raw": C,           "C_norm": C_norm,
        "U_raw": U,           "U_norm": U_norm,
        "G_raw": G,           "G_norm": G_norm,
        "N_raw": N_pressure,  "N_norm": N_norm,
        "W_raw": W,           "W_norm": W_norm,
        "V_raw": V,           "V_norm": V_norm,
        "MSI": y,
    }

    # 5. Slice inputs into sequences
    X_list = []
    # Input sequences end at t, so t goes from seq_len to T - h
    for t in range(seq_len, T - h + 1):
        x = feature_tensor[t - seq_len: t]  # [seq_len, N, F]
        X_list.append(x)

    X = np.stack(X_list, axis=0)  # [samples, seq_len, N, F]
    
    logger.info(f"Created continuous MSI target (scaling={scaling_method}, horizon={h})")
    
    if predict_delta:
        # Align Delta targets: y_delta[t] = y[t+1] - y[t]
        # X is sliced as X[1:] to match y[1:] targets
        y_delta = y[1:] - y[:-1]
        X = X[1:]
        y = y_delta
        logger.info(f"Target Formulation: Delta MSI (shape={y.shape})")
    else:
        logger.info(f"Target Formulation: Future MSI (shape={y.shape})")

    logger.info(f"MSI Stats: mean={y.mean():.4f}, median={np.median(y):.4f}, min={y.min():.4f}, max={y.max():.4f}, std={y.std():.4f}")
    logger.info(f"Created {len(X)} sequences (seq_len={seq_len}): X={X.shape}, y={y.shape}")
    return X, y


def time_based_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple:
    """
    Split data chronologically (NO shuffling).

    Args:
        X: input sequences [samples, seq_len, N, F]
        y: targets [samples, N]
        train_ratio: fraction for training
        val_ratio: fraction for validation
        (test = 1 - train - val)

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    logger.info(f"Time-based split: train={len(X_train)}, "
                f"val={len(X_val)}, test={len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


class SpatioTemporalDataset(Dataset):
    """
    PyTorch Dataset for spatiotemporal sequences.

    Args:
        X: input features [samples, seq_len, N, F]
        y: targets [samples, N]
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple:
    """
    Create train/val/test DataLoaders from feature arrays.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    X_train, y_train, X_val, y_val, X_test, y_test = time_based_split(
        X, y, train_ratio, val_ratio
    )

    train_ds = SpatioTemporalDataset(X_train, y_train)
    val_ds = SpatioTemporalDataset(X_val, y_val)
    test_ds = SpatioTemporalDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    logger.info(f"DataLoaders: train={len(train_ds)}, "
                f"val={len(val_ds)}, test={len(test_ds)}, "
                f"batch_size={batch_size}")

    return train_loader, val_loader, test_loader
