"""
Feature Engineering Module
==========================
Compute derived features, attach temporal/external features, normalize,
and build the final feature tensor [T × N × F].
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def compute_derived_features(agg_df: pd.DataFrame, adjacency_matrix: np.ndarray = None) -> pd.DataFrame:
    """
    Compute derived features from aggregated counts.
    
    Adds:
      - U_raw: unresolved ratio = unresolved / (unresolved + resolved + 1)
      - D_raw: density = complaint_count / (max_complaint_count_per_window + 1)
      - Rolling complaint averages: 3-day and 7-day averages.
      - Rolling unresolved averages: 3-day and 7-day averages.
      - Trend features: complaint velocity (diff).
      - Persistence features: days since last complaint/open complaint.
      - Neighbor averages: graph neighbor complaint/unresolved average.
    """
    df = agg_df.copy()

    # Unresolved ratio
    df["U_raw"] = df["unresolved_count"] / (
        df["unresolved_count"] + df["resolved_count"] + 1
    )

    # Use raw complaint_count for density (will be MinMax normalized later)
    df["D_raw"] = df["complaint_count"].astype(float)
    
    # Sort primarily to ensure time continuity within zones for diffing and rolling
    df = df.sort_values(["zone_id", "time_window"])
    
    # Delta density & 3-step rolling average density per zone
    df["delta_density"] = df.groupby("zone_id")["D_raw"].diff().fillna(0.0)
    df["rolling_avg_density"] = df.groupby("zone_id")["D_raw"].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )

    # Rolling Complaint averages (3-day and 7-day)
    df["3_day_complaint_avg"] = df.groupby("zone_id")["complaint_count"].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df["7_day_complaint_avg"] = df.groupby("zone_id")["complaint_count"].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )

    # Rolling Unresolved averages (3-day and 7-day)
    df["3_day_unresolved_avg"] = df.groupby("zone_id")["unresolved_count"].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df["7_day_unresolved_avg"] = df.groupby("zone_id")["unresolved_count"].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )

    # Trend: complaint velocity (first-difference of complaint counts)
    df["complaint_velocity"] = df.groupby("zone_id")["complaint_count"].diff().fillna(0.0)

    # Persistence: steps since last complaint / last open complaint
    def steps_since_last(series):
        out = []
        count = 999.0  # high default value
        for val in series:
            if val > 0:
                count = 0.0
            else:
                count += 1.0
            out.append(count)
        return pd.Series(out, index=series.index)

    df["days_since_last_complaint"] = df.groupby("zone_id")["complaint_count"].transform(steps_since_last)
    df["days_since_last_open_complaint"] = df.groupby("zone_id")["unresolved_count"].transform(steps_since_last)

    # Neighbor averages
    if adjacency_matrix is not None:
        num_zones = adjacency_matrix.shape[0]
        neighbors_dict = {}
        for z in range(num_zones):
            neighbors_dict[z] = [j for j in range(num_zones) if j != z and adjacency_matrix[z, j] > 0]
            
        # Compute using pivot-melt
        complaints_pivot = df.pivot(index="time_window", columns="zone_id", values="complaint_count").fillna(0.0)
        unresolved_pivot = df.pivot(index="time_window", columns="zone_id", values="unresolved_count").fillna(0.0)
        
        complaints_neighbor = pd.DataFrame(index=complaints_pivot.index, columns=complaints_pivot.columns, dtype=float)
        unresolved_neighbor = pd.DataFrame(index=unresolved_pivot.index, columns=unresolved_pivot.columns, dtype=float)
        
        for z in range(num_zones):
            neighs = neighbors_dict[z]
            if len(neighs) > 0:
                complaints_neighbor[z] = complaints_pivot[neighs].mean(axis=1)
                unresolved_neighbor[z] = unresolved_pivot[neighs].mean(axis=1)
            else:
                complaints_neighbor[z] = 0.0
                unresolved_neighbor[z] = 0.0
                
        complaints_melt = complaints_neighbor.reset_index().melt(id_vars="time_window", value_name="neighbor_complaint_avg")
        unresolved_melt = unresolved_neighbor.reset_index().melt(id_vars="time_window", value_name="neighbor_unresolved_avg")
        
        # Merge back
        df = df.merge(complaints_melt, on=["time_window", "zone_id"], how="left")
        df = df.merge(unresolved_melt, on=["time_window", "zone_id"], how="left")
    else:
        df["neighbor_complaint_avg"] = 0.0
        df["neighbor_unresolved_avg"] = 0.0

    # Output Data Distribution stats
    logger.info("Complaint count stats:\n" + str(df['complaint_count'].describe()))
    logger.info(f"Non-zero ratio: {(df['complaint_count'] > 0).mean():.4f}")

    logger.info("Computed derived features: U_raw, D_raw, delta_density, rolling_avg_density, rolling averages, persistence, neighbor pressure")
    return df


def add_temporal_features(
    agg_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure temporal features are present in the aggregated DataFrame.
    
    If not already attached during aggregation, extract from time_window.
    Columns: hour_of_day, day_of_week, is_weekend, month, is_festival_eve
    """
    df = agg_df.copy()
    tw = pd.to_datetime(df["time_window"])

    if "hour_of_day" not in df.columns:
        df["hour_of_day"] = tw.dt.hour
    if "day_of_week" not in df.columns:
        df["day_of_week"] = tw.dt.dayofweek
    if "is_weekend" not in df.columns:
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    if "month" not in df.columns:
        df["month"] = tw.dt.month
    if "is_festival_eve" not in df.columns:
        df["is_festival_eve"] = 0

    logger.info("Temporal features ensured in aggregated data")
    return df


def normalize_features(
    df: pd.DataFrame,
    columns: list = None,
) -> tuple:
    """
    Apply MinMax scaling to specified columns.
    
    Args:
        df: DataFrame to normalize
        columns: columns to scale
    
    Returns:
        (normalized_df, scaler)
    """
    if columns is None:
        columns = [
            "U_raw", "D_raw", "delta_density", "rolling_avg_density", 
            "temperature", "rainfall", "humidity",
            "3_day_complaint_avg", "7_day_complaint_avg",
            "3_day_unresolved_avg", "7_day_unresolved_avg",
            "complaint_velocity", "days_since_last_complaint",
            "days_since_last_open_complaint", "neighbor_complaint_avg",
            "neighbor_unresolved_avg"
        ]

    # Only normalize columns that exist
    cols_to_scale = [c for c in columns if c in df.columns]

    df = df.copy()
    scaler = MinMaxScaler()

    if cols_to_scale:
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale].fillna(0))
        logger.info(f"MinMax normalized: {cols_to_scale}")

    # Rename U_raw → U, D_raw → D after normalization
    if "U_raw" in cols_to_scale:
        df.rename(columns={"U_raw": "U"}, inplace=True)
    if "D_raw" in cols_to_scale:
        df.rename(columns={"D_raw": "D"}, inplace=True)

    return df, scaler


def build_feature_tensor(
    agg_df: pd.DataFrame,
    num_zones: int,
    feature_cols: list = None,
) -> np.ndarray:
    """
    Build the final feature tensor [T × N × F].
    
    Args:
        agg_df: aggregated DataFrame (sorted by time_window, zone_id)
        num_zones: number of spatial zones
        feature_cols: list of feature column names to include
    
    Returns:
        3D numpy array of shape [T, N, F]
    """
    if feature_cols is None:
        # Default feature set
        feature_cols = [
            # Core counts
            "complaint_count", "unresolved_count", "resolved_count",
            # Derived
            "U", "D", "delta_density", "rolling_avg_density",
            # New Advanced Spatiotemporal Features
            "3_day_complaint_avg", "7_day_complaint_avg",
            "3_day_unresolved_avg", "7_day_unresolved_avg",
            "complaint_velocity", "days_since_last_complaint",
            "days_since_last_open_complaint", "neighbor_complaint_avg",
            "neighbor_unresolved_avg",
            # Temporal
            "hour_of_day", "day_of_week", "is_weekend", "month", "is_festival_eve",
            # External
            "temperature", "rainfall", "humidity", "festival_flag",
        ]

    # Only use columns that exist
    available_features = [c for c in feature_cols if c in agg_df.columns]
    logger.info(f"Feature columns ({len(available_features)}): {available_features}")

    # Sort
    df = agg_df.sort_values(["time_window", "zone_id"]).reset_index(drop=True)

    time_windows = sorted(df["time_window"].unique())
    T = len(time_windows)
    N = num_zones
    F = len(available_features)

    tensor = np.zeros((T, N, F), dtype=np.float32)

    # Vectorized fill using pivot — avoids slow row-by-row iteration
    df_indexed = df.set_index(["time_window", "zone_id"])[available_features].fillna(0.0)
    for t_idx, tw in enumerate(time_windows):
        if tw in df_indexed.index:
            window_data = df_indexed.loc[tw]
            for z_id, row in window_data.iterrows():
                if int(z_id) < N:
                    tensor[t_idx, int(z_id), :] = row.values.astype(np.float32)

    logger.info(f"Feature tensor shape: [{T}, {N}, {F}]")
    return tensor, available_features


def feature_pipeline(
    agg_df: pd.DataFrame,
    num_zones: int,
    adjacency_matrix: np.ndarray = None,
) -> tuple:
    """
    Run the full feature engineering pipeline.
    
    Returns:
        (feature_tensor, feature_names, scaler, df)
    """
    df = compute_derived_features(agg_df, adjacency_matrix)
    df = add_temporal_features(df)
    df, scaler = normalize_features(df)
    tensor, feature_names = build_feature_tensor(df, num_zones)
    
    # Inject Static Baseline Features if enabled in config
    import config
    use_static = getattr(config, "USE_STATIC_FEATURES", False)
    if use_static:
        import os
        static_path = os.path.join(config.DATA_DIR, "zone_static_features.csv")
        if os.path.exists(static_path):
            static_df = pd.read_csv(static_path)
            # Normalize static features using MinMax scaling
            static_cols = [c for c in static_df.columns if c != "Zone_ID"]
            from sklearn.preprocessing import MinMaxScaler
            static_scaler = MinMaxScaler()
            
            # Sort by Zone_ID to align with N zones
            static_df_sorted = static_df.sort_values("Zone_ID").head(num_zones)
            scaled_static = static_scaler.fit_transform(static_df_sorted[static_cols].fillna(0.0))
            
            # Concatenate to tensor along the feature axis (axis 2)
            T, N, F = tensor.shape
            static_expanded = np.repeat(scaled_static[np.newaxis, :, :], T, axis=0) # shape [T, N, 11]
            tensor = np.concatenate([tensor, static_expanded], axis=2) # shape [T, N, F + 11]
            
            feature_names = feature_names + static_cols
            logger.info(f"Injected 11 static baseline features. New tensor shape: {tensor.shape}")
            
    return tensor, feature_names, scaler, df
