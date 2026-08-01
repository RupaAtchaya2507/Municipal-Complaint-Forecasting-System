"""
Temporal Aggregation Module
===========================
Assign complaints to 6-hour time windows and aggregate counts per zone.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def create_time_windows(
    df: pd.DataFrame,
    window_hours: int = 6,
) -> pd.DataFrame:
    """
    Assign each complaint to a fixed-interval time window.
    
    Windows are aligned to midnight: [00:00-06:00), [06:00-12:00), etc.
    
    Args:
        df: DataFrame with 'created_at' datetime column
        window_hours: window size in hours (default 6)
    
    Returns:
        DataFrame with 'time_window' column (period start timestamp)
    """
    df = df.copy()
    freq = f"{window_hours}h"
    df["time_window"] = df["created_at"].dt.floor(freq)

    n_windows = df["time_window"].nunique()
    logger.info(f"Created {n_windows} time windows ({window_hours}h each)")

    return df


def aggregate_by_zone_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate complaint data by (zone_id, time_window).
    
    Computes per group:
      - complaint_count: total complaints
      - unresolved_count: complaints with status_encoded == 1
      - resolved_count: complaints with status_encoded == 0
    
    Also carries forward aggregated temporal and external features
    (mean values per window for numeric features).
    """
    # Core aggregation
    agg = df.groupby(["time_window", "zone_id"]).agg(
        complaint_count=("status_encoded", "count"),
        unresolved_count=("status_encoded", "sum"),  # Open=1, so sum = unresolved
    ).reset_index()

    agg["resolved_count"] = agg["complaint_count"] - agg["unresolved_count"]

    # Aggregate temporal features (mode or mean per window)
    temporal_cols = ["hour_of_day", "day_of_week", "is_weekend", "month", "is_festival_eve"]
    available_temporal = [c for c in temporal_cols if c in df.columns]

    if available_temporal:
        temporal_agg = df.groupby(["time_window", "zone_id"])[available_temporal].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
        ).reset_index()
        agg = agg.merge(temporal_agg, on=["time_window", "zone_id"], how="left")

    # Aggregate external features (mean per window)
    external_cols = ["temperature", "rainfall", "humidity", "festival_flag"]
    available_external = [c for c in external_cols if c in df.columns]

    if available_external:
        external_agg = df.groupby(["time_window", "zone_id"])[available_external].mean(
            numeric_only=True
        ).reset_index()
        agg = agg.merge(external_agg, on=["time_window", "zone_id"], how="left")

    logger.info(f"Aggregated to {len(agg)} (zone, window) records")
    return agg


def fill_missing_windows(
    agg_df: pd.DataFrame,
    num_zones: int,
) -> pd.DataFrame:
    """
    Ensure every (time_window, zone_id) combination exists.
    
    Fills missing combinations with zero counts and forward-filled features.
    """
    all_windows = sorted(agg_df["time_window"].unique())
    all_zones = list(range(num_zones))

    # Create full index
    full_index = pd.MultiIndex.from_product(
        [all_windows, all_zones],
        names=["time_window", "zone_id"]
    )

    agg_df = agg_df.set_index(["time_window", "zone_id"])
    agg_df = agg_df.reindex(full_index)

    # Fill count columns with 0
    count_cols = ["complaint_count", "unresolved_count", "resolved_count"]
    for col in count_cols:
        if col in agg_df.columns:
            agg_df[col] = agg_df[col].fillna(0).astype(int)

    # Forward-fill then backward-fill other features per zone
    agg_df = agg_df.groupby("zone_id").apply(
        lambda g: g.ffill().bfill()
    )

    # Drop the extra zone_id index level if created
    if isinstance(agg_df.index, pd.MultiIndex) and agg_df.index.nlevels > 2:
        agg_df = agg_df.droplevel(0)

    agg_df = agg_df.reset_index()

    logger.info(f"Filled missing windows: {len(agg_df)} total records "
                f"({len(all_windows)} windows × {num_zones} zones)")
    return agg_df


def create_time_index(
    agg_df: pd.DataFrame,
    num_zones: int,
    count_cols: list = None,
) -> np.ndarray:
    """
    Convert aggregated DataFrame to a 3D numpy array.
    
    Args:
        agg_df: aggregated DataFrame sorted by time_window, zone_id
        num_zones: number of spatial zones
        count_cols: columns to include (default: complaint/unresolved/resolved counts)
    
    Returns:
        Array of shape [T, N, C] where T=time steps, N=zones, C=count columns
    """
    if count_cols is None:
        count_cols = ["complaint_count", "unresolved_count", "resolved_count"]

    # Sort chronologically
    agg_df = agg_df.sort_values(["time_window", "zone_id"]).reset_index(drop=True)

    time_windows = sorted(agg_df["time_window"].unique())
    T = len(time_windows)
    N = num_zones
    C = len(count_cols)

    tensor = np.zeros((T, N, C), dtype=np.float32)

    for t_idx, tw in enumerate(time_windows):
        window_data = agg_df[agg_df["time_window"] == tw]
        for _, row in window_data.iterrows():
            z = int(row["zone_id"])
            if z < N:
                tensor[t_idx, z, :] = [row[col] for col in count_cols]

    logger.info(f"Time index tensor: shape {tensor.shape} "
                f"(T={T}, N={N}, features={C})")
    return tensor
