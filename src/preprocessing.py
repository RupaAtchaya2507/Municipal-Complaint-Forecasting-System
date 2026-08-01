"""
Data Preprocessing Module
=========================
Clean data, extract time features, encode status, compute resolution time,
standardize complaint categories, and remove duplicates.

Handles 6 complaint statuses from the actual CSV:
  Open, Resolved, On-the-Job, Re-opened, Rejected, Closed
"""

import pandas as pd
import numpy as np
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the unified dataset.

    - Remove exact duplicate rows
    - Remove rows with missing latitude or longitude
    - Fill NULL categorical fields with 'Unknown'
    - Handle category_id NaN → 0
    """
    initial_len = len(df)

    # Remove exact duplicate rows
    df = df.drop_duplicates().copy()
    dupes_dropped = initial_len - len(df)
    if dupes_dropped > 0:
        logger.info(f"Removed {dupes_dropped} duplicate rows")

    # Drop rows with missing coordinates
    before_coords = len(df)
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    dropped = before_coords - len(df)
    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with missing coordinates")

    # Fill NULL categoricals (string columns)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        null_count = df[col].isna().sum()
        if null_count > 0:
            df[col] = df[col].fillna("Unknown")
            logger.info(f"Filled {null_count} NULLs in '{col}' with 'Unknown'")

    # Handle category_id NaN
    if "category_id" in df.columns and df["category_id"].isna().any():
        n = df["category_id"].isna().sum()
        df["category_id"] = df["category_id"].fillna(0).astype(int)
        logger.info(f"Filled {n} NaN category_id with 0")

    # Handle civic_agency_id NaN (619 nulls in actual data)
    if "civic_agency_id" in df.columns and df["civic_agency_id"].isna().any():
        n = df["civic_agency_id"].isna().sum()
        df["civic_agency_id"] = df["civic_agency_id"].fillna(0).astype(int)
        logger.info(f"Filled {n} NaN civic_agency_id with 0")

    logger.info(f"Clean dataset: {len(df)} rows")
    return df


def compute_resolution_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute resolution time in hours for resolved complaints.

    Requires 'created_at' and 'updated_at' (or 'resolved_at') columns.
    Adds:
      - resolution_time_hours: float (NaN for unresolved complaints)

    Falls back to NaN for all rows if the resolved timestamp column is absent.
    """
    df = df.copy()

    # Determine which column holds the resolution timestamp
    resolved_col = None
    for col in ["resolved_at", "updated_at", "closed_at"]:
        if col in df.columns:
            resolved_col = col
            break

    if resolved_col is None:
        df["resolution_time_hours"] = np.nan
        logger.info("No resolution timestamp column found — resolution_time_hours set to NaN")
        return df

    df[resolved_col] = pd.to_datetime(df[resolved_col], errors="coerce")

    # Only compute for resolved complaints (status_encoded == 0)
    mask = df.get("status_encoded", pd.Series(1, index=df.index)) == 0
    delta = (df.loc[mask, resolved_col] - df.loc[mask, "created_at"]).dt.total_seconds() / 3600.0

    df["resolution_time_hours"] = np.nan
    df.loc[mask, "resolution_time_hours"] = delta

    # Clamp negative values (data entry errors)
    df.loc[df["resolution_time_hours"] < 0, "resolution_time_hours"] = np.nan

    resolved_count = mask.sum()
    valid_count = df["resolution_time_hours"].notna().sum()
    logger.info(
        f"Computed resolution_time_hours: {valid_count}/{resolved_count} resolved "
        f"complaints have valid times (mean={df['resolution_time_hours'].mean():.1f}h)"
    )
    return df


CATEGORY_STANDARDIZATION_MAP = {
    # Drainage / Water
    "drainage": "Drainage",
    "drain": "Drainage",
    "waterlogging": "Drainage",
    "flood": "Drainage",
    "sewage": "Drainage",
    "sewer": "Drainage",
    # Roads / Potholes
    "pothole": "Pothole",
    "road": "Pothole",
    "road repair": "Pothole",
    "road damage": "Pothole",
    # Garbage / Waste
    "garbage": "Garbage",
    "waste": "Garbage",
    "solid waste": "Garbage",
    "litter": "Garbage",
    "trash": "Garbage",
    # Street Lights
    "street light": "Street Light",
    "streetlight": "Street Light",
    "light": "Street Light",
    # Water Supply
    "water supply": "Water Supply",
    "water": "Water Supply",
    "pipe": "Water Supply",
    # Noise
    "noise": "Noise",
    "sound": "Noise",
    # Construction
    "construction": "Construction",
    "building": "Construction",
}


def standardize_categories(df: pd.DataFrame, category_col: str = "category_title") -> pd.DataFrame:
    """
    Standardize free-text complaint categories into canonical groups.

    Maps raw category strings to standard names using keyword matching.
    Unmapped categories are labelled 'Other'.

    Adds column: 'complaint_category' (standardized label)
    """
    df = df.copy()

    if category_col not in df.columns:
        logger.warning(f"Column '{category_col}' not found — complaint_category set to 'Other'")
        df["complaint_category"] = "Other"
        return df

    def map_category(raw: str) -> str:
        if pd.isna(raw):
            return "Other"
        lower = str(raw).strip().lower()
        for keyword, standard in CATEGORY_STANDARDIZATION_MAP.items():
            if keyword in lower:
                return standard
        return "Other"

    df["complaint_category"] = df[category_col].apply(map_category)

    dist = df["complaint_category"].value_counts()
    logger.info(f"Standardized categories distribution:\n{dist.to_string()}")
    return df


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from the 'created_at' timestamp.

    Adds columns:
      - hour_of_day (0–23)
      - day_of_week (0=Monday, 6=Sunday)
      - is_weekend (0 or 1)
      - month (1–12)
      - is_festival_eve (1 if day before a festival, else 0)
    """
    df = df.copy()
    dt = df["created_at"]

    df["hour_of_day"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = dt.dt.month

    # Compute is_festival_eve
    if "festival_flag" in df.columns and "date" in df.columns:
        # Find all festival dates
        festival_dates = set(
            df.loc[df["festival_flag"] == 1, "date"].unique()
        )

        if festival_dates:
            # A date is festival_eve if the NEXT day is a festival
            df["is_festival_eve"] = df["date"].apply(
                lambda d: 1 if (d + timedelta(days=1)) in festival_dates else 0
            )
        else:
            df["is_festival_eve"] = 0
            logger.info("No festival dates found — is_festival_eve set to 0")
    else:
        df["is_festival_eve"] = 0
        logger.info("No festival column — is_festival_eve set to 0")

    logger.info("Extracted time features: hour_of_day, day_of_week, "
                "is_weekend, month, is_festival_eve")
    return df


def encode_status(
    df: pd.DataFrame,
    open_statuses: list = None,
    resolved_statuses: list = None,
) -> pd.DataFrame:
    """
    Encode complaint status as binary.

    Actual statuses in the CSV and their encoding:
      Open        → 1  (unresolved)
      On-the-Job  → 1  (still in progress)
      Re-opened   → 1  (unresolved again)
      Resolved    → 0  (terminal)
      Closed      → 0  (terminal)
      Rejected    → 0  (terminal)

    Args:
        df: DataFrame with complaint_status_title column
        open_statuses: list of status strings to encode as 1 (open/unresolved)
        resolved_statuses: list of status strings to encode as 0 (resolved/closed)
    """
    df = df.copy()

    if open_statuses is None:
        open_statuses = ["open", "on-the-job", "re-opened"]
    if resolved_statuses is None:
        resolved_statuses = ["resolved", "closed", "rejected"]

    if "complaint_status_title" not in df.columns:
        logger.warning("No 'complaint_status_title' column found")
        df["status_encoded"] = 1  # default to open
        return df

    status = df["complaint_status_title"].str.strip().str.lower()

    # Map based on configured lists
    def map_status(s):
        s = str(s).strip().lower()
        if s in open_statuses:
            return 1
        elif s in resolved_statuses:
            return 0
        else:
            # Fallback: keyword matching
            if any(kw in s for kw in ["resolved", "closed", "completed", "rejected"]):
                return 0
            return 1  # default to open/unresolved

    df["status_encoded"] = status.apply(map_status)

    # Log the distribution
    status_dist = df.groupby("complaint_status_title")["status_encoded"].first()
    logger.info(f"Status encoding mapping:")
    for orig, encoded in status_dist.items():
        count = (df["complaint_status_title"] == orig).sum()
        label = "Open(1)" if encoded == 1 else "Resolved(0)"
        logger.info(f"  {orig:15s} → {label} ({count} rows)")

    open_count = (df["status_encoded"] == 1).sum()
    resolved_count = (df["status_encoded"] == 0).sum()
    logger.info(f"Total: {open_count} Open/Unresolved, {resolved_count} Resolved/Closed")

    return df


def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline.

    Steps:
      1. clean_data          — dedup, drop bad coords, fill nulls
      2. encode_status       — binary status encoding
      3. compute_resolution_time — resolution time in hours
      4. standardize_categories  — canonical complaint_category labels
      5. extract_time_features   — temporal feature columns

    Returns cleaned DataFrame ready for aggregation.
    """
    df = clean_data(df)
    df = encode_status(df)
    df = compute_resolution_time(df)
    df = standardize_categories(df)
    df = extract_time_features(df)
    return df
