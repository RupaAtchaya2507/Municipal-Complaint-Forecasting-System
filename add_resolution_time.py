"""
Add resolved_at timestamps to synthetic_complaints.csv
=======================================================
Adds a realistic 'resolved_at' column to all "Resolved" complaints
based on category-specific resolution time distributions derived from
real BBMP municipal response patterns.

Resolution time distributions per category (hours):
  Pothole          : 24 – 168h  (1–7 days)
  Garbage          : 6  – 48h   (same day to 2 days)
  Street Light     : 12 – 72h   (half day to 3 days)
  Drainage         : 24 – 120h  (1–5 days)
  Water Supply     : 6  – 48h   (same day to 2 days)
  Noise            : 4  – 24h   (few hours to 1 day)
  Construction     : 48 – 240h  (2–10 days)
  Other            : 12 – 96h   (half day to 4 days)

Run:
  python add_resolution_time.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ── Resolution time parameters per category keyword (min_hours, max_hours, mean_hours) ──
CATEGORY_RESOLUTION_PARAMS = {
    "pothole":      (24,  168, 72),
    "road":         (24,  168, 72),
    "garbage":      (6,   48,  18),
    "waste":        (6,   48,  18),
    "street light": (12,  72,  36),
    "streetlight":  (12,  72,  36),
    "drainage":     (24,  120, 60),
    "drain":        (24,  120, 60),
    "sewage":       (24,  120, 60),
    "water supply": (6,   48,  20),
    "water":        (6,   48,  20),
    "noise":        (4,   24,  10),
    "pollution":    (12,  72,  36),
    "construction": (48,  240, 96),
    "building":     (48,  240, 96),
    "default":      (12,  96,  48),
}


def get_resolution_params(category_title: str) -> tuple:
    """Return (min_hours, max_hours, mean_hours) for a given category title."""
    if pd.isna(category_title):
        return CATEGORY_RESOLUTION_PARAMS["default"]
    lower = str(category_title).strip().lower()
    for keyword, params in CATEGORY_RESOLUTION_PARAMS.items():
        if keyword == "default":
            continue
        if keyword in lower:
            return params
    return CATEGORY_RESOLUTION_PARAMS["default"]


def generate_resolution_times(
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.Series:
    """
    Generate resolved_at timestamps for all Resolved complaints.

    Uses a log-normal distribution shaped by category-specific
    min/max/mean resolution hours. Unresolved complaints get NaT.
    """
    resolved_at = pd.Series([pd.NaT] * len(df), dtype="datetime64[ns]")

    resolved_mask = df["complaint_status_title"].str.strip().str.lower() == "resolved"
    resolved_indices = df.index[resolved_mask]

    if len(resolved_indices) == 0:
        return resolved_at

    # Process in one vectorized pass per category
    cat_col = "category_title" if "category_title" in df.columns else None

    for idx in resolved_indices:
        row = df.loc[idx]
        cat = row[cat_col] if cat_col else None
        min_h, max_h, mean_h = get_resolution_params(cat)

        # Log-normal sampling: mu and sigma derived from mean and range
        # sigma chosen so 95% of values fall within [min_h, max_h]
        sigma = (np.log(max_h) - np.log(min_h)) / 4.0
        mu = np.log(mean_h)
        hours = float(rng.lognormal(mean=mu, sigma=sigma))

        # Clamp to valid range
        hours = max(min_h, min(max_h, hours))

        created = row["created_at"]
        if pd.notna(created):
            resolved_at.loc[idx] = created + timedelta(hours=hours)

    return resolved_at


def main():
    csv_path = config.COMPLAINTS_CSV
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path, encoding="latin-1")

    # Check if already has resolved_at
    if "resolved_at" in df.columns:
        print("Column 'resolved_at' already exists. Skipping.")
        return

    print(f"Total records: {len(df):,}")

    # Parse created_at
    df["created_at"] = pd.to_datetime(df["created_at"], format="mixed",
                                       dayfirst=False, errors="coerce")

    resolved_count = (df["complaint_status_title"].str.strip().str.lower() == "resolved").sum()
    print(f"Resolved complaints: {resolved_count:,}")

    # Generate resolution times
    rng = np.random.default_rng(42)
    print("Generating resolved_at timestamps...")

    # Vectorized approach — process by category for speed
    df["resolved_at"] = pd.NaT
    resolved_mask = df["complaint_status_title"].str.strip().str.lower() == "resolved"

    cat_col = "category_title" if "category_title" in df.columns else None

    # Group by category for fast batch processing
    if cat_col:
        for cat_title, group_idx in df[resolved_mask].groupby(cat_col).groups.items():
            min_h, max_h, mean_h = get_resolution_params(cat_title)
            n = len(group_idx)

            sigma = (np.log(max_h) - np.log(min_h)) / 4.0
            mu = np.log(mean_h)
            hours_array = rng.lognormal(mean=mu, sigma=sigma, size=n)
            hours_array = np.clip(hours_array, min_h, max_h)

            created_times = df.loc[group_idx, "created_at"]
            resolved_times = [
                ct + timedelta(hours=float(h)) if pd.notna(ct) else pd.NaT
                for ct, h in zip(created_times, hours_array)
            ]
            df.loc[group_idx, "resolved_at"] = resolved_times
    else:
        # Fallback: use default params for all
        min_h, max_h, mean_h = CATEGORY_RESOLUTION_PARAMS["default"]
        n = resolved_mask.sum()
        sigma = (np.log(max_h) - np.log(min_h)) / 4.0
        mu = np.log(mean_h)
        hours_array = rng.lognormal(mean=mu, sigma=sigma, size=n)
        hours_array = np.clip(hours_array, min_h, max_h)
        group_idx = df.index[resolved_mask]
        created_times = df.loc[group_idx, "created_at"]
        resolved_times = [
            ct + timedelta(hours=float(h)) if pd.notna(ct) else pd.NaT
            for ct, h in zip(created_times, hours_array)
        ]
        df.loc[group_idx, "resolved_at"] = resolved_times

    # Format resolved_at back to string matching created_at format
    df["resolved_at"] = pd.to_datetime(df["resolved_at"]).dt.strftime("%m/%d/%Y %H:%M")

    # Format created_at back to original string format
    df["created_at"] = df["created_at"].dt.strftime("%m/%d/%Y %H:%M")

    # Verify
    valid_resolved = df["resolved_at"].notna().sum()
    print(f"Generated resolved_at for {valid_resolved:,} complaints")

    # Show sample stats per category
    df["_resolved_at_dt"] = pd.to_datetime(df["resolved_at"], format="%m/%d/%Y %H:%M", errors="coerce")
    df["_created_at_dt"]  = pd.to_datetime(df["created_at"],  format="%m/%d/%Y %H:%M", errors="coerce")
    df["_res_hours"] = (df["_resolved_at_dt"] - df["_created_at_dt"]).dt.total_seconds() / 3600.0

    print("\nResolution time stats by category:")
    if cat_col:
        stats = df[df["_res_hours"].notna()].groupby(cat_col)["_res_hours"].agg(["mean", "min", "max"]).round(1)
        print(stats.to_string())

    print(f"\nOverall mean resolution time: {df['_res_hours'].mean():.1f} hours")
    print(f"Overall median resolution time: {df['_res_hours'].median():.1f} hours")

    # Drop helper columns
    df.drop(columns=["_resolved_at_dt", "_created_at_dt", "_res_hours"], inplace=True)

    # Save back
    print(f"\nSaving updated CSV to {csv_path} ...")
    df.to_csv(csv_path, index=False, encoding="latin-1")
    print("Done. 'resolved_at' column added successfully.")


if __name__ == "__main__":
    main()
