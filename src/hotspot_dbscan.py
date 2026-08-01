"""
Hotspot Detection Module (Module 4)
=====================================
DBSCAN-based spatial clustering to identify:
  - Repeated complaint locations (persistent hotspots)
  - High-density risk zones
  - Emerging complaint clusters (recent surge hotspots)

Uses Haversine distance so epsilon is in real-world metres.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# Earth radius in metres for Haversine conversion
_EARTH_RADIUS_M = 6_371_000


def _coords_to_radians(df: pd.DataFrame, lat_col: str, lon_col: str) -> np.ndarray:
    """Convert lat/lon degrees to radians array [N, 2] for Haversine DBSCAN."""
    coords = df[[lat_col, lon_col]].values.astype(np.float64)
    return np.radians(coords)


def run_dbscan(
    df: pd.DataFrame,
    eps_meters: float = 500.0,
    min_samples: int = 5,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> pd.DataFrame:
    """
    Run DBSCAN on complaint GPS coordinates using Haversine distance.

    Args:
        df: DataFrame with lat/lon columns
        eps_meters: neighbourhood radius in metres (default 500m)
        min_samples: minimum complaints to form a dense cluster
        lat_col: latitude column name
        lon_col: longitude column name

    Returns:
        df with added 'hotspot_cluster' column
          -1 = noise (not part of any hotspot)
           0, 1, 2, ... = hotspot cluster ids
    """
    coords_rad = _coords_to_radians(df, lat_col, lon_col)

    # eps for Haversine must be in radians: eps_rad = eps_metres / earth_radius
    eps_rad = eps_meters / _EARTH_RADIUS_M

    db = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        algorithm="ball_tree",
        metric="haversine",
    )

    df = df.copy()
    df["hotspot_cluster"] = db.fit_predict(coords_rad)

    n_clusters = len(set(df["hotspot_cluster"])) - (1 if -1 in df["hotspot_cluster"].values else 0)
    n_noise = (df["hotspot_cluster"] == -1).sum()
    logger.info(
        f"DBSCAN (eps={eps_meters}m, min_samples={min_samples}): "
        f"{n_clusters} hotspot clusters, {n_noise} noise points"
    )
    return df


def detect_hotspots(
    df: pd.DataFrame,
    eps_meters: float = 500.0,
    min_samples: int = 5,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    category_col: str = "complaint_category",
    status_col: str = "complaint_status_title",
    ward_col: str = "ward_id",
) -> list[dict]:
    """
    Detect and describe spatial hotspots from complaint data.

    For each DBSCAN cluster, computes:
      - centroid (lat, lon)
      - complaint count
      - dominant complaint category
      - unresolved ratio
      - list of ward IDs covered
      - density score (complaints per sq-km)

    Args:
        df: preprocessed complaints DataFrame
        eps_meters: DBSCAN neighbourhood radius
        min_samples: minimum points to form a cluster
        lat_col: latitude column
        lon_col: longitude column
        category_col: standardized complaint category column
        status_col: complaint status column
        ward_col: ward identifier column

    Returns:
        List of hotspot dicts sorted by complaint count descending.
        Each dict has keys:
          hotspot_id, centroid_lat, centroid_lon, complaint_count,
          dominant_category, category_distribution, unresolved_ratio,
          wards_covered, density_score, risk_level
    """
    df = run_dbscan(df, eps_meters, min_samples, lat_col, lon_col)

    # Drop noise points
    clustered = df[df["hotspot_cluster"] != -1].copy()

    if clustered.empty:
        logger.warning("No hotspot clusters found. Try lowering eps_meters or min_samples.")
        return []

    hotspots = []
    cluster_ids = sorted(clustered["hotspot_cluster"].unique())

    for cid in cluster_ids:
        cluster_df = clustered[clustered["hotspot_cluster"] == cid]

        # Centroid
        centroid_lat = float(cluster_df[lat_col].mean())
        centroid_lon = float(cluster_df[lon_col].mean())
        complaint_count = len(cluster_df)

        # Dominant category
        if category_col in cluster_df.columns:
            cat_counts = cluster_df[category_col].value_counts()
            dominant_category = str(cat_counts.index[0])
            category_distribution = cat_counts.to_dict()
        else:
            dominant_category = "Unknown"
            category_distribution = {}

        # Unresolved ratio
        if status_col in cluster_df.columns:
            unresolved_mask = cluster_df[status_col].str.strip().str.lower().isin(
                ["open", "on-the-job", "re-opened"]
            )
            unresolved_ratio = float(unresolved_mask.sum() / len(cluster_df))
        else:
            unresolved_ratio = float("nan")

        # Wards covered
        if ward_col in cluster_df.columns:
            wards_covered = sorted(cluster_df[ward_col].dropna().unique().tolist())
        else:
            wards_covered = []

        # Density score: complaints per sq-km
        # Approximate bounding box area
        lat_range = cluster_df[lat_col].max() - cluster_df[lat_col].min()
        lon_range = cluster_df[lon_col].max() - cluster_df[lon_col].min()
        # 1 degree lat ≈ 111 km, 1 degree lon ≈ 111 * cos(lat) km
        lat_km = lat_range * 111.0
        lon_km = lon_range * 111.0 * abs(np.cos(np.radians(centroid_lat)))
        area_sqkm = max(lat_km * lon_km, 0.01)   # avoid division by zero
        density_score = round(complaint_count / area_sqkm, 2)

        hotspots.append({
            "hotspot_id": int(cid),
            "centroid_lat": round(centroid_lat, 6),
            "centroid_lon": round(centroid_lon, 6),
            "complaint_count": complaint_count,
            "dominant_category": dominant_category,
            "category_distribution": category_distribution,
            "unresolved_ratio": round(unresolved_ratio, 4),
            "wards_covered": wards_covered,
            "density_score": density_score,
            "risk_level": None,   # filled by classify_hotspot_risk()
        })

    # Sort by complaint count descending
    hotspots.sort(key=lambda h: h["complaint_count"], reverse=True)

    logger.info(f"Detected {len(hotspots)} hotspots")
    return hotspots


def classify_hotspot_risk(hotspots: list[dict]) -> list[dict]:
    """
    Assign risk levels to hotspots based on complaint count and unresolved ratio.

    Combines:
      - Normalized complaint count (weight 0.6)
      - Unresolved ratio (weight 0.4)

    Risk levels:
      High   → combined score >= 0.7
      Medium → combined score >= 0.4
      Low    → combined score <  0.4

    Args:
        hotspots: list of hotspot dicts from detect_hotspots()

    Returns:
        Same list with 'risk_level' and 'risk_score' populated
    """
    if not hotspots:
        return hotspots

    counts = np.array([h["complaint_count"] for h in hotspots], dtype=float)
    unresolved = np.array(
        [h["unresolved_ratio"] if not np.isnan(h["unresolved_ratio"]) else 0.0
         for h in hotspots],
        dtype=float,
    )

    # Normalize counts to [0, 1]
    max_count = counts.max()
    norm_counts = counts / max_count if max_count > 0 else counts

    risk_scores = 0.6 * norm_counts + 0.4 * unresolved

    for i, h in enumerate(hotspots):
        score = float(risk_scores[i])
        if score >= 0.7:
            level = "High"
        elif score >= 0.4:
            level = "Medium"
        else:
            level = "Low"
        h["risk_score"] = round(score, 4)
        h["risk_level"] = level

    high = sum(1 for h in hotspots if h["risk_level"] == "High")
    med  = sum(1 for h in hotspots if h["risk_level"] == "Medium")
    low  = sum(1 for h in hotspots if h["risk_level"] == "Low")
    logger.info(f"Hotspot risk classification: High={high}, Medium={med}, Low={low}")

    return hotspots


def detect_emerging_hotspots(
    df: pd.DataFrame,
    recent_days: int = 30,
    eps_meters: float = 500.0,
    min_samples: int = 3,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    timestamp_col: str = "created_at",
    category_col: str = "complaint_category",
) -> list[dict]:
    """
    Detect emerging hotspots by running DBSCAN only on recent complaints.

    A cluster is 'emerging' if it appears in recent data but has
    significantly more complaints than the historical average for that area.

    Args:
        df: preprocessed complaints DataFrame with timestamp column
        recent_days: window in days to define 'recent'
        eps_meters: DBSCAN neighbourhood radius
        min_samples: minimum points to form a cluster
        lat_col: latitude column
        lon_col: longitude column
        timestamp_col: datetime column
        category_col: complaint category column

    Returns:
        List of emerging hotspot dicts with 'surge_ratio' added
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col])

    cutoff = df[timestamp_col].max() - pd.Timedelta(days=recent_days)
    recent_df = df[df[timestamp_col] >= cutoff]

    if len(recent_df) < min_samples:
        logger.warning(f"Only {len(recent_df)} recent complaints — not enough for emerging detection.")
        return []

    logger.info(f"Detecting emerging hotspots from {len(recent_df)} complaints in last {recent_days} days")

    recent_hotspots = detect_hotspots(
        recent_df,
        eps_meters=eps_meters,
        min_samples=min_samples,
        lat_col=lat_col,
        lon_col=lon_col,
        category_col=category_col,
    )
    recent_hotspots = classify_hotspot_risk(recent_hotspots)

    # Compute surge ratio: recent cluster count vs historical rate
    total_days = max((df[timestamp_col].max() - df[timestamp_col].min()).days, 1)
    historical_daily_rate = len(df) / total_days
    recent_daily_rate = len(recent_df) / recent_days
    global_surge = recent_daily_rate / max(historical_daily_rate, 1e-6)

    for h in recent_hotspots:
        h["surge_ratio"] = round(global_surge, 4)
        h["is_emerging"] = global_surge > 1.2   # 20% surge threshold

    emerging = [h for h in recent_hotspots if h.get("is_emerging", False)]
    logger.info(f"Emerging hotspots (surge > 1.2x): {len(emerging)} / {len(recent_hotspots)}")

    return recent_hotspots


def hotspots_to_dataframe(hotspots: list[dict]) -> pd.DataFrame:
    """
    Convert hotspot list to a flat DataFrame for reporting or API output.

    Explodes category_distribution into separate columns and
    flattens wards_covered to a semicolon-joined string.
    """
    if not hotspots:
        return pd.DataFrame()

    rows = []
    for h in hotspots:
        row = {k: v for k, v in h.items() if k not in ("category_distribution", "wards_covered")}
        row["wards_covered"] = ";".join(str(w) for w in h.get("wards_covered", []))
        row["top_category"] = h.get("dominant_category", "Unknown")
        rows.append(row)

    return pd.DataFrame(rows)


def run_hotspot_pipeline(
    df: pd.DataFrame,
    eps_meters: float = 500.0,
    min_samples: int = 5,
    recent_days: int = 30,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    category_col: str = "complaint_category",
    status_col: str = "complaint_status_title",
    ward_col: str = "ward_id",
) -> dict:
    """
    Full hotspot detection pipeline. Returns a summary dict with:
      - all_hotspots: list of all detected hotspots with risk levels
      - emerging_hotspots: list of recent-surge hotspots
      - hotspot_df: flat DataFrame of all hotspots
      - summary: counts by risk level

    Args:
        df: preprocessed complaints DataFrame
        eps_meters: DBSCAN neighbourhood radius in metres
        min_samples: minimum complaints to form a cluster
        recent_days: window for emerging hotspot detection
        lat_col: latitude column
        lon_col: longitude column
        category_col: standardized category column
        status_col: complaint status column
        ward_col: ward identifier column
    """
    logger.info("=" * 50)
    logger.info("HOTSPOT DETECTION PIPELINE")
    logger.info("=" * 50)

    # All-time hotspots
    hotspots = detect_hotspots(
        df,
        eps_meters=eps_meters,
        min_samples=min_samples,
        lat_col=lat_col,
        lon_col=lon_col,
        category_col=category_col,
        status_col=status_col,
        ward_col=ward_col,
    )
    hotspots = classify_hotspot_risk(hotspots)

    # Emerging hotspots
    emerging = detect_emerging_hotspots(
        df,
        recent_days=recent_days,
        eps_meters=eps_meters,
        min_samples=max(min_samples - 2, 2),
        lat_col=lat_col,
        lon_col=lon_col,
        category_col=category_col,
    )

    hotspot_df = hotspots_to_dataframe(hotspots)

    summary = {
        "total_hotspots": len(hotspots),
        "high_risk": sum(1 for h in hotspots if h["risk_level"] == "High"),
        "medium_risk": sum(1 for h in hotspots if h["risk_level"] == "Medium"),
        "low_risk": sum(1 for h in hotspots if h["risk_level"] == "Low"),
        "emerging_count": sum(1 for h in emerging if h.get("is_emerging", False)),
        "noise_points": int((df.get("hotspot_cluster", pd.Series([-1] * len(df))) == -1).sum()),
    }

    logger.info(
        f"Hotspot pipeline complete: {summary['total_hotspots']} total | "
        f"High={summary['high_risk']} | Medium={summary['medium_risk']} | "
        f"Low={summary['low_risk']} | Emerging={summary['emerging_count']}"
    )

    return {
        "all_hotspots": hotspots,
        "emerging_hotspots": emerging,
        "hotspot_df": hotspot_df,
        "summary": summary,
    }
