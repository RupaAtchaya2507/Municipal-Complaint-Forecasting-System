"""
Real-Time Data Scheduler
========================
Runs a background job every 24 hours to:
  1. Fetch latest weather from Open-Meteo API
  2. Reload complaint data (new rows appended to CSV)
  3. Re-run feature engineering on fresh data
  4. Refresh model predictions and risk scores

Start automatically when the FastAPI server starts.
"""

import logging
import threading
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Weather Fetcher
# ──────────────────────────────────────────────

def fetch_latest_weather(lat: float, lon: float, days_back: int = 7) -> pd.DataFrame:
    """
    Fetch recent + today's weather.

    - Past days  → Open-Meteo Archive API
    - Today      → Open-Meteo Forecast API
    Both are merged into a single DataFrame.
    """
    today      = date.today()
    yesterday  = today - timedelta(days=1)
    start_date = yesterday - timedelta(days=days_back - 1)

    frames = []

    # ── Historical (archive) ──
    archive_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date":   yesterday.strftime("%Y-%m-%d"),
        "daily":      "temperature_2m_mean,rain_sum,relative_humidity_2m_mean",
        "timezone":   "auto",
    }
    try:
        resp = requests.get(archive_url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        daily = data["daily"]
        frames.append(pd.DataFrame({
            "date":        pd.to_datetime(daily["time"]).date,
            "temperature": daily["temperature_2m_mean"],
            "rainfall":    daily["rain_sum"],
            "humidity":    daily.get("relative_humidity_2m_mean",
                                     [0.0] * len(daily["time"])),
        }))
        logger.info(f"Archive weather: {len(frames[-1])} days fetched")
    except Exception as e:
        logger.error(f"Archive weather fetch failed: {e}")

    # ── Today (forecast API) ──
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":      lat,
        "longitude":     lon,
        "daily":         "temperature_2m_mean,rain_sum,relative_humidity_2m_mean",
        "forecast_days": 1,
        "timezone":      "auto",
    }
    try:
        resp = requests.get(forecast_url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        daily = data["daily"]
        frames.append(pd.DataFrame({
            "date":        pd.to_datetime(daily["time"]).date,
            "temperature": daily["temperature_2m_mean"],
            "rainfall":    daily["rain_sum"],
            "humidity":    daily.get("relative_humidity_2m_mean",
                                     [0.0] * len(daily["time"])),
        }))
        logger.info(f"Today's forecast weather fetched ({today})")
    except Exception as e:
        logger.error(f"Forecast weather fetch failed: {e}")

    if not frames:
        logger.error("No weather data fetched")
        return None

    df = (pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset="date")
            .sort_values("date"))
    logger.info(f"Fetched {len(df)} days of weather (up to {today})")
    return df


# ──────────────────────────────────────────────
# Refresh Job
# ──────────────────────────────────────────────

def refresh_pipeline(app_state):
    """
    Full refresh cycle:
      1. Fetch latest weather
      2. Merge into existing complaint data
      3. Re-run feature engineering
      4. Re-run inference and update risk scores
    """
    logger.info("=" * 50)
    logger.info(f"SCHEDULED REFRESH — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)

    import config

    try:
        # Step 1: Fetch latest weather
        weather_df = fetch_latest_weather(
            lat=config.WEATHER_API_LAT,
            lon=config.WEATHER_API_LON,
            days_back=7,
        )

        # Step 2: Merge weather into complaint DataFrame
        if weather_df is not None and app_state.df_complaints is not None:
            df = app_state.df_complaints.copy()

            # Ensure date column exists
            if "date" not in df.columns:
                df["date"] = pd.to_datetime(df["created_at"]).dt.date

            # Drop old weather columns and re-merge
            for col in ["temperature", "rainfall", "humidity"]:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)

            df = df.merge(weather_df, on="date", how="left")

            # Forward-fill missing weather values
            for col in ["temperature", "rainfall", "humidity"]:
                if col in df.columns:
                    df[col] = df[col].ffill().bfill().fillna(0.0)

            app_state.df_complaints = df
            logger.info("Weather data merged into complaint DataFrame")

        # Step 3: Re-run feature engineering on updated data
        from src.aggregation import create_time_windows, aggregate_by_zone_window, fill_missing_windows
        from src.features import feature_pipeline
        from src.dataset import create_sequences

        df = app_state.df_complaints.copy()
        df = create_time_windows(df, config.TIME_WINDOW_HOURS)
        agg_df = aggregate_by_zone_window(df)
        agg_df = fill_missing_windows(agg_df, app_state.num_zones)

        feature_tensor, feature_names, _, agg_df_featured = feature_pipeline(
            agg_df, app_state.num_zones
        )

        # Update state with fresh features
        app_state.feature_tensor  = feature_tensor
        app_state.feature_names   = feature_names
        app_state.num_features    = feature_tensor.shape[2]
        app_state.agg_df_featured = agg_df_featured

        # Rebuild sequences
        X, y_msi = create_sequences(
            feature_tensor,
            seq_len=config.DEFAULT_SEQ_LEN,
            adjacency_matrix=app_state.adj_matrix,
            scaling_method=getattr(config, "SCALING_METHOD", "robust"),
            predict_delta=False,
        )
        app_state.X_sequences = X
        app_state.y_msi       = y_msi

        # Step 4: Re-run inference with fresh data
        app_state._run_inference()

        logger.info(f"Refresh complete — predictions updated for {app_state.num_zones} zones")
        logger.info(f"Latest risk: High={sum(1 for r in app_state.last_risk_results if r['risk_level']=='High')}, "
                    f"Medium={sum(1 for r in app_state.last_risk_results if r['risk_level']=='Medium')}, "
                    f"Low={sum(1 for r in app_state.last_risk_results if r['risk_level']=='Low')}")

    except Exception as e:
        logger.error(f"Refresh cycle failed: {e}", exc_info=True)


# ──────────────────────────────────────────────
# Background Scheduler
# ──────────────────────────────────────────────

class DataScheduler:
    """
    Runs refresh_pipeline() on a fixed interval in a background thread.
    Default interval: every 24 hours.
    """

    def __init__(self, app_state, interval_hours: int = 24):
        self.app_state      = app_state
        self.interval_secs  = interval_hours * 3600
        self._thread        = None
        self._stop_event    = threading.Event()

    def start(self):
        """Start the background scheduler thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"Scheduler started — refresh every {self.interval_secs // 3600}h")

    def stop(self):
        """Stop the background scheduler."""
        self._stop_event.set()
        logger.info("Scheduler stopped")

    def _run(self):
        """Main scheduler loop — waits then refreshes."""
        while not self._stop_event.is_set():
            # Wait for the interval (checking stop every 60s)
            for _ in range(self.interval_secs // 60):
                if self._stop_event.is_set():
                    return
                time.sleep(60)

            if not self._stop_event.is_set():
                refresh_pipeline(self.app_state)

    def trigger_now(self):
        """Manually trigger a refresh immediately (for testing)."""
        thread = threading.Thread(target=refresh_pipeline, args=(self.app_state,), daemon=True)
        thread.start()
        logger.info("Manual refresh triggered")
