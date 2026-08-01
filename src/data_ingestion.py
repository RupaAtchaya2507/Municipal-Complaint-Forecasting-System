"""
Data Ingestion Module
====================
Load complaint, weather, and festival datasets and merge them
into a single unified DataFrame.

Handles the actual complaints.csv format:
  - Encoding: latin-1 (Windows-1252)
  - Date format: M/D/YYYY H:MM
  - 17 columns including ward_id, title, description, etc.
"""

import os
import pandas as pd
import numpy as np
import logging
import requests
import config

logger = logging.getLogger(__name__)


def load_complaints(path: str, encoding: str = "latin-1") -> pd.DataFrame:
    """
    Load complaint dataset from CSV.

    Actual columns in the CSV:
      created_at, ward_id, title, description, sub_category_id,
      civic_agency_id, location, address, latitude, longitude,
      ward_title, category_id, category_title, sub_category_title,
      civic_agency_title, complaint_status_title, comment_count

    Returns DataFrame with 'created_at' parsed as datetime.
    """
    logger.info(f"Loading complaints from: {path}")
    df = pd.read_csv(path, encoding=encoding)

    # Parse timestamp — format is "M/D/YYYY H:MM" (e.g., "1/1/2019 6:33")
    df["created_at"] = pd.to_datetime(
        df["created_at"],
        format="mixed",
        dayfirst=False,
        errors="coerce",
    )

    # Drop rows where timestamp parsing failed
    bad_dates = df["created_at"].isna().sum()
    if bad_dates > 0:
        logger.warning(f"Dropped {bad_dates} rows with unparseable dates")
        df = df.dropna(subset=["created_at"])

    # Extract date column for merging with weather/festivals
    df["date"] = df["created_at"].dt.date

    # Handle category_id NaN (32 rows have null category_id)
    if df["category_id"].isna().any():
        null_count = df["category_id"].isna().sum()
        df["category_id"] = df["category_id"].fillna(0).astype(int)
        logger.info(f"Filled {null_count} null category_id values with 0")
    else:
        df["category_id"] = df["category_id"].astype(int)

    logger.info(f"Loaded {len(df)} complaints, date range: "
                f"{df['created_at'].min()} to {df['created_at'].max()}")
    logger.info(f"Statuses: {df['complaint_status_title'].value_counts().to_dict()}")

    return df


def load_weather(path: str) -> pd.DataFrame:
    """
    Load weather dataset from CSV.

    Expected columns: date (or timestamp), temperature, rainfall, humidity

    Returns DataFrame with 'date' parsed as date.
    Returns None if file does not exist.
    """
    if not os.path.exists(path):
        logger.warning(f"Weather file not found: {path}. Skipping weather data.")
        return None

    logger.info(f"Loading weather data from: {path}")
    df = pd.read_csv(path)

    # Normalize date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    elif "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date
        df.drop(columns=["timestamp"], inplace=True)
    else:
        raise ValueError("Weather CSV must contain 'date' or 'timestamp' column")

    # Keep only relevant columns
    weather_cols = ["date", "temperature", "rainfall", "humidity"]
    available = [c for c in weather_cols if c in df.columns]
    df = df[available]

    # Aggregate to daily (in case of multiple readings per day)
    df = df.groupby("date", as_index=False).mean(numeric_only=True)

    logger.info(f"Loaded weather data: {len(df)} days")
    return df


def fetch_weather_api(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch weather data covering start_date to end_date.

    Strategy:
      - Historical dates (before today) → Open-Meteo Archive API
      - Today / future dates            → Open-Meteo Forecast API
    Both results are merged into a single DataFrame.
    """
    from datetime import date as date_type
    today = date_type.today()
    start = pd.to_datetime(start_date).date()
    end   = pd.to_datetime(end_date).date()

    frames = []

    # ── Historical portion (archive API) ──
    archive_end = min(end, today - pd.Timedelta(days=1))
    if start <= archive_end:
        logger.info(f"Fetching archive weather {start} → {archive_end}")
        params = {
            "latitude":   config.WEATHER_API_LAT,
            "longitude":  config.WEATHER_API_LON,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date":   archive_end.strftime("%Y-%m-%d"),
            "daily":      "temperature_2m_mean,rain_sum,relative_humidity_2m_mean",
            "timezone":   "auto",
        }
        try:
            resp = requests.get(config.WEATHER_API_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                logger.error(f"Archive API error: {data.get('reason')}")
            else:
                daily = data["daily"]
                frames.append(pd.DataFrame({
                    "date":        pd.to_datetime(daily["time"]).date,
                    "temperature": daily["temperature_2m_mean"],
                    "rainfall":    daily["rain_sum"],
                    "humidity":    daily.get("relative_humidity_2m_mean",
                                             [0.0] * len(daily["time"])),
                }))
                logger.info(f"Archive: fetched {len(frames[-1])} days")
        except Exception as e:
            logger.error(f"Archive weather fetch failed: {e}")

    # ── Today / forecast portion (forecast API) ──
    forecast_start = max(start, today)
    if forecast_start <= end:
        days_ahead = (end - today).days + 1
        days_ahead = min(days_ahead, 7)   # forecast API supports up to 16 days
        logger.info(f"Fetching forecast weather today + {days_ahead} days")
        forecast_url = getattr(config, "WEATHER_FORECAST_API_URL",
                               "https://api.open-meteo.com/v1/forecast")
        params = {
            "latitude":         config.WEATHER_API_LAT,
            "longitude":        config.WEATHER_API_LON,
            "daily":            "temperature_2m_mean,rain_sum,relative_humidity_2m_mean",
            "forecast_days":    days_ahead,
            "timezone":         "auto",
        }
        try:
            resp = requests.get(forecast_url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                logger.error(f"Forecast API error: {data.get('reason')}")
            else:
                daily = data["daily"]
                frames.append(pd.DataFrame({
                    "date":        pd.to_datetime(daily["time"]).date,
                    "temperature": daily["temperature_2m_mean"],
                    "rainfall":    daily["rain_sum"],
                    "humidity":    daily.get("relative_humidity_2m_mean",
                                             [0.0] * len(daily["time"])),
                }))
                logger.info(f"Forecast: fetched {len(frames[-1])} days")
        except Exception as e:
            logger.error(f"Forecast weather fetch failed: {e}")

    if not frames:
        logger.error("No weather data fetched from any source")
        return None

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="date").sort_values("date")
    logger.info(f"Total weather fetched: {len(df)} days ({df['date'].min()} → {df['date'].max()})")
    return df


def load_festivals(path: str) -> pd.DataFrame:
    """
    Load festival dataset from CSV.

    Expected columns: date, festival_flag (0/1), festival_name (optional)

    Returns DataFrame with 'date' parsed as date.
    Returns None if file does not exist.
    """
    if not os.path.exists(path):
        logger.warning(f"Festival file not found: {path}. Skipping festival data.")
        return None

    logger.info(f"Loading festival data from: {path}")
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    # Ensure festival_flag exists
    if "festival_flag" not in df.columns:
        df["festival_flag"] = 1  # If all rows are festivals

    # Keep relevant columns
    cols = ["date", "festival_flag"]
    if "festival_name" in df.columns:
        cols.append("festival_name")
    df = df[cols]

    logger.info(f"Loaded {len(df)} festival entries")
    return df


def merge_datasets(
    complaints: pd.DataFrame,
    weather: pd.DataFrame = None,
    festivals: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Merge complaint, weather, and festival data into a unified DataFrame.

    Weather and festival DataFrames are optional — if None, those features
    are filled with defaults (0 for festival_flag, NaN for weather).
    """
    logger.info("Merging datasets...")
    df = complaints.copy()

    # Merge weather by date (if available)
    if weather is not None:
        df = df.merge(weather, on="date", how="left")
        # Fill missing weather with forward-fill then backward-fill
        weather_cols = ["temperature", "rainfall", "humidity"]
        for col in weather_cols:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()
        logger.info("Merged weather data")
    else:
        # Create placeholder weather columns with 0
        df["temperature"] = 0.0
        df["rainfall"] = 0.0
        df["humidity"] = 0.0
        logger.info("No weather data — using zeros as placeholder")

    # Merge festivals by date (if available)
    if festivals is not None:
        df = df.merge(festivals[["date", "festival_flag"]], on="date", how="left")
        df["festival_flag"] = df["festival_flag"].fillna(0).astype(int)
        logger.info("Merged festival data")
    else:
        df["festival_flag"] = 0
        logger.info("No festival data — using 0 as placeholder")

    logger.info(f"Merged dataset: {len(df)} rows, {len(df.columns)} columns")
    return df


def ingest_all(
    complaints_path: str,
    weather_path: str = None,
    festivals_path: str = None,
    encoding: str = "latin-1",
) -> pd.DataFrame:
    """
    Full ingestion pipeline: load all sources and merge.

    Weather and festival paths are optional — if files don't exist,
    the pipeline proceeds with complaint data only.

    Returns unified DataFrame.
    """
    complaints = load_complaints(complaints_path, encoding=encoding)

    weather = None
    if getattr(config, "USE_WEATHER_API", False):
        if not complaints.empty:
            from datetime import date
            today = date.today()
            min_date = complaints["date"].min().strftime("%Y-%m-%d")
            # Fetch up to today — fetch_weather_api handles archive vs forecast split
            end_date = today.strftime("%Y-%m-%d")
            weather  = fetch_weather_api(min_date, end_date)
            if weather is not None:
                logger.info(
                    f"Weather fetched for {min_date} to {end_date} "
                    f"({len(weather)} days). Future complaint dates will use "
                    f"forward-filled values from the latest available weather."
                )
        else:
            logger.warning("No complaints data to determine weather API date range")
    elif weather_path:
        weather = load_weather(weather_path)

    festivals = None
    if festivals_path:
        festivals = load_festivals(festivals_path)

    return merge_datasets(complaints, weather, festivals)
