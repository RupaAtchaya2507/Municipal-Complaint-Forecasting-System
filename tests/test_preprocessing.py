"""
Tests for Data Preprocessing Module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.preprocessing import clean_data, extract_time_features, encode_status


@pytest.fixture
def sample_df():
    """Create a sample unified DataFrame for testing."""
    dates = [datetime(2024, 1, 15, h) for h in [8, 14, 20]]
    festival_date = datetime(2024, 1, 16).date()

    return pd.DataFrame({
        "created_at": dates,
        "latitude": [12.97, None, 12.95],
        "longitude": [77.59, 77.60, None],
        "category_id": [1, 2, 3],
        "complaint_status_title": ["Open", "Resolved", "Closed"],
        "temperature": [28.0, 30.0, 27.0],
        "rainfall": [0.0, 5.0, 0.0],
        "humidity": [65.0, 80.0, 60.0],
        "festival_flag": [0, 0, 0],
        "date": [d.date() for d in dates],
    })


@pytest.fixture
def clean_sample_df():
    """Create a sample DF with valid coords for time feature tests."""
    base = datetime(2024, 1, 15, 10)
    festival_date = datetime(2024, 1, 16).date()

    return pd.DataFrame({
        "created_at": [
            base,
            base + timedelta(days=1),  # Jan 16 = festival
            base + timedelta(days=2),
        ],
        "latitude": [12.97, 12.96, 12.95],
        "longitude": [77.59, 77.60, 77.58],
        "category_id": [1, 2, 3],
        "complaint_status_title": ["Open", "Resolved", "Open"],
        "festival_flag": [0, 1, 0],
        "date": [
            datetime(2024, 1, 15).date(),
            festival_date,
            datetime(2024, 1, 17).date(),
        ],
    })


class TestCleanData:
    def test_removes_null_coordinates(self, sample_df):
        result = clean_data(sample_df)
        # Row 1 has null lat, row 2 has null lon → only row 0 survives
        assert len(result) == 1
        assert result["latitude"].notna().all()
        assert result["longitude"].notna().all()

    def test_fills_null_categoricals(self):
        df = pd.DataFrame({
            "latitude": [12.97, 12.96],
            "longitude": [77.59, 77.60],
            "complaint_status_title": ["Open", None],
            "category_id": [1, 2],
        })
        result = clean_data(df)
        assert (result["complaint_status_title"] == "Unknown").sum() == 1

    def test_handles_null_category_id(self):
        df = pd.DataFrame({
            "latitude": [12.97, 12.96],
            "longitude": [77.59, 77.60],
            "category_id": [5.0, np.nan],
        })
        result = clean_data(df)
        assert result["category_id"].isna().sum() == 0
        assert result["category_id"].iloc[1] == 0

    def test_handles_null_civic_agency_id(self):
        df = pd.DataFrame({
            "latitude": [12.97, 12.96],
            "longitude": [77.59, 77.60],
            "civic_agency_id": [10.0, np.nan],
        })
        result = clean_data(df)
        assert result["civic_agency_id"].isna().sum() == 0
        assert result["civic_agency_id"].iloc[1] == 0


class TestExtractTimeFeatures:
    def test_hour_extraction(self, clean_sample_df):
        result = extract_time_features(clean_sample_df)
        assert "hour_of_day" in result.columns
        assert result["hour_of_day"].iloc[0] == 10

    def test_day_of_week(self, clean_sample_df):
        result = extract_time_features(clean_sample_df)
        assert "day_of_week" in result.columns
        # Jan 15, 2024 is Monday → day_of_week = 0
        assert result["day_of_week"].iloc[0] == 0

    def test_is_weekend(self, clean_sample_df):
        result = extract_time_features(clean_sample_df)
        assert "is_weekend" in result.columns
        # Monday → not weekend
        assert result["is_weekend"].iloc[0] == 0

    def test_month(self, clean_sample_df):
        result = extract_time_features(clean_sample_df)
        assert "month" in result.columns
        assert result["month"].iloc[0] == 1

    def test_is_festival_eve(self, clean_sample_df):
        result = extract_time_features(clean_sample_df)
        assert "is_festival_eve" in result.columns
        # Jan 15 → eve of Jan 16 festival → should be 1
        assert result["is_festival_eve"].iloc[0] == 1
        # Jan 16 → eve of Jan 17 (no festival) → should be 0
        assert result["is_festival_eve"].iloc[1] == 0

    def test_no_festivals_default_zero(self):
        """When no festivals exist, is_festival_eve should be 0."""
        df = pd.DataFrame({
            "created_at": [datetime(2024, 1, 15, 10)],
            "latitude": [12.97],
            "longitude": [77.59],
            "festival_flag": [0],
            "date": [datetime(2024, 1, 15).date()],
        })
        result = extract_time_features(df)
        assert result["is_festival_eve"].iloc[0] == 0


class TestEncodeStatus:
    def test_open_encoded_as_1(self):
        df = pd.DataFrame({
            "complaint_status_title": ["Open"],
            "latitude": [12.0],
            "longitude": [77.0],
        })
        result = encode_status(df)
        assert result["status_encoded"].iloc[0] == 1

    def test_resolved_encoded_as_0(self):
        df = pd.DataFrame({
            "complaint_status_title": ["Resolved"],
            "latitude": [12.0],
            "longitude": [77.0],
        })
        result = encode_status(df)
        assert result["status_encoded"].iloc[0] == 0

    def test_closed_encoded_as_0(self):
        df = pd.DataFrame({
            "complaint_status_title": ["Closed"],
            "latitude": [12.0],
            "longitude": [77.0],
        })
        result = encode_status(df)
        assert result["status_encoded"].iloc[0] == 0

    # ─── Tests for new statuses from actual CSV ───

    def test_on_the_job_encoded_as_1(self):
        df = pd.DataFrame({
            "complaint_status_title": ["On-the-Job"],
            "latitude": [12.0],
            "longitude": [77.0],
        })
        result = encode_status(df)
        assert result["status_encoded"].iloc[0] == 1

    def test_reopened_encoded_as_1(self):
        df = pd.DataFrame({
            "complaint_status_title": ["Re-opened"],
            "latitude": [12.0],
            "longitude": [77.0],
        })
        result = encode_status(df)
        assert result["status_encoded"].iloc[0] == 1

    def test_rejected_encoded_as_0(self):
        df = pd.DataFrame({
            "complaint_status_title": ["Rejected"],
            "latitude": [12.0],
            "longitude": [77.0],
        })
        result = encode_status(df)
        assert result["status_encoded"].iloc[0] == 0

    def test_all_six_statuses(self):
        """Test all 6 actual statuses from the CSV at once."""
        df = pd.DataFrame({
            "complaint_status_title": [
                "Open", "Resolved", "On-the-Job",
                "Re-opened", "Rejected", "Closed",
            ],
            "latitude": [12.0] * 6,
            "longitude": [77.0] * 6,
        })
        result = encode_status(df)
        expected = [1, 0, 1, 1, 0, 0]
        assert result["status_encoded"].tolist() == expected
