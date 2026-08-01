"""
Tests for Feature Engineering Module
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from src.features import (
    compute_derived_features,
    normalize_features,
    build_feature_tensor,
)


@pytest.fixture
def agg_df():
    """Aggregated DataFrame for testing."""
    return pd.DataFrame({
        "time_window": pd.to_datetime([
            "2024-01-15 00:00", "2024-01-15 00:00",
            "2024-01-15 06:00", "2024-01-15 06:00",
        ]),
        "zone_id": [0, 1, 0, 1],
        "complaint_count": [10, 5, 8, 12],
        "unresolved_count": [7, 2, 5, 10],
        "resolved_count": [3, 3, 3, 2],
        "hour_of_day": [0, 0, 6, 6],
        "day_of_week": [0, 0, 0, 0],
        "is_weekend": [0, 0, 0, 0],
        "month": [1, 1, 1, 1],
        "is_festival_eve": [0, 0, 0, 0],
        "temperature": [25.0, 25.0, 28.0, 28.0],
        "rainfall": [0.0, 0.0, 2.0, 2.0],
        "humidity": [60.0, 60.0, 70.0, 70.0],
        "festival_flag": [0, 0, 0, 0],
    })


class TestDerivedFeatures:
    def test_u_raw_range(self, agg_df):
        result = compute_derived_features(agg_df)
        assert "U_raw" in result.columns
        assert result["U_raw"].min() >= 0
        assert result["U_raw"].max() <= 1

    def test_d_raw_range(self, agg_df):
        result = compute_derived_features(agg_df)
        assert "D_raw" in result.columns
        assert result["D_raw"].min() >= 0
        # Assert D_raw matches the raw complaint count
        pd.testing.assert_series_equal(result["D_raw"].sort_index(), agg_df["complaint_count"].astype(float).sort_index(), check_names=False)

    def test_u_raw_formula(self, agg_df):
        result = compute_derived_features(agg_df)
        row = result.iloc[0]
        expected = row["unresolved_count"] / (
            row["unresolved_count"] + row["resolved_count"] + 1
        )
        assert abs(result["U_raw"].iloc[0] - expected) < 1e-6


class TestNormalization:
    def test_normalized_range(self, agg_df):
        df = compute_derived_features(agg_df)
        df_norm, scaler = normalize_features(df, ["U_raw", "D_raw"])
        assert df_norm["U"].min() >= 0
        assert df_norm["U"].max() <= 1
        assert df_norm["D"].min() >= 0
        assert df_norm["D"].max() <= 1


class TestFeatureTensor:
    def test_tensor_shape(self, agg_df):
        df = compute_derived_features(agg_df)
        df, _ = normalize_features(df, ["U_raw", "D_raw"])
        feature_cols = ["complaint_count", "unresolved_count", "U", "D"]
        tensor, names = build_feature_tensor(df, num_zones=2, feature_cols=feature_cols)
        assert tensor.shape == (2, 2, 4)  # 2 time windows, 2 zones, 4 features

    def test_no_nan_in_tensor(self, agg_df):
        df = compute_derived_features(agg_df)
        df, _ = normalize_features(df, ["U_raw", "D_raw"])
        tensor, _ = build_feature_tensor(df, num_zones=2)
        assert not np.isnan(tensor).any()
