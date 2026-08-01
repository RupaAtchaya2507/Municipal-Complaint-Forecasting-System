"""
Unit Tests for SpatioTemporal Synthetic Data Generator
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from src.synthetic_generator import SpatioTemporalSyntheticGenerator

@pytest.fixture
def sample_complaints_data():
    """Create a sample historical complaints dataframe for fitting."""
    # 20 complaints over 4 days
    rng = np.random.default_rng(42)
    dates = pd.to_datetime([
        "2024-01-15 08:30", "2024-01-15 12:00", "2024-01-15 18:45",
        "2024-01-16 02:15", "2024-01-16 10:00", "2024-01-16 14:30",
        "2024-01-17 09:00", "2024-01-17 11:30", "2024-01-17 21:00",
        "2024-01-18 04:00", "2024-01-18 13:15", "2024-01-18 16:45",
        "2024-01-15 09:30", "2024-01-16 15:00", "2024-01-17 18:45",
        "2024-01-18 02:15", "2024-01-15 10:00", "2024-01-16 14:30",
        "2024-01-17 09:00", "2024-01-18 11:30"
    ])
    
    # 2 zones, 3 categories
    df = pd.DataFrame({
        "created_at": dates,
        "latitude": [12.91 + rng.normal(0, 0.005) for _ in range(20)],
        "longitude": [77.61 + rng.normal(0, 0.005) for _ in range(20)],
        "zone_id": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "category_id": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2],
        "complaint_status_title": [
            "Open", "Resolved", "On-the-Job", "Closed", "Open", "Resolved",
            "Open", "Resolved", "On-the-Job", "Closed", "Open", "Resolved",
            "Open", "Resolved", "On-the-Job", "Closed", "Open", "Resolved",
            "Open", "Resolved"
        ],
        # Add metadata fields
        "ward_id": [1] * 20,
        "title": ["Sample Title"] * 20,
        "description": ["Sample Description"] * 20,
        "sub_category_id": [10] * 20,
        "civic_agency_id": [2.0] * 20,
        "location": ["Sample Location"] * 20,
        "address": ["Sample Address"] * 20,
        "ward_title": ["Sample Ward"] * 20,
        "category_title": ["Sample Category"] * 20,
        "sub_category_title": ["Sample Subcategory"] * 20,
        "civic_agency_title": ["BBMP"] * 20,
        "comment_count": [0] * 20
    })
    return df

@pytest.fixture
def sample_weather_data():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18"]).date,
        "temperature": [24.0, 26.0, 25.0, 23.0],
        "rainfall": [0.0, 2.0, 8.0, 0.0],
        "humidity": [60.0, 65.0, 80.0, 55.0]
    })

@pytest.fixture
def sample_festivals_data():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-16"]).date,
        "festival_flag": [1],
        "festival_name": ["Sample Festival"]
    })

class TestSyntheticGeneratorCore:
    def test_fit_and_fitted_state(self, sample_complaints_data, sample_weather_data, sample_festivals_data):
        gen = SpatioTemporalSyntheticGenerator(random_seed=42)
        assert not gen.is_fitted
        
        gen.fit(sample_complaints_data, sample_weather_data, sample_festivals_data)
        assert gen.is_fitted
        
        # Check spatial centroids learned
        assert len(gen.zone_centroids) == 2
        assert 0 in gen.zone_centroids
        assert 1 in gen.zone_centroids
        
        # Check temporal hours prob shape
        assert len(gen.temporal_hours_prob) == 24
        
        # Check weather multipliers
        assert gen.weather_multipliers["rain_none"] == 1.0
        assert gen.weather_multipliers["rain_heavy"] > 0
        
        # Check category status probabilities learned
        assert 1 in gen.category_status_prob
        # Status "Open" or "On-the-Job" are unresolved (value 1)
        # Category 1 has 7 items, 4 of which are unresolved ("Open")
        assert abs(gen.category_status_prob[1] - 4/7) < 1e-6

    def test_generate_bounds_and_validity(self, sample_complaints_data, sample_weather_data, sample_festivals_data):
        gen = SpatioTemporalSyntheticGenerator(random_seed=42)
        gen.fit(sample_complaints_data, sample_weather_data, sample_festivals_data)
        
        # Generate 100 synthetic complaints over 2 days
        synth_df = gen.generate(
            start_date="2024-01-15",
            end_date="2024-01-16",
            target_records=100,
            temporal_augmentation=False,
            spatial_augmentation=False,
            behavioral_augmentation=False,
            weather_df=sample_weather_data,
            festivals_df=sample_festivals_data
        )
        
        assert len(synth_df) > 0
        
        # Coordinates must fall within city bounds
        min_lat, max_lat = sample_complaints_data["latitude"].min(), sample_complaints_data["latitude"].max()
        min_lon, max_lon = sample_complaints_data["longitude"].min(), sample_complaints_data["longitude"].max()
        
        assert (synth_df["latitude"] >= min_lat).all()
        assert (synth_df["latitude"] <= max_lat).all()
        assert (synth_df["longitude"] >= min_lon).all()
        assert (synth_df["longitude"] <= max_lon).all()
        
        # Check all 17 metadata columns exist
        required_cols = [
            "created_at", "ward_id", "title", "description", "sub_category_id",
            "civic_agency_id", "location", "address", "latitude", "longitude",
            "ward_title", "category_id", "category_title", "sub_category_title",
            "civic_agency_title", "complaint_status_title", "comment_count"
        ]
        for col in required_cols:
            assert col in synth_df.columns

    def test_adjacency_smoothing_rates(self, sample_complaints_data):
        # 2 zones
        gen = SpatioTemporalSyntheticGenerator(random_seed=42)
        gen.fit(sample_complaints_data)
        
        # Adjacency matrix: connected
        adj = np.array([[0, 1], [1, 0]], dtype=float)
        
        synth_df = gen.generate(
            start_date="2024-01-15",
            end_date="2024-01-15",
            target_records=50,
            adjacency_matrix=adj,
            spatial_smoothing_eta=0.5, # 50% rate smoothing
            temporal_augmentation=False,
            spatial_augmentation=False,
            behavioral_augmentation=False
        )
        
        assert len(synth_df) > 0

    def test_validation_computations(self, sample_complaints_data, sample_weather_data, sample_festivals_data):
        gen = SpatioTemporalSyntheticGenerator(random_seed=42)
        gen.fit(sample_complaints_data, sample_weather_data, sample_festivals_data)
        
        synth_df = gen.generate(
            start_date="2024-01-15",
            end_date="2024-01-16",
            target_records=200,
            weather_df=sample_weather_data,
            festivals_df=sample_festivals_data
        )
        
        metrics = gen.validate(sample_complaints_data, synth_df)
        
        assert "wasserstein_latitude" in metrics
        assert "wasserstein_longitude" in metrics
        assert "kl_divergence_hour_of_day" in metrics
        assert "kl_divergence_categories" in metrics
        assert "real_unresolved_ratio" in metrics
        
        assert metrics["wasserstein_latitude"] >= 0
        assert metrics["kl_divergence_hour_of_day"] >= 0
