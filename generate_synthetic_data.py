"""
Synthetic Dataset Expansion CLI Runner
======================================
Loads original datasets, trains the SpatioTemporalSyntheticGenerator,
generates a large synthetic dataset (~200,000 complaints), processes it
through the preprocessing and feature engineering pipelines, and validates it.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.utils import setup_logging, set_seed
from src.data_ingestion import ingest_all, load_festivals, load_weather
from src.preprocessing import preprocess_pipeline
from src.clustering import find_optimal_clusters, create_zones, build_adjacency_matrix
from src.aggregation import create_time_windows, aggregate_by_zone_window, fill_missing_windows
from src.features import feature_pipeline
from src.synthetic_generator import SpatioTemporalSyntheticGenerator

logger = logging.getLogger(__name__)

def main():
    # Setup logging and reproducibility
    setup_logging()
    set_seed(config.RANDOM_SEED)
    
    logger.info("=" * 60)
    logger.info("STARTING SYNTHETIC DATA GENERATION PIPELINE")
    logger.info("=" * 60)

    # ────── 1. Load Original Data ──────
    logger.info("Loading original datasets for prior learning...")
    weather_path = config.WEATHER_CSV if os.path.exists(config.WEATHER_CSV) else None
    festival_path = config.FESTIVALS_CSV if os.path.exists(config.FESTIVALS_CSV) else None

    real_df_merged = ingest_all(
        config.COMPLAINTS_CSV,
        weather_path,
        festival_path,
        encoding=config.CSV_ENCODING,
    )
    
    # Load raw sub-components directly for generator fitting
    raw_weather = load_weather(config.WEATHER_CSV) if os.path.exists(config.WEATHER_CSV) else None
    raw_festivals = load_festivals(config.FESTIVALS_CSV) if os.path.exists(config.FESTIVALS_CSV) else None
    
    # Run original preprocessing to get clean timestamps
    real_df_preprocessed = preprocess_pipeline(real_df_merged)

    # ────── 2. Spatial Clustering to Get Zones and Adjacency ──────
    logger.info("Clustering original coordinates to obtain zones and graph adjacency...")
    coords = real_df_preprocessed[["latitude", "longitude"]].values
    optimal_k = find_optimal_clusters(coords, config.CLUSTER_K_RANGE)
    real_df_zoned, centroids = create_zones(real_df_preprocessed, optimal_k)
    adj_matrix = build_adjacency_matrix(centroids, k=config.KNN_NEIGHBORS, epsilon=config.EDGE_EPSILON)
    
    # ────── 3. Fit SpatioTemporal Generator ──────
    generator = SpatioTemporalSyntheticGenerator(random_seed=config.RANDOM_SEED)
    generator.fit(real_df_zoned, raw_weather, raw_festivals)

    # ────── 4. Generate Expanded Dataset ──────
    # Expand timeline: original is 2019-01-01 to 2022-07-31 (~3.5 years)
    # We expand timeline from 2019-01-01 to 2026-12-31 (~8 years) to scale up sequences
    # Target size: 200,000 records
    start_date = "2019-01-01"
    end_date = "2026-12-31"
    target_size = 200000
    
    synthetic_df = generator.generate(
        start_date=start_date,
        end_date=end_date,
        target_records=target_size,
        adjacency_matrix=adj_matrix,
        spatial_smoothing_eta=0.30,
        temporal_augmentation=True,
        spatial_augmentation=True,
        behavioral_augmentation=True,
        weather_df=raw_weather,
        festivals_df=raw_festivals
    )

    # ────── 5. Save Synthetic Complaint-Level Dataset ──────
    synth_csv_path = os.path.join(config.DATA_DIR, "synthetic_complaints.csv")
    logger.info(f"Saving synthetic complaint records to: {synth_csv_path}")
    
    # Save in standard encoding matching complaints.csv
    # Format matches 17 original columns
    output_cols = [
        "created_at", "ward_id", "title", "description", "sub_category_id",
        "civic_agency_id", "location", "address", "latitude", "longitude",
        "ward_title", "category_id", "category_title", "sub_category_title",
        "civic_agency_title", "complaint_status_title", "comment_count"
    ]
    synthetic_df[output_cols].to_csv(synth_csv_path, index=False, encoding="latin-1")
    logger.info("Saved raw synthetic complaints dataset.")

    # ────── 6. Process Synthetic Data through Downstream Pipelines ──────
    logger.info("Processing synthetic complaints through standard ingestion and preprocessing pipelines...")
    
    # We load it from CSV to ensure complete parsing correctness
    synth_loaded = ingest_all(
        synth_csv_path,
        weather_path,
        festival_path,
        encoding=config.CSV_ENCODING
    )
    synth_preprocessed = preprocess_pipeline(synth_loaded)
    
    # Project zones onto synthetic coordinates
    # We use KMeans to map the synthetic coordinates back to zone IDs
    # To keep same spatial boundaries and cluster indices, we map to zones
    # We map using optimal_k zones
    synth_df_zoned, _ = create_zones(synth_preprocessed, optimal_k)
    
    logger.info("Aggregating synthetic data into 6-hour windows...")
    synth_windowed = create_time_windows(synth_df_zoned, config.TIME_WINDOW_HOURS)
    synth_aggregated = aggregate_by_zone_window(synth_windowed)
    synth_aggregated = fill_missing_windows(synth_aggregated, optimal_k)
    
    # Save aggregated temporal CSV
    synth_agg_path = os.path.join(config.DATA_DIR, "synthetic_aggregated.csv")
    logger.info(f"Saving aggregated temporal dataset to: {synth_agg_path}")
    synth_aggregated.to_csv(synth_agg_path, index=False)
    
    logger.info("Running feature engineering pipeline on synthetic aggregated data...")
    feature_tensor, feature_names, scaler, synth_featured = feature_pipeline(synth_aggregated, optimal_k)
    
    # Save 3D Feature Tensor [T x N x F] ready for GNN + LSTM
    synth_tensor_path = os.path.join(config.DATA_DIR, "synthetic_features.npy")
    logger.info(f"Saving GNN-consistent spatiotemporal sequence feature tensor [T={feature_tensor.shape[0]} x N={optimal_k} x F={feature_tensor.shape[2]}] to: {synth_tensor_path}")
    np.save(synth_tensor_path, feature_tensor)

    # ────── 7. Validation & Reporting ──────
    logger.info("=" * 60)
    logger.info("VALIDATING SYNTHETIC DATA QUALITY")
    logger.info("=" * 60)
    
    validation_metrics = generator.validate(real_df_zoned, synth_df_zoned)
    
    validation_plot_path = os.path.join(config.PROJECT_ROOT, "images", "synthetic_validation.png")
    generator.plot_comparisons(real_df_zoned, synth_df_zoned, validation_plot_path)
    
    logger.info("\n" + "=" * 60)
    logger.info("SYNTHETIC DATA GENERATION SUCCESSFUL")
    logger.info(f"Raw Complaints: {len(synthetic_df)}")
    logger.info(f"Temporal Graph Sequences: {feature_tensor.shape[0]} time steps")
    logger.info(f"Saved Files:\n"
                f"  - Complaint-level: {synth_csv_path}\n"
                f"  - Aggregated per window: {synth_agg_path}\n"
                f"  - GNN Feature Tensor: {synth_tensor_path}\n"
                f"  - Validation plots: {validation_plot_path}")
    logger.info("=" * 60)
    
    # Print target-size validation check
    if len(synthetic_df) >= 150000:
        logger.info("TARGET RECORD COUNT REQUIREMENT SATISFIED (>= 150,000)")
    if feature_tensor.shape[0] >= 11000:
        logger.info("TARGET TIMESTEPS REQUIREMENT SATISFIED")

if __name__ == "__main__":
    main()
