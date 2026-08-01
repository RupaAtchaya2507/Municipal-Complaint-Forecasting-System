# Project Directory Structure

Below is the complete tree structure of all files and folders in the project directory, excluding virtual environment (`venv`) and compiler/test cache directories:

```text
project/
├── LONG_HORIZON_ROLLING_BASELINE_REPORT.md
├── archive
│   ├── plans
│   │   ├── implementation_plan_legacy.md
│   │   └── task_legacy.md
│   ├── scratch_discover_static_features.py
│   ├── scratch_reconstruction_analysis.py
│   ├── scratch_test_reconstruction_alignment.py
│   ├── spatiotemporal.zip
│   ├── test_output.log
│   ├── tmp_analyze.py
│   ├── tmp_analyze_large.py
│   ├── tmp_api_test.py
│   └── tmp_festivals.py
├── baseline_formulation_comparison.csv
├── config.py
├── data
│   ├── complaints.csv
│   ├── custom_100_complaints.csv
│   ├── festivals.csv
│   ├── synthetic_aggregated.csv
│   ├── synthetic_complaints.csv
│   ├── synthetic_features.npy
│   └── zone_static_features.csv
├── diagnostics
│   ├── actual_msi_distribution.csv
│   ├── delta_msi_temporal_analysis.csv
│   ├── delta_msi_zone_statistics.csv
│   ├── diagnostic_alternative_formulations.csv
│   ├── diagnostic_analysis.csv
│   ├── diagnostic_contributions.csv
│   ├── diagnostic_correlations.csv
│   ├── diagnostic_zone3_investigation.csv
│   ├── feature_importance.csv
│   ├── gradient_diagnostics.csv
│   ├── hidden_state_diagnostics.csv
│   ├── msi_zone_diagnostics.csv
│   ├── msi_zone_diagnostics_root.csv
│   ├── ranking_quality.csv
│   ├── seq_len_benchmark.csv
│   ├── top_zone_comparison.csv
│   ├── zone_explanations.csv
│   └── zone_prediction_error.csv
├── final_evaluation_report.md
├── final_production_pipeline.md
├── generate_synthetic_data.py
├── hotspot_forecasting_analysis.csv
├── images
│   ├── RESOURCE_ALLOCATION_VISUALS.png
│   ├── learning_curves.png
│   ├── risk_assessment.png
│   ├── spatial_clusters.png
│   └── synthetic_validation.png
├── long_horizon_sequence_audit.csv
├── main.py
├── models
│   └── best_model.pt
├── msi_zone_diagnostics.csv
├── outputs
│   ├── BASELINE_COMPARISON.csv
│   ├── RESOURCE_ALLOCATION_RESULTS.csv
│   ├── baseline_comparisons_difficulty.csv
│   ├── baseline_formulation_comparison.csv
│   ├── baseline_target_comparison.csv
│   ├── baseline_variance_analysis.csv
│   ├── hotspot_forecasting_analysis.csv
│   ├── long_horizon_sequence_audit.csv
│   ├── resource_allocation_comparison.csv
│   └── sequence_length_audit.csv
├── paper
│   ├── drafts
│   ├── figures
│   ├── references
│   ├── supplementary
│   └── tables
├── project.md
├── reports
│   ├── final_evaluation_report.md
│   ├── final_model_selection_walkthrough.md
│   ├── final_stage2_report.md
│   ├── final_stage3_investigation_report.md
│   ├── stage3
│   │   ├── LARGE_DATASET_READINESS_REPORT.md
│   │   ├── PRETRAINING_READINESS_REPORT.md
│   │   └── final_evaluation_report_root.md
│   ├── stage4
│   │   ├── ABSOLUTE_MSI_IMPROVEMENT_REPORT.md
│   │   ├── BASELINE_COMPARISON_REPORT.md
│   │   ├── DELTA_RECONSTRUCTION_AUDIT.md
│   │   ├── MODEL_COMPLEXITY_REPORT.md
│   │   ├── RESOURCE_ALLOCATION_REPORT.md
│   │   ├── SPATIAL_INFORMATION_AUDIT.md
│   │   ├── STATIC_FEATURE_ANALYSIS.md
│   │   └── STATIC_FEATURE_EXPERIMENT_REPORT.md
│   ├── stage5
│   │   ├── BASELINE_DECOMPOSITION_ANALYSIS.md
│   │   └── RESIDUAL_FORECASTING_REPORT.md
│   └── stage6
│       ├── DATASET_DIFFICULTY_REPORT.md
│       ├── DATASET_LINEAGE_REPORT.md
│       ├── EVENT_DYNAMICS_REPORT.md
│       ├── GRAPH_SIGNAL_ANALYSIS.md
│       ├── PERSISTENCE_ANALYSIS.md
│       ├── SPATIOTEMPORAL_REALISM_REPORT.md
│       └── SYNTHETIC_DATA_AUDIT_MASTER_REPORT.md
├── requirements.txt
├── research
│   ├── pretraining_validation.py
│   ├── run_baseline_benchmarks.py
│   ├── run_final_selection_experiment.py
│   ├── run_long_horizon_residual_optimization.py
│   ├── run_residual_evaluation.py
│   ├── run_resource_evaluation.py
│   ├── run_spatial_audit.py
│   ├── run_stage2_experiments.py
│   ├── run_stage3_investigation.py
│   └── run_static_features_experiment.py
├── resource_allocation_comparison.csv
├── src
│   ├── __init__.py
│   ├── aggregation.py
│   ├── clustering.py
│   ├── data_ingestion.py
│   ├── dataset.py
│   ├── features.py
│   ├── inference.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── risk_engine.py
│   ├── synthetic_generator.py
│   ├── train.py
│   ├── utils.py
│   └── visualization.py
├── tests
│   ├── __init__.py
│   ├── test_clustering.py
│   ├── test_dataset.py
│   ├── test_features.py
│   ├── test_model.py
│   ├── test_preprocessing.py
│   ├── test_risk_engine.py
│   └── test_synthetic_generator.py
└── zone_static_features.csv
```
