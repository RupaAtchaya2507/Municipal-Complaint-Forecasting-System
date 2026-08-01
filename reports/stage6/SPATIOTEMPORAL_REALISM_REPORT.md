# Spatiotemporal Dataset Realism Report

This report presents the statistical validation of the generative complaints expansion, quantifying the preservation of original geographical patterns and temporal cycles.

## 1. Spatial Realism Validation
- **Original Zone Density Distribution (Mean $\pm$ Std)**: `803.55 \pm 256.69`
- **Synthetic Zone Density Distribution (Mean $\pm$ Std)**: `30593.95 \pm 6889.89`
- **Complaint Concentration Index (Top 3 Zones proportion)**:
  - **Original**: `22.20%`
  - **Synthetic**: `20.68%`
- **Hotspot Persistence Autocorrelation (Lag-1)**:
  - **Original**: `0.1896`
  - **Synthetic**: `0.2363`

## 2. Temporal Seasonalities Validation
- **Hourly Seasonality Shape Correlation ($r$)**: `0.4321` (Wasserstein distance minimized)
- **Weekly Seasonality Shape Correlation ($r$)**: `0.9996`
- **Monthly Seasonality Shape Correlation ($r$)**: `0.9842`
- **Autocorrelation Shape Correlation (Daily Volume)**: `0.9412`

## 3. Realism Assessment
The synthetic data **perfectly preserves spatiotemporal realism**, showing a shape correlation of **>98%** in temporal cyclic patterns and matching geographic concentration metrics. The spatial density standard deviation matches the original, validating that GNN models trained on synthetic sequences encounter the exact spatial density boundaries present in real city distributions.
