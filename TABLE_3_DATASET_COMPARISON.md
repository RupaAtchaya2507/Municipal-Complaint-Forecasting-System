# Table 3: Dataset Comparison (Original vs. Synthetic)

This table compares the statistical, temporal, spatial, and feature properties of the original municipal dataset against the probabilistic expanded synthetic dataset.

| Metric | Original Dataset | Synthetic Dataset |
| :--- | :---: | :---: |
| **Number of complaints** | 16,071 | 611,879 |
| **Temporal span** | 2019-01-01 to 2022-07-31 (~3.5 years) | 2019-01-01 to 2026-12-31 (8.0 years) |
| **Total Calendar Days** | 1,308 days | 2,922 days |
| **Incident Categories** | 709 slots | 709 slots |
| **Spatial Zones ($K$)** | 20 | 20 |
| **Weather features** | 3 (Temperature, Rainfall, Humidity) | 3 (Temperature, Rainfall, Humidity) |
| **Festival features** | 2 (Festival Flag, Festival Eve) | 2 (Festival Flag, Festival Eve) |
| **Resolution states** | 2 (Open, Resolved) | 2 (Open, Resolved) |
| **Average complaints/day** | 12.29 | 209.40 |
| **Unresolved backlog ratio** | 60.01% | 58.74% |
| **Expansion ratio** | 1.0x (Baseline) | 38.07x (Probabilistic expansion) |
| **GNN Continuous Timesteps ($T$)** | ~1,300 daily windows | 2,919 daily windows |
| **Hotspot Concentration Index** | 22.20% (Top 3 zones) | 20.68% (Top 3 zones) |
| **Lag-1 Hotspot Autocorrelation** | 0.1896 | 0.2363 |
| **Moran's I Spatial Autocorrelation** | 0.1676 | 0.1676 |
| **Neighbor MSI Correlation** | 0.4942 | 0.4942 (Modeled with 15% spillover) |
| **Sudden Surge Frequency** | N/A | 0.4224% of zone-windows |
| **Verification Status** | Real prior baseline | 100% GNN-consistent & validated |
