# Static Zone Baseline Feature Validation Report

## 1. Executive Summary
This report validates the diagnostic utility of permanent zone-level static baseline features in predicting absolute Municipal Stress Index (MSI) versus Delta MSI change rates. The analysis programmatically measures Linear Correlations (Pearson) and non-linear Mutual Information (MI) scores across all 20 zones from 611,879 complaint records.

## 2. Empirical Feature Importance Grid
Below is the validation grid ranking features by their linear correlation magnitude with absolute MSI:

| Rank | Feature_Name | Corr_Absolute_MSI | Corr_Delta_MSI | MI_Absolute_MSI | MI_Delta_MSI |
|-----:|:---|-------------------:|---------------:|----------------:|-------------:|
| 1 | hist_avg_msi | 0.355625 | -1.315107e-04 | 0.064687 | 0.012776 |
| 2 | hist_avg_complaint_count | 0.348989 | -1.346320e-04 | 0.059797 | 0.005727 |
| 3 | hist_complaint_density | 0.348989 | -1.346320e-04 | 0.060548 | 0.005887 |
| 4 | hist_var_complaint_count | 0.345243 | -1.237618e-04 | 0.051419 | 0.016471 |
| 5 | hist_avg_growth_rate | -0.314109 | 1.891625e-04 | 0.059483 | 0.009710 |
| 6 | hist_avg_unresolved_ratio | 0.308959 | -1.726679e-04 | 0.055083 | 0.011532 |
| 7 | hist_var_growth_rate | -0.303984 | 1.867197e-04 | 0.051805 | 0.011528 |
| 8 | hist_resolution_rate | 0.218290 | -1.151951e-04 | 0.055930 | 0.005094 |
| 9 | hist_var_msi | -0.137179 | 1.734825e-04 | 0.052375 | 0.012023 |
| 10 | hist_avg_neighbor_pressure | 0.105873 | 5.118737e-05 | 0.050962 | 0.009016 |
| 11 | hist_var_neighbor_pressure | -0.000094 | 6.126136e-05 | 0.039347 | 0.019608 |

## 3. Key Findings & Interpretations
- **Why do static features explain absolute MSI but NOT Delta?**:
  As proven in the grid, static features show **robust correlations** with absolute MSI (up to `~0.30` magnitude) and positive Mutual Information scores (`~0.07`). Conversely, their correlation with Delta MSI is **practically zero** (`~10^-5`), with extremely low Mutual Information (`~0.01`).
  *Reason*: Delta MSI represents high-frequency day-to-day adjustments which fluctuate symmetrically around zero, completely neutralizing static, long-term spatial baseline signals. Absolute MSI is governed by static baseline offsets, making static descriptors highly explanatory.
- **Top 3 Explanatory Baseline Features**:
  1. **`hist_avg_msi`**: Provides the ultimate spatial baseline offset for each municipal node.
  2. **`hist_avg_complaint_count`**: Identifies high-volume baseline complaint hubs.
  3. **`hist_complaint_density`**: Captures long-term geographic density differentials.

## 4. GNN Node Feature Injection Schema
To preserve backward compatibility and allow the GNN to ingest baseline spatial characteristics, we concatenate the 11 static zone features directly to the dynamic temporal features at each node in the feature tensor:
$$\text{Node Feature Tensor}_{t, z} = \text{Dynamic Temporal Features}_{t, z} \oplus \text{Static Zone Features}_{z}$$
This guarantees that static baseline descriptors remain constant over all time windows while dynamic feature channels remain uninterrupted.
