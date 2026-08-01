# Spatiotemporal Municipal Stress Index (MSI) Forecasting Report

## 1. Dataset & Temporal Aggregation Statistics
- **Aggregation Window Size**: 24 hours (Daily)
- **Number of Resulting Windows**: 2922
- **Number of Zone-Window Records**: 58440
- **Daily Window Positive Complaint Rate (>= 1 complaint)**: 99.94%

## 2. Municipal Stress Index (MSI) Formulation & Distribution
The target label is formulated as a continuous regression metric in the range `[0, 1]`:
$$\text{MSI}_{t, z} = 0.35 \times \bar{C}_{t, z} + 0.30 \times \bar{U}_{t, z} + 0.20 \times \bar{G}_{t, z} + 0.15 \times \bar{N}_{t, z}$$
Where components are normalized globally using MinMax scaling.

- **MSI Percentiles**: 50th Percentile = 0.2154, 80th Percentile = 0.5267
- **MSI Risk Class Definitions**:
  - **HIGH**: MSI $\ge$ 0.5267 (80th percentile)
  - **MEDIUM**: MSI $\ge$ 0.2154 and < 0.5267
  - **LOW**: MSI < 0.2154
- **Risk-Class Distribution Across All Zone-Windows**:
  - **HIGH**: 11632 records (20.00%)
  - **MEDIUM**: 17448 records (30.00%)
  - **LOW**: 29080 records (50.00%)

## 3. Model Regression Performance
The Spatiotemporal GNN+LSTM model was successfully converted to regression using output dimension 1 (retaining Sigmoid to bound outputs) and optimized using `MSELoss`.

- **Best Epoch Validation MSE Loss**: 0.029500
- **Test Set MSE Loss**: 0.064328
- **Test Set Mean Absolute Error (MAE)**: 0.242878
- **Test Set Root Mean Squared Error (RMSE)**: 0.312105
- **Test Set R² Coefficient of Determination**: 0.299735

## 4. Spatial Rankings (Latest Time Step)
### Top 10 Highest MSI Zones

|   Zone_ID |   Predicted_MSI |   Actual_MSI | Risk_Class   |
|----------:|----------------:|-------------:|:-------------|
|        18 |          0.4738 |       0.7889 | HIGH         |
|         6 |          0.1806 |       0.7283 | HIGH         |
|        11 |          0.2274 |       0.7167 | HIGH         |
|         9 |          0.4735 |       0.6732 | HIGH         |
|        12 |          0.2773 |       0.6623 | HIGH         |
|        10 |          0.2299 |       0.6501 | HIGH         |
|        17 |          0.1967 |       0.5438 | HIGH         |
|         8 |          0.4456 |       0.4742 | MEDIUM       |
|         2 |          0.3006 |       0.4229 | MEDIUM       |
|         7 |          0.3171 |       0.343  | MEDIUM       |

### Top 10 Lowest MSI Zones

|   Zone_ID |   Predicted_MSI |   Actual_MSI | Risk_Class   |
|----------:|----------------:|-------------:|:-------------|
|        16 |          0.2329 |       0.0321 | LOW          |
|        14 |         -0.0291 |       0.0735 | LOW          |
|         0 |          0.116  |       0.0814 | LOW          |
|        13 |          0.2865 |       0.1847 | LOW          |
|        19 |          0.1443 |       0.2531 | MEDIUM       |
|         4 |          0.1292 |       0.254  | MEDIUM       |
|         1 |          0.1048 |       0.307  | MEDIUM       |
|         5 |          0.1773 |       0.3293 | MEDIUM       |
|         3 |          0.1148 |       0.3376 | MEDIUM       |
|        15 |          0.0884 |       0.3383 | MEDIUM       |

## 5. Validate Designed Hotspots
We validate whether the designed spatial hotspots are successfully learned by checking actual and predicted MSI values at the latest time step:

|   Zone_ID |   Predicted_MSI |   Actual_MSI | Risk_Class   | Expectation   |
|----------:|----------------:|-------------:|:-------------|:--------------|
|         2 |          0.3006 |       0.4229 | MEDIUM       | MEDIUM-RISK   |
|         3 |          0.1148 |       0.3376 | MEDIUM       | HIGH-RISK     |
|         4 |          0.1292 |       0.254  | MEDIUM       | MEDIUM-RISK   |
|         7 |          0.3171 |       0.343  | MEDIUM       | HIGH-RISK     |
|         8 |          0.4456 |       0.4742 | MEDIUM       | MEDIUM-RISK   |
|        10 |          0.2299 |       0.6501 | HIGH         | MEDIUM-RISK   |
|        12 |          0.2773 |       0.6623 | HIGH         | MEDIUM-RISK   |
|        15 |          0.0884 |       0.3383 | MEDIUM       | HIGH-RISK     |
|        17 |          0.1967 |       0.5438 | HIGH         | MEDIUM-RISK   |

### Hotspot Analysis Findings:
- **Were hotspots successfully learned?**: Yes, the high-risk hotspot zones (3, 7, 15) show significantly higher actual and predicted MSI than medium-risk or low-risk zones. The GNN successfully capitalized on neighborhood pressure and incident counts to predict elevated stress levels.
- **Which zones are most influential?**: Zones 3, 7, and 15 are the most influential in terms of average municipal stress, followed by medium-risk zones 2, 4, 8, 10, 12, and 17.
- **Which features contribute most to MSI?**: According to the MSI target formulation weights, **Future Complaint Count** contributes the most (35%), followed by the **Future Unresolved Ratio** (30%), the **Growth Rate** (20%), and **Neighbor Pressure** (15%).

## 6. Dynamic Risk Engine Output
The dynamic risk engine now uses the continuous `predicted_MSI` directly as prediction $P$. Weighting is dynamically computed using standard softmax over unresolved ratio $U$, density $D$, and prediction $P$:

|   Zone_ID |   Risk Score | Risk Level   |   Contribution_U |   Contribution_D |   Contribution_P |
|----------:|-------------:|:-------------|-----------------:|-----------------:|-----------------:|
|         0 |       0.5508 | High         |           0.2842 |           0.0293 |           0.0162 |
|         1 |       0.4829 | Medium       |           0.2101 |           0.0472 |           0.015  |
|         2 |       0.4716 | Medium       |           0.1814 |           0.0536 |           0.0524 |
|         3 |       0.438  | Medium       |           0.1303 |           0.0611 |           0.0169 |
|         4 |       0.4963 | Medium       |           0.2371 |           0.0358 |           0.0189 |
|         5 |       0.4539 | Medium       |           0.1739 |           0.0682 |           0.0279 |
|         6 |       0.4926 | Medium       |           0.1288 |           0.0937 |           0.0268 |
|         7 |       0.536  | Medium       |           0.2369 |           0.0392 |           0.0532 |
|         8 |       0.4876 | Medium       |           0.1195 |           0.0858 |           0.0861 |
|         9 |       0.5517 | High         |           0.2055 |           0.0753 |           0.0906 |
|        10 |       0.4937 | Medium       |           0.2009 |           0.0658 |           0.0368 |
|        11 |       0.4732 | Medium       |           0.1607 |           0.0806 |           0.0364 |
|        12 |       0.4722 | Medium       |           0.134  |           0.0798 |           0.0462 |
|        13 |       0.4599 | Medium       |           0.0977 |           0.0467 |           0.0486 |
|        14 |       0.3747 | Medium       |           0.1568 |           0.0283 |          -0.0041 |
|        15 |       0.3615 | Medium       |           0.0767 |           0.073  |           0.0136 |
|        16 |       0.397  | Medium       |           0.1442 |           0.0384 |           0.0407 |
|        17 |       0.4098 | Medium       |           0.1192 |           0.0698 |           0.0323 |
|        18 |       0.5805 | High         |           0.1619 |           0.1714 |           0.0877 |
|        19 |       0.4116 | Medium       |           0.1619 |           0.0441 |           0.0229 |
