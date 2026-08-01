# Baseline Stress Decomposition Audit

This document presents the spatial variance decomposition of the Municipal Stress Index (MSI) across all 20 zones, validating the necessity of separating long-term geographical baselines from short-term deviations.

## 1. Global Variance & Decomposition Metrics

- **Total Test Set Variance**: `0.177873`
- **Variance Explained by Constant Baseline ($R^2$)**: `10.03%`
- **Residual Variance Proportion**: `89.88%`
- **Baseline Lag-1 Autocorrelation (Persistence)**: `-0.1128`
- **Baseline Lag-7 Autocorrelation**: `0.1357`
- **Baseline-to-Future Pearson Correlation**: `0.3182`

## 2. Zone-Specific Baseline Characterization

| Zone ID | Historical Mean MSI | Historical Median MSI | 30-Day Rolling MSI | 80th Percentile MSI |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 0.0968 | 0.0872 | -0.0104 | 0.4402 |
| 1 | 0.1416 | 0.1380 | 0.0477 | 0.4895 |
| 2 | 0.3330 | 0.3348 | 0.3070 | 0.6429 |
| 3 | 0.1074 | 0.1099 | -0.0308 | 0.4545 |
| 4 | 0.1251 | 0.1342 | 0.0585 | 0.4728 |
| 5 | 0.2513 | 0.2525 | 0.1551 | 0.5686 |
| 6 | 0.2794 | 0.2867 | 0.2598 | 0.6044 |
| 7 | 0.3589 | 0.3650 | 0.2098 | 0.6666 |
| 8 | 0.4761 | 0.4784 | 0.4130 | 0.7764 |
| 9 | 0.3625 | 0.3613 | 0.2734 | 0.6650 |
| 10 | 0.2168 | 0.2082 | 0.1417 | 0.5482 |
| 11 | 0.1943 | 0.1866 | 0.1778 | 0.5286 |
| 12 | 0.2820 | 0.2895 | 0.2347 | 0.6073 |
| 13 | 0.2358 | 0.2434 | 0.1368 | 0.5663 |
| 14 | -0.1027 | -0.1198 | -0.0763 | 0.3184 |
| 15 | 0.1544 | 0.1490 | 0.0908 | 0.4938 |
| 16 | 0.1747 | 0.1788 | 0.1295 | 0.5115 |
| 17 | 0.2513 | 0.2587 | 0.0720 | 0.5759 |
| 18 | 0.4138 | 0.4258 | 0.2861 | 0.7039 |
| 19 | 0.0352 | 0.0357 | -0.0413 | 0.3932 |
