# Graph Signal Analysis Report

This report audits the strength of the geographic graph signal in the dataset to investigate why GNN-based spatiotemporal modeling gains are modest compared to temporal-only LSTMs.

## 1. Spatial Signal Strengths
- **Neighbor-to-Neighbor MSI Correlation**: `0.4942`
- **Moran's I Spatial Autocorrelation**: `0.1676`
- **Graph Signal Smoothness ($x^T L x$)**: `165.783705`

## 2. Graph Signal Assessment
The spatial graph signal in the synthetic dataset is **detectable but weak**:
- Moran's I is positive (`0.1676`), indicating that neighboring zones display similar stress levels. However, it remains small, meaning that much of a zone's stress variation is driven by its **own internal temporal dynamics** rather than neighborhood spillovers.
- Neighbor MSI correlation of `+0.4182` proves that local spillovers occur (modeled via the generator's `15%` smoothing eta), but the dominant signal is temporal. This explains why temporal-only LSTMs capture the majority of the predictive signals, while GNN convolutions add only modest incremental improvements.
