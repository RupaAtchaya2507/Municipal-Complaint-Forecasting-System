# Event Dynamics Audit Report

This report dissects the presence of dynamic, high-frequency municipal shocks and event bursts in the synthetic complaints dataset.

## 1. Municipal Shock Quantification
- **Sudden Surge Frequency (Surge Rate)**: `0.4224% of zone-windows` (surges exceed mean + 3 std)
- **Co-occurring Neighbor Surges (Multi-Zone Spillovers)**: `0.00% of surges` trigger co-surges in adjacent zones.
- **Extreme Stress Periods (Global Stress Shocks)**: `0.00% of windows` experience a city-wide stress surge.

## 2. Municipal Events Assessment
The dataset **contains spatiotemporal event bursts** (represented by local bursts convolved with neighbors). However, because the generative process relies heavily on Poisson sampling under stable baseline zone probabilities, the overall spatiotemporal structure is highly **regular and stationary**. Extreme city-wide stress shocks are scarce, and local surges quickly decay back to the historical mean. The absence of massive, multi-zone spillovers or sudden structural shifts (such as a zone completely migrating from low-risk to high-risk) limits the model's ability to demonstrate the full potential of complex spatiotemporal GNN tracking.
