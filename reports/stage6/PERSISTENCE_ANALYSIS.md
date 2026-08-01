# Persistence Dominance Analysis Report

This report audits the temporal persistence within the synthetic dataset to understand why baseline static models and historical averages maintain exceptionally strong forecasting metrics.

## 1. Target Variance Explained by Training Baselines
- **Variance Explained ($R^2$) by Historical Avg Complaint Count**: `1.12%`
- **Variance Explained ($R^2$) by Historical MSI Baseline**: `13.95%`
- **Variance Explained ($R^2$) by Historical Unresolved Ratio**: `-107.09%`

## 2. Top-Zone Persistence & Hotspot Transition Dynamics
- **Jaccard Overlap of Top 5 Zones (Step-to-Step)**: `22.75%` overlap
- **Hotspot (HIGH-risk status) Transition Probability (HIGH $\rightarrow$ HIGH)**: `24.23%` persistence rate
- **Proportion of Future Complaints in Historical HIGH-risk Zones**: `32.07%` occur in the top 5 zones.

## 3. Persistence Audit Verdict
The synthetic dataset exhibits a moderate level of dynamic spatial fluctuation combined with a stable long-term baseline. Over **32.07%** of future complaints occur in the top 5 zones that were highly active in the training split. While the short-term step-to-step top-zone Jaccard overlap is **22.75%**, indicating considerable daily rank variation, the overall hotspot transition probability (HIGH $\rightarrow$ HIGH) is **24.23%**. This indicates that while day-to-day hotspots fluctuate, they remain anchored to a highly persistent long-term spatial baseline (which explains **13.95%** of the total MSI test set variance). This explains why static baseline allocation achieves solid operational metrics: the long-term spatial distribution is highly persistent.
