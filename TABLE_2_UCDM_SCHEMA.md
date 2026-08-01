# Table 2: Unified Urban Complaint Data Model (UCDM) Schema

This table presents the detailed spatiotemporal schema of the Unified Urban Complaint Data Model (UCDM), mapping each input feature and target variable to its operational category, underlying data source, and technical description.

| Feature Name | Category | Source | Description |
| :--- | :---: | :---: | :--- |
| **`complaint_count`** | Dynamic | Complaint | Raw daily volume of incoming service complaints logged within the zone boundary. |
| **`unresolved_count`** | Dynamic | Complaint | Active backlog of unresolved complaints (open, on-the-job, or re-opened) at the end of the window. |
| **`resolved_count`** | Dynamic | Complaint | Raw volume of service complaints successfully resolved or closed within the daily window. |
| **`U`** | Dynamic | Derived | MinMax-normalized ratio of unresolved outstanding service burden: $\frac{\text{unresolved\_count}}{\text{unresolved\_count} + \text{resolved\_count} + 1}$. |
| **`D`** | Dynamic | Derived | MinMax-normalized daily complaint volume, representing raw density. |
| **`delta_density`** | Dynamic | Derived | First-difference of complaint density between consecutive daily steps ($D_{t} - D_{t-1}$). |
| **`rolling_avg_density`** | Dynamic | Derived | Moving average of daily complaint density computed over a 3-day sliding temporal window. |
| **`3_day_complaint_avg`** | Dynamic | Derived | Rolling average of daily incoming complaint volumes computed over a 3-day window. |
| **`7_day_complaint_avg`** | Dynamic | Derived | Rolling average of daily incoming complaint volumes computed over a 7-day window. |
| **`3_day_unresolved_avg`** | Dynamic | Derived | Rolling average of daily outstanding unresolved backlogs over a 3-day window. |
| **`7_day_unresolved_avg`** | Dynamic | Derived | Rolling average of daily outstanding unresolved backlogs over a 7-day window. |
| **`complaint_velocity`** | Dynamic | Derived | First-difference of daily complaint counts, capturing raw stress acceleration. |
| **`days_since_last_complaint`** | Dynamic | Derived | Temporal persistence counter tracking consecutive days since a complaint was logged (capped at 999.0). |
| **`days_since_last_open_complaint`** | Dynamic | Derived | Temporal persistence counter tracking consecutive days since an open backlog was active (capped at 999.0). |
| **`neighbor_complaint_avg`** | Dynamic | Graph | Dynamic spatial spillover calculated as the mean daily complaint volume across adjacent spatial graph neighbors. |
| **`neighbor_unresolved_avg`** | Dynamic | Graph | Dynamic spatial spillover backlog calculated as the mean unresolved count across adjacent spatial graph neighbors. |
| **`hist_avg_complaint_count`** | Static | Derived | Long-term historical average of daily complaint volumes logged in the zone. |
| **`hist_var_complaint_count`** | Static | Derived | Historical variance of daily complaint counts, capturing volumetric volatility. |
| **`hist_avg_unresolved_ratio`** | Static | Derived | Historical average ratio of unresolved complaints, identifying chronically backlogged zones. |
| **`hist_resolution_rate`** | Static | Derived | Long-term average rate of daily complaint resolutions, measuring historical efficiency. |
| **`hist_avg_msi`** | Static | Derived | Historical average of the Municipal Stress Index (MSI) computed for the zone. |
| **`hist_var_msi`** | Static | Derived | Long-term historical variance of daily MSI values. |
| **`hist_complaint_density`** | Static | Derived | Historical spatial density of complaints, anchoring localized geographical intensity offsets. |
| **`hist_avg_neighbor_pressure`** | Static | Graph | Historical average neighbor pressure convolved from adjacent spatial graph neighbors. |
| **`hist_var_neighbor_pressure`** | Static | Graph | Variance of historical spatiotemporal neighborhood pressure. |
| **`hist_avg_growth_rate`** | Static | Derived | Historical average of day-over-day complaint growth rate. |
| **`hist_var_growth_rate`** | Static | Derived | Variance of historical growth rates, indicating surge predictability. |
| **`hour_of_day`** | External | Derived | Aggregation window starting hour (0–23), capturing daily cyclic patterns. |
| **`day_of_week`** | External | Derived | Day index of the week (0 = Monday, 6 = Sunday), capturing weekly cyclic patterns. |
| **`is_weekend`** | External | Derived | Binary flag indicating if the window corresponds to a weekend day (Saturday/Sunday). |
| **`month`** | External | Derived | Calendar month index (1–12), capturing seasonal and monthly administrative cycles. |
| **`is_festival_eve`** | External | Festival | Binary flag indicating if the daily window immediately precedes a major cultural holiday. |
| **`temperature`** | External | Weather | Daily mean ambient temperature in Celsius, capturing seasonal meteorological stress. |
| **`rainfall`** | External | Weather | Daily total rain precipitation in millimeters, capturing weather-induced incident triggers. |
| **`humidity`** | External | Weather | Daily mean relative humidity percentage, capturing local climatic conditions. |
| **`festival_flag`** | External | Festival | Binary indicator flagging the occurrence of major cultural and public festivals. |
| **`MSI`** | Target | Derived | Continuous target index measuring dynamic zone strain at prediction horizon $h$: $0.35 C_{\text{norm}} + 0.30 U_{\text{norm}} + 0.20 G_{\text{norm}} + 0.15 N_{\text{norm}}$. |
| **`Delta MSI`** | Target | Derived | Target measuring directional change in MSI between current and future windows ($MSI_{t+h} - MSI_t$). |
| **`Residual MSI`** | Target | Derived | Target measuring residual stress variance after subtracting baseline forecasts ($MSI_{t+h} - \text{Baseline}_t$). |
| **`Binary High-Risk Flag`** | Target | Derived | Auxiliary target flagging if future MSI belongs to the top 40% of non-zero values for active risk alerts. |
