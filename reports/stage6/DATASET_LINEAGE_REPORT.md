# Synthetic Dataset Lineage Audit Report

## 1. Data Source & Synthesis Lineage
- **Original Dataset Size**: `16,071 complaints` (`data/complaints.csv`)
- **Synthetic Dataset Size**: `611,879 complaints` (`data/synthetic_complaints.csv`)
- **Real Records Used**: `16,071 (100% of real prior records)`
- **Synthetic Records Synthesized**: `611,879 (100% generative probabilistic expansion)`

## 2. Generative Algorithm Description
The expansion pipeline utilizes the **`SpatioTemporalSyntheticGenerator`** to learn conditional and joint probabilities from original prior distributions, preserving city bounds, daily schedules, monthly trends, and weather/festival surges:
1. **Spatial Coordinates Pattern**: Center zone Gaussian coordinate sampling centered around KMeans spatial centroids. Hotspot variance is mathematically expanded during heavy rain simulations to model geographic incident disperse.
2. **Poisson Arrival Rates**: Models window rate $\lambda_{t, z}$ as: 
   $$\lambda_{t, z} = \text{BaseRate} \times P(\text{zone}_z) \times \text{SeasonalityMultiplier} \times \text{WeatherMultiplier} \times \text{FestivalMultiplier}$$
3. **Behavioral Recurrence Augmentation**: Duplicate report simulation. Backlogged unresolved queues increase Poisson rates dynamically by up to 30%, modeling repeat reporting behavior.
4. **Adjacency-Aware Graph Spillover**: Distributes `15%` of estimated local Poisson rates to contiguous geographic neighbors using the normalized $K$-nearest neighbor adjacency graph.
5. **Temporal Micro-distributions**: Uniform time window interpolation, ensuring complaints are distributed naturally across 6-hour windows.

## 3. Columns Synthesis Strategy
| Column Name | Source Type | Generation Mechanism |
|:---|:---:|:---|
| `created_at` | **Synthesized** | Generated probabilistically from learned prior conditionals |
| `latitude` | **Synthesized** | Generated probabilistically from learned prior conditionals |
| `longitude` | **Synthesized** | Generated probabilistically from learned prior conditionals |
| `category_id` | **Synthesized** | Generated probabilistically from learned prior conditionals |
| `complaint_status_title` | **Synthesized** | Generated probabilistically from learned prior conditionals |
| `comment_count` | **Synthesized** | Generated probabilistically from learned prior conditionals |
| `ward_id` | **Copied (Mapped)** | Mapped from matching historical records based on `(zone_id, category_id)` lookup |
| `title` | **Copied (Mapped)** | Mapped from matching historical records based on `(zone_id, category_id)` lookup |
| `description` | **Copied (Mapped)** | Mapped from matching historical records based on `(zone_id, category_id)` lookup |
| `sub_category_id` | **Copied (Mapped)** | Mapped from matching historical records based on `(zone_id, category_id)` lookup |
| `civic_agency_id` | **Copied (Mapped)** | Mapped from matching historical records based on `(zone_id, category_id)` lookup |
| `location` | **Copied (Mapped)** | Mapped from matching historical records based on `(zone_id, category_id)` lookup |
| `address` | **Copied (Mapped)** | Mapped from matching historical records based on `(zone_id, category_id)` lookup |
| `ward_title` | **Copied (Mapped)** | Mapped from matching historical records based on `(zone_id, category_id)` lookup |
| `category_title` | **Copied (Mapped)** | Mapped from matching historical records based on `(zone_id, category_id)` lookup |
| `sub_category_title` | **Copied (Mapped)** | Mapped from matching historical records based on `(zone_id, category_id)` lookup |
| `civic_agency_title` | **Copied (Mapped)** | Mapped from matching historical records based on `(zone_id, category_id)` lookup |
| `date` | **Copied (Mapped)** | Mapped from matching historical records based on `(zone_id, category_id)` lookup |
| `temperature` | **Copied (Mapped)** | Mapped from matching historical records based on `(zone_id, category_id)` lookup |
| `rainfall` | **Copied (Mapped)** | Mapped from matching historical records based on `(zone_id, category_id)` lookup |
| `humidity` | **Copied (Mapped)** | Mapped from matching historical records based on `(zone_id, category_id)` lookup |
| `festival_flag` | **Copied (Mapped)** | Mapped from matching historical records based on `(zone_id, category_id)` lookup |
