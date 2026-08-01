# Section IV: Synthetic Urban Complaint Generation Framework Summary

This document details the spatiotemporal prior learning, generative workflows, environmental conditioning, and spatial spillover propagation mechanisms embedded within the `SpatioTemporalSyntheticGenerator` pipeline. 

---

## 1. Original vs. Synthetic Dataset Characteristics

### A. Original Dataset Characteristics
* **Total Incident Records**: `16,071 complaints` (sourced from `data/complaints.csv`)
* **Temporal Coverage**: `2019-01-01` to `2022-07-31` (approximately 3.5 years of historical complaints)
* **Categories**: `709 category slots` (representing diverse service incident codes, with slot `0` reserved for unknown categories)
* **Geographical Coverage**: The municipal coordinates cluster into `20 spatial zones` across a tight latitude/longitude bounding box representing the metropolitan core:
  * Latitude: `[12.9658, 12.9658]` centroid core.
  * Spatial representation: Formulated using $K$-Means clustering ($K=20$) based on coordinates.

### B. Synthetic Dataset Characteristics
* **Total Incident Records**: `611,879 complaints` (saved to `data/synthetic_complaints.csv`)
* **Expansion Factor**: **`38.07x`** volumetric expansion.
* **Temporal Coverage**: Extended to `2019-01-01` to `2026-12-31` (8 full years), providing sequence-heavy GNN continuous windows ($T=2,922$ time steps).

---

## 2. SpatioTemporalSyntheticGenerator Workflow

The generator executes in two distinct sequential phases: a prior learning phase (`fit`) and a probabilistic data synthesis phase (`generate`).

```text
Prior Learning (fit)
  ├── 1. Spatial Hotspots (K-Means centroids & Lat/Lon variances per zone)
  ├── 2. Temporal PMFs (Diurnal hours, weekly days, seasonal months)
  ├── 3. Category & Status Priors (Conditional distribution P(Cat|Zone), P(Open|Cat))
  └── 4. Environmental Slopes (Covariance slopes for Temperature, Humidity, Rain, Festivals)
        │
        ▼
Probabilistic Generation (generate)
  ├── 1. Timeline & Weather Simulation (Tiled calendar averages + daily Gaussian noise)
  ├── 2. Poisson Rate Lambda Formulation (Window arrival rate per zone)
  ├── 3. Dynamic Backlog Recurrence (backlog unresolved queue boosts subsequent rates)
  ├── 4. Adjacency Graph Spillover convolving (15% rate diffusion over row-normalized KNN)
  ├── 5. Poisson Incident Sampling (Daily zone-window counts)
  └── 6. Micro-level Incident Synthesizer (Gaussian coord mapping, P(Cat|Zone) sampling, P(Open|Cat) status mapping)
```

### Micro-Level Attribute Synthesizer Details:
1. **Spatial Sampling**: Synthesizes exact coordinates using Gaussian sampling centered on the zone centroid:
   $$\text{Lat} \sim \mathcal{N}(\mu_{\text{lat}, z}, \sigma_{\text{lat}, z}), \quad \text{Lon} \sim \mathcal{N}(\mu_{\text{lon}, z}, \sigma_{\text{lon}, z})$$
   Variances are scaled by **$1.15$** under heavy rainfall simulations to model dispersed geographic report behavior.
2. **Temporal Sampling**: Incident times are distributed uniformly within each 6-hour window by adding random offsets:
   $$\text{time} = \text{Window\_Start} + \delta, \quad \delta \sim \mathcal{U}(0, 360 \text{ minutes})$$
3. **Category Generation**: Samples a category ID for each incident using the zone-specific conditional prior $P(\text{Category} \mid \text{Zone}_z)$.
4. **Status Generation**: Maps incident resolution state ("Open" backlog or terminal "Resolved") using the category-specific conditional probability $P(\text{Open} \mid \text{Category}_c)$.
5. **Recurrence Generation**: Simulates duplicate reporting behavior. Unresolved backlogs dynamically scale up Poisson rates in the subsequent interval:
   $$\lambda_{t, z} = \lambda_{t, z} \times \left(1.0 + \min(0.3, \text{Queue}_z \cdot 0.05)\right)$$
6. **Backlog Generation**: Queue tracking accumulates unresolved incidents from the previous window and decays them by a factor of $0.7$ at each step to represent typical municipal resolution latency:
   $$\text{Queue}_z(t) = \lfloor 0.7 \cdot \text{Queue}_z(t-1) + \text{NewUnresolved}_z(t) \rfloor$$

---

## 3. Weather & Festival Conditioning Mechanisms

The generator scales the baseline window arrival rate using empirical multipliers derived from environmental covariances.

### A. Weather Conditioning Logic
The weather multiplier ($M_{\text{weather}}$) combines step-wise discrete rainfall levels and continuous covariance slopes for temperature and relative humidity:
$$M_{\text{weather}}(t) = M_{\text{rain}}(t) \times \left(1.0 + \beta_{\text{temp}} \cdot \frac{T_t - \bar{T}}{100.0}\right) \times \left(1.0 + \beta_{\text{hum}} \cdot \frac{H_t - \bar{H}}{100.0}\right)$$

Where:
* **Rainfall Multiplier ($M_{\text{rain}}$)**: 
  * $1.0$ if $\text{Rainfall} = 0$ (No Rain)
  * $1.10$ (Empirical fallback: $M_{\text{light\_rain}}$) if $0 < \text{Rainfall} < 5.0$ mm (Light Rain)
  * $1.40$ (Empirical fallback: $M_{\text{heavy\_rain}}$) if $\text{Rainfall} \geq 5.0$ mm (Heavy Rain)
* **Temperature Slope ($\beta_{\text{temp}}$)**: $\frac{\text{Cov}(\text{Count}, T)}{\text{Var}(T)}$ (Historical daily covariance coefficient).
* **Humidity Slope ($\beta_{\text{hum}}$)**: $\frac{\text{Cov}(\text{Count}, H)}{\text{Var}(H)}$ (Historical daily covariance coefficient).

### B. Festival Conditioning Logic
The festival multiplier ($M_{\text{festival}}$) amplifies base incident generation rates during major cultural holidays and their eves:
* **Festival Day**: $1.30$ fallback surge multiplier ($M_{\text{festival}}$) applied to dates flagged on the festival calendar.
* **Festival Eve**: $1.15$ surge multiplier ($M_{\text{festival\_eve}}$) applied to the 24-hour window preceding a major holiday.

### C. Joint Environmental Backlog Amplification
Under extreme weather conditions (Heavy Rainfall $\geq 5.0$ mm), municipal resolution backlogs are amplified by scaling up the unresolved probability $P(\text{Open} \mid \text{Category})$ by a factor of **$1.30$** (capped at a maximum probability of $0.90$) to model systemic delays and slower physical dispatch times during storms.

---

## 4. Adjacency-Aware Graph Spillover Propagation

Spatial diffusion is convolved over the neighborhood structure to represent dynamic incident spillovers across geographical boundaries (e.g., traffic congestions or drainage overflows affecting adjacent zones).

### A. Graph Construction & Neighbor Selection
* **Centroid-Based Node Distance**: Calculates Euclidean distances between K-Means zone centroids:
  $$d(z, j) = \sqrt{(\text{Lat}_z - \text{Lat}_j)^2 + (\text{Lon}_z - \text{Lon}_j)^2}$$
* **Graph Connectivity**: A geographic $k$-nearest neighbor ($k$-NN) graph is constructed with $k=3$ spatial neighbors. Edge weights are formulated using an inverse distance decay with epsilon padding:
  $$W_{z, j} = \frac{1}{d(z, j) + \epsilon}, \quad \text{where } \epsilon = 10^{-6}$$
* **Row-Normalization**: To preserve overall city-wide incident rate scales, the adjacency matrix is row-normalized:
  $$A_{\text{norm}}[z, j] = \frac{W_{z, j}}{\sum_{l} W_{z, l}}$$

### B. Spillover Coefficient & Rate Propagation Formula
At each 6-hour window, a portion ($\eta = 0.15$) of the calculated raw Poisson rates is convolved and diffused across the geographical boundaries of contiguous graph neighbors. 

The spatial rate propagation formula is:
$$\Lambda = (1.0 - \eta) \Lambda_{\text{raw}} + \eta \left(A_{\text{norm}} \Lambda_{\text{raw}}\right)$$

Where:
* $\Lambda_{\text{raw}} \in \mathbb{R}^K$: The vector of raw Poisson arrival rates computed for each of the $K$ zones using temporal, weather, and festival priors.
* $\Lambda \in \mathbb{R}^K$: The final diffused Poisson rates used to sample daily zone-window counts.
* $\eta = 0.15$: The spatial spillover convolving coefficient, indicating that $15\%$ of the incident rate is driven by geographic neighborhood pressure.
