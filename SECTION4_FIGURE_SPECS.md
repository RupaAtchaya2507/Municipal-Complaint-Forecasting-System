# Section IV: Publication-Quality Figure Specifications

This document defines the structural specifications, hierarchical blocks, flow routing directions, and IEEE-format captions for the architectural diagrams in Section IV (Synthetic Urban Complaint Generation Framework). These specifications are designed for direct rendering in vector drawing suites (such as TikZ, Draw.io, or Visio) for IEEE journal publication.

---

## Figure 4: Synthetic Data Generation Pipeline

### 1. Title & Classification
* **Title**: Figure 4. Unified Spatiotemporal Prior Learning and Synthetic Complaint Generation Pipeline.
* **Type**: Block Architecture & Data Pipeline.
* **Canvas Dimensions**: 16:9 widescreen ratio, double-column width.

### 2. Hierarchical Blocks & Components
* **Block A: Prior Learning Engine (Stateful Fitter)** [Left Column, Dark Blue Header]
  * `Block A1`: Spatial Hotspot Extractor (K-Means Centroids, Lat/Lon variance boundaries)
  * `Block A2`: Temporal PMF Estimator (24h Diurnal, 7d Weekly, 12m Seasonal PMFs)
  * `Block A3`: Categorical Conditional Prior (Zone-specific $P(\text{Cat}\mid\text{Zone})$ matrix)
  * `Block A4`: Environmental Shock Coeffs (Weather covariance slopes, Festival holiday surge factor)
* **Block B: Timeline & Environmental Simulator** [Top-Middle, Gray Background]
  * `Block B1`: Expanded Date Timeline Series Generator (2019-01-01 to 2026-12-31)
  * `Block B2`: Daily Weather Calendar Synthesizer (Monthly-averages + Gaussian daily perturbations)
  * `Block B3`: Public Festival Calendar Mapper (Binary Holiday & eve mapping)
* **Block C: Probabilistic Incident Engine** [Bottom-Middle, Teal Header]
  * `Block C1`: Window Rate lambda calculator (Diurnal * Weekly * Seasonal * Weather * Festival rate)
  * `Block C2`: Active Backlog Recurrence Scaler (Duplicate reporting lambda boost based on unresolved queue)
  * `Block C3`: Adjacency-Aware Graph Spillover (15% rate convolution over row-normalized KNN)
  * `Block C4`: Poisson sampler (Window incident count generator: $\text{Count} \sim \text{Poisson}(\lambda)$)
* **Block D: Micro-incident Synthesis Layer** [Right Column, Dark Blue Header]
  * `Block D1`: Coordinates Generator (Zone Gaussian sampler + storm variance scaling + coordinate jittering)
  * `Block D2`: Micro-Timestamp Interpolator (Uniform time slice offset within 6-hour window)
  * `Block D3`: Category & Status Assignee (Samples $P(\text{Cat}\mid\text{Zone})$ and applies $P(\text{Open}\mid\text{Cat})$)
  * `Block D4`: High-Fidelity Text Metadata Mapper (Historical lookup map based on `(zone_id, category_id)`)

### 3. Pipeline Flow Routing (Arrows)
1. **Original Datasets** (`complaints.csv`, `weather.csv`, `festivals.csv`) $\longrightarrow$ *Ingest & Cluster* $\longrightarrow$ **Block A** (Prior Learning Engine).
2. **Block A** learned priors $\longrightarrow$ *Prior Parameters* $\longrightarrow$ **Block C** (Probabilistic Incident Engine).
3. **Block B** simulated dates/weather $\longrightarrow$ *Environmental Conditioners* $\longrightarrow$ **Block C1** (Rate Calculator).
4. **Block C** output counts $\longrightarrow$ *Incident Counts* $\longrightarrow$ **Block D** (Micro-incident Synthesis).
5. **Block D** outputs $\longrightarrow$ *Export* $\longrightarrow$ **Output Files** (`synthetic_complaints.csv` $\rightarrow$ `synthetic_aggregated.csv` $\rightarrow$ `synthetic_features.npy`).

### 4. Caption
> **Figure 4.** Flow schematic of the prior learning and probabilistic synthesis pipeline. The generator fits historical data on the left to learn multidimensional priors, overlays simulated daily climate and holiday events on the top-middle, and executes Poisson incident sampling and micro-level attribute mapping to synthesize GNN-consistent spatiotemporal sequence outputs.

---

## Figure 5: Complaint Expansion Flowchart

### 1. Title & Classification
* **Title**: Figure 5. Step-by-Step Incident Generation and Volumetric Expansion Flowchart.
* **Type**: Process Decision Flowchart.
* **Canvas Dimensions**: Vertical orientation, single-column width.

### 2. Flowchart Nodes & Decisional Hierarchy
1. **Start** [Oval, White Fill, Blue Border]
2. **Initialize Timeline Step** [Rectangle, Gray Fill]: Set date $d$, 6-hour interval $t$, and spatial zone $z=0$.
3. **Calculate Raw Rate** [Rectangle, Gray Fill]: Compute $\lambda_{\text{raw}} = \text{BaseRate} \times P(\text{zone}_z) \times \text{Seasonality}$.
4. **Apply Environmental Multipliers** [Rectangle, Gray Fill]: Multiply by Weather Surge $M_{\text{weather}}(t)$ and Festival Surge $M_{\text{fest}}(t)$.
5. **Backlog Recurrence Boost?** [Decision Diamond, Yellow Fill]: Is unresolved backlog queue $\text{Queue}_z > 0$?
   * **Yes**: Scale rate $\lambda_{t, z} = \lambda_{t, z} \times \left(1.0 + \min(0.3, \text{Queue}_z \cdot 0.05)\right)$.
   * **No**: Proceed.
6. **Apply Spatial Graph Spillover** [Rectangle, Gray Fill]: Diffuse rates across KNN boundary: $\Lambda = (1.0 - \eta)\Lambda + \eta (A_{\text{norm}}\Lambda)$.
7. **Poisson Sampling** [Rectangle, Gray Fill]: Sample total window-incident count: $N_{z, t} \sim \text{Poisson}(\lambda_{z, t})$.
8. **Loop Incidents ($i = 0$ to $N_{z, t}-1$)** [Decision Loop Diamond, Teal Fill]:
   * `8a. Coordinates`: Lat/Lon $\sim \mathcal{N}(\mu_z, \sigma_z)$. If heavy rain, standard deviation scaled by $1.15$ + small Gaussian coordinate jittering.
   * `8b. Timestamp`: Time = Window\_Start + $\delta$, where $\delta \sim \mathcal{U}(0, 360 \text{ min})$.
   * `8c. Category`: Sample Category ID from $P(\text{Cat}\mid\text{Zone}_z)$.
   * `8d. Status`: Map status to "Open" with probability $P(\text{Open}\mid\text{Cat})$ (scaled $1.30\times$ if heavy rain), else "Resolved".
   * `8e. Metadata`: Copy ward details, titles, agencies, and address text from historical lookup.
9. **Next Zone?** [Decision Diamond, Yellow Fill]: Is $z < K-1$?
   * **Yes**: Increment $z = z+1$, loop back to **Step 3**.
   * **No**: Proceed.
10. **Decay Backlog Queue** [Rectangle, Gray Fill]: Update queue for next step: $\text{Queue}_z = \lfloor 0.7 \cdot \text{Queue}_z + \text{NewUnresolved}_z \rfloor$.
11. **Next Timeline Step?** [Decision Diamond, Yellow Fill]: Is timeline step $t < T-1$?
    * **Yes**: Increment step $t = t+1$, loop back to **Step 2**.
    * **No**: Proceed.
12. **End** [Oval, White Fill, Blue Border]: Export raw synthetic complaint database.

### 3. Caption
> **Figure 5.** Operational process flowchart for the spatiotemporal complaint synthesis. The flowchart details the sequential rate calculations, active backlog recurrence boosts, spatial graph spillover convolutions, Poisson incident sampling, and micro-attribute synthesis loops executed at each daily interval.

---

## Figure 6: Spatial Spillover Mechanism

### 1. Title & Classification
* **Title**: Figure 6. Adjacency-Aware Spatial Spillover and Rate Propagation Mechanism.
* **Type**: Topological Graph Signal Diagram.
* **Canvas Dimensions**: Square, single-column width.

### 2. Block Diagram & Spatial Topography
* **Geographical Nodes** [Set of circular nodes representing spatial centroids]
  * Node $z$ (Target Zone): Centered in diagram, colored Deep Blue, labeled $\lambda_{\text{raw}, z}$.
  * Nodes $j_1, j_2, j_3$ (3-Nearest Neighbors): Colored Teal, labeled $\lambda_{\text{raw}, j_1}$, $\lambda_{\text{raw}, j_2}$, $\lambda_{\text{raw}, j_3}$.
  * Nodes $u_1, u_2$ (Unconnected Zones): Colored Muted Gray, outside neighbor boundary.
* **Topological Boundaries** [Graphical dashed boundaries]
  * Inner Circle: Zone centroid coordinates.
  * Dashed Red Boundary: Spatial Graph Neighbor Horizon ($k$-NN search radius, $k=3$).
* **Directed Rate Diffusion Arrows** [Weighted visual flow arrows]
  * Directed arrow from Node $z$ to Neighbors $j_1, j_2, j_3$ representing outward diffusion:
    * Weight $\text{Out} = \eta \cdot A_{\text{norm}}[z, j]$.
  * Directed arrow from Neighbors $j_1, j_2, j_3$ to Node $z$ representing neighborhood pressure spillover:
    * Weight $\text{In} = \eta \cdot A_{\text{norm}}[j, z]$.
  * Self-Loop arrow on Node $z$ representing local persistence retention:
    * Weight $\text{Retention} = 1.0 - \eta$.

### 3. Annotation Formulas
* Overlaid mathematical callout boxes on the canvas:
  1. *Centroid Distance Box*: $d(z, j) = \|x_z - x_j\|_2$
  2. *Edge Decay Weight Box*: $W_{z, j} = \frac{1}{d(z, j) + \epsilon}$
  3. *Row-Normalized Adjacency Box*: $A_{\text{norm}}[z, j] = \frac{W_{z, j}}{\sum_l W_{z, l}}$
  4. *Spillover Integration Box*: $\lambda_{\text{final}, z} = (1.0 - \eta)\lambda_{\text{raw}, z} + \eta \sum_{j \in \mathcal{N}(z)} A_{\text{norm}}[z, j] \lambda_{\text{raw}, j}$

### 4. Caption
> **Figure 6.** Schematic representation of the adjacency-aware spatial spillover mechanism. The target zone convolving rates are formulated over a 3-nearest neighbor graph constructed from zone centroids. A convolving coefficient $\eta = 0.15$ ensures that $15\%$ of the Poisson incident rates are diffused across spatial boundaries, smoothing spatiotemporal gradients and embedding geographical connectivity into the GNN sequence target.
