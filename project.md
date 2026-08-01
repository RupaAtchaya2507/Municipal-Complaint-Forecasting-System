# Spatiotemporal Incident Prediction and Dynamic Risk Assessment System

---

## 1. SYSTEM OBJECTIVE

Build a spatiotemporal prediction system that:
- Processes complaint data across time and location
- Learns spatial dependencies using Graph Neural Networks (GNN)
- Learns temporal patterns using LSTM
- Computes a dynamic risk score using:
  - unresolved complaints
  - recent complaint density
  - model predictions
- Updates periodically (every 6 hours)

---

## 2. DATA INGESTION

### 2.1 Input Sources

#### Complaint Dataset (CSV)
Fields:
- created_at (timestamp)
- latitude, longitude
- category_id
- complaint_status_title

#### Weather Data
- temperature
- rainfall
- humidity

#### Festival Dataset (CSV)
- date
- festival_flag (0/1)
- festival_name (optional)

---

### 2.2 Tasks
- Load complaint dataset into DataFrame
- Convert created_at → datetime
- Merge weather data by timestamp/date
- Merge festival data by date

---

### 2.3 Output
Unified dataset:
[timestamp, lat, lon, category, status, weather, festival_flag]

---

## 3. DATA PREPROCESSING

### 3.1 Cleaning
- Remove rows with missing latitude/longitude
- Fill NULL categorical fields with "Unknown"

---

### 3.2 Time Features
- hour_of_day
- day_of_week
- is_weekend
- month
- is_festival_eve (1 if day before festival)

---

### 3.3 Status Encoding
- Open → 1
- Resolved → 0

---

### 3.4 Output
Clean dataset with structured features

---

## 4. SPATIAL MODELING (GRAPH)

### 4.1 Zone Creation
- Apply KMeans clustering on (latitude, longitude)
- Number of clusters: 10–15 (tunable via elbow method if needed)

---

### 4.2 Zone Assignment
- Assign each complaint → zone_id

---

### 4.3 Graph Construction
- Nodes = zones
- Edges:
  - k-nearest neighbors (k = 3)

### Edge Weights
- weight = 1 / (distance + ε)

---

### 4.4 Graph Representation
- Adjacency matrix A (NxN)
- Fixed across time

---

### 4.5 Output
- zone_id per complaint
- adjacency matrix
- zone centroids

---

## 5. TEMPORAL AGGREGATION

### 5.1 Time Window
- 6-hour fixed intervals

---

### 5.2 Aggregation per (zone, window)
- complaint_count
- unresolved_count
- resolved_count

---

### 5.3 Time Indexing
- Sort chronologically
- Create ordered time steps

---

### 5.4 Output
[time_steps × num_zones × counts]

---

## 6. FEATURE ENGINEERING

### 6.1 Core Features
- complaint_count
- unresolved_count
- resolved_count

---

### 6.2 Derived Features

#### Unresolved Ratio
U_raw = unresolved_count / (unresolved_count + resolved_count + 1)

#### Density
D_raw = complaint_count / (max_complaint_count_per_window + 1)

---

### 6.3 Temporal Features
- hour_of_day
- day_of_week
- is_weekend
- month
- is_festival_eve

---

### 6.4 External Features
- temperature
- rainfall
- humidity
- festival_flag

---

### 6.5 Category Encoding
- Use embedding layer for category_id

---

### 6.6 Normalization
Apply MinMax scaling:
- U_raw → U
- D_raw → D
- weather features

---

### 6.7 Output
Feature tensor:
[T × N × F]

---

## 7. SEQUENCE DATASET

### 7.1 Sequence Length
- Use {3, 5, 7} (select via validation)

---

### 7.2 Input-Target

Input:
[t-3 ... t-1]

Target:
t

---

### 7.3 Constraints
- Maintain chronological order
- No future data leakage

---

### 7.4 Output
X: [samples × seq_len × N × F]  
y: [samples × N]

---

## 8. MODEL ARCHITECTURE

### 8.1 Input
[T, N, F]

---

### 8.2 Category Embedding
- Embed category_id into dense vector

---

### 8.3 GNN (Spatial)
- 2-layer GCN
- hidden_dim = 32
- residual connection

---

### 8.4 Temporal (LSTM)
- 1–2 layers
- hidden_dim = 64

---

### 8.5 Regularization
- Dropout = 0.3–0.4
- Batch Normalization before FC

---

### 8.6 Output Layer
- Fully connected
- Sigmoid activation
- Output = P (probability per zone)

---

## 9. TRAINING STRATEGY

### 9.1 Data Split (CRITICAL)
- Train: first 70%
- Validation: next 15%
- Test: last 15%

(no random shuffling)

---

### 9.2 Loss Function
- Focal Loss (handles class imbalance)

---

### 9.3 Optimizer
- Adam
- weight_decay = 1e-4

---

### 9.4 Learning Rate
- Use ReduceLROnPlateau scheduler

---

### 9.5 Early Stopping
- Based on validation F1-score

---

### 9.6 Metrics
- Precision
- Recall
- F1-score
- AUC-ROC

---

### 9.7 Class Imbalance Handling
- Apply class weights OR oversampling

---

## 10. DYNAMIC RISK ENGINE (FIXED)

### 10.1 Inputs
- P = prediction
- U = normalized unresolved ratio
- D = normalized density

---

### 10.2 Softmax Weighting

Compute:
s_u = U  
s_d = D  
s_p = P  

[w_u, w_d, w_p] = softmax([s_u, s_d, s_p])

---

### 10.3 Risk Score

Risk_raw =
    w_u * U
  + w_d * D
  + w_p * P

---

### 10.4 Temporal Smoothing (EMA)

Risk_t =
    α * Risk_raw
  + (1 - α) * Risk_{t-1}

Where:
- α = 0.3

---

### 10.5 Risk Levels
- Low / Medium / High based on thresholds (configurable)

---

### 10.6 Output
- risk_score
- risk_level
- component contributions

---

## 11. UPDATE & FEEDBACK

### 11.1 Every 6 Hours
- Ingest new complaints
- Update:
  - counts
  - density
  - unresolved ratio
- Recompute risk

---

### 11.2 Every 24 Hours
- Retrain model

---

### 11.3 Feedback Loop
- Update unresolved_count when resolved
- Improve future training labels

---

### 11.4 Data Retention
- Rolling window (60 days)
OR
- weighted historical data

---

## 12. IMPLEMENTATION CONSTRAINTS

- Fixed graph across time
- No feature leakage
- Normalize all features
- Maintain strict temporal order
- Keep model small to avoid overfitting

---

## 13. BUILD ORDER

1. Clustering (zones)
2. Graph creation
3. Temporal aggregation
4. Feature engineering
5. Sequence dataset
6. Train baseline
7. Add GNN
8. Add risk engine

---