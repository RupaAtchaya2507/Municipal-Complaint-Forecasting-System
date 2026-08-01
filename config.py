"""
Central configuration for the Spatiotemporal Incident Prediction System.
All hyperparameters, file paths, and feature lists are defined here.
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

COMPLAINTS_CSV = os.path.join(DATA_DIR, "synthetic_complaints.csv")  # full 2019-2026
WEATHER_CSV = os.path.join(DATA_DIR, "weather.csv")
FESTIVALS_CSV = os.path.join(DATA_DIR, "festivals.csv")

# ──────────────────────────────────────────────
# Weather API Integration
# ──────────────────────────────────────────────
USE_WEATHER_API = True
WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_API_LAT = 12.9658
WEATHER_API_LON = 77.6144


# ──────────────────────────────────────────────
# Data Ingestion
# ──────────────────────────────────────────────
CSV_ENCODING = "latin-1"        # complaints.csv uses Windows-1252/latin-1
TIMESTAMP_COL = "created_at"
LAT_COL = "latitude"
LON_COL = "longitude"
CATEGORY_COL = "category_id"
STATUS_COL = "complaint_status_title"

# Status encoding:
#   Open (unresolved)   → 1
#   On-the-Job          → 1  (still in progress)
#   Re-opened           → 1  (unresolved again)
#   Resolved            → 0
#   Closed              → 0
#   Rejected            → 0  (treated as terminal/closed)
STATUS_OPEN = ["open", "on-the-job", "re-opened"]
STATUS_RESOLVED = ["resolved", "closed", "rejected"]

# ──────────────────────────────────────────────
# Category
# ──────────────────────────────────────────────
NUM_CATEGORIES = 709            # max category_id = 708, so 709 slots (0-indexed)
CATEGORY_UNKNOWN_ID = 0         # fill NaN category_id with 0

# ──────────────────────────────────────────────
# Spatial Clustering
# ──────────────────────────────────────────────
NUM_CLUSTERS = 12               # default; tunable via elbow method
CLUSTER_K_RANGE = (20, 20)      # range to search for optimal k
KNN_NEIGHBORS = 4               # increased from 3 — richer spatial graph
EDGE_EPSILON = 1e-6             # epsilon for edge weight: 1 / (dist + ε)

# ──────────────────────────────────────────────
# Temporal Aggregation
# ──────────────────────────────────────────────
TIME_WINDOW_HOURS = 24          # aggregation window size

# ──────────────────────────────────────────────
# Sequence Dataset
# ──────────────────────────────────────────────
SEQ_LENGTHS = [7, 10, 14]       # candidate sequence lengths
DEFAULT_SEQ_LEN = 14            # 14 days captures bi-weekly patterns
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ──────────────────────────────────────────────
# Model Architecture
# ──────────────────────────────────────────────
GCN_HIDDEN_DIM = 96             # balanced between speed and capacity
GCN_NUM_LAYERS = 2
LSTM_HIDDEN_DIM = 192           # balanced between speed and capacity
LSTM_NUM_LAYERS = 2
DROPOUT_RATE = 0.2              # reduced from 0.3 — less regularization for larger model
CATEGORY_EMBED_DIM = 8          # embedding size for category_id

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
LEARNING_RATE = 5e-4            # reduced from 1e-3 — slower learning for larger model
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64                 # reduced from 128 — better gradient estimates
LR_SCHEDULER_PATIENCE = 8      # increased from 5
LR_SCHEDULER_FACTOR = 0.5      # LR reduction factor

# Focal Loss
FOCAL_LOSS_GAMMA = 2.0
FOCAL_LOSS_ALPHA = 0.25         # focus on hard examples, reduce confidence over smooth

# Label Thresholding
USE_PERCENTILE_LABEL = True     # Use top percentile for high risk
LABEL_PERCENTILE = 60           # Define top 40% of non-zero as high risk

# ──────────────────────────────────────────────
# Dynamic Risk Engine
# ──────────────────────────────────────────────
EMA_ALPHA = 0.3                 # smoothing factor
RISK_THRESHOLDS = (0.3, 0.7)   # (low/medium boundary, medium/high boundary)

# ──────────────────────────────────────────────
# Update & Feedback
# ──────────────────────────────────────────────
RISK_UPDATE_INTERVAL_HOURS = 6
MODEL_RETRAIN_INTERVAL_HOURS = 24
DATA_RETENTION_DAYS = 60

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
RANDOM_SEED = 42

# ──────────────────────────────────────────────
# Training speed controls
# ──────────────────────────────────────────────
SKIP_TRAINING = True     # use best saved model
MAX_EPOCHS = 300
EARLY_STOP_PATIENCE = 30
USE_PIPELINE_CACHE = True

# ──────────────────────────────────────────────
# Production Pipeline Settings
# ──────────────────────────────────────────────
MODEL_TYPE = "multi_task"       # multi-task learns MSI + count + unresolved simultaneously
PREDICT_DELTA = False           # train on absolute MSI directly for honest R² and zone differentiation
LOSS_TYPE = "huber"             # huber loss is robust to outliers in absolute MSI
SCALING_METHOD = "robust"       # robust scaling handles skewed complaint distributions
USE_SIGMOID = True              # sigmoid bounds output to [0,1] matching MSI range
RISK_WEIGHTING_METHOD = "dynamic" # dynamic weights adapt to current zone conditions
USE_STATIC_FEATURES = True      # inject static baseline features as node attributes

# ──────────────────────────────────────────────
# Multi-Task Loss Weights (used only if MODEL_TYPE = "multi_task")
# ──────────────────────────────────────────────
MSI_LOSS_WEIGHT   = 0.6         # primary target gets highest weight
COUNT_LOSS_WEIGHT = 0.2         # auxiliary head
UNRES_LOSS_WEIGHT = 0.2         # auxiliary head

# ──────────────────────────────────────────────
# Risk Engine Thresholds
# Tuned to match absolute MSI percentile distribution
# 50th percentile ≈ 0.21, 80th percentile ≈ 0.51
# ──────────────────────────────────────────────
RISK_THRESHOLDS = (0.25, 0.55)  # (low/medium boundary, medium/high boundary)


