# Spatiotemporal Municipal Complaint Forecasting System

A GNN+LSTM deep learning system for predicting municipal infrastructure stress across city zones using spatiotemporal complaint data.

## Architecture

- **GNN** — Graph Convolutional Network for spatial zone relationships
- **LSTM** — Long Short-Term Memory for temporal complaint patterns
- **Multi-Task Head** — Jointly predicts MSI, complaint count, and unresolved ratio
- **Dynamic Risk Engine** — EMA-based risk scoring (Low / Medium / High)
- **FastAPI** — REST API for real-time inference and hotspot detection

## Project Structure

```
project/
├── api/            # FastAPI endpoints
├── src/            # Core ML pipeline modules
├── research/       # Experiment scripts
├── tests/          # Unit tests
├── data/           # Data directory (see Data Setup below)
├── models/         # Trained model weights (not tracked in git)
├── config.py       # Central configuration
├── main.py         # Full pipeline entry point
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Data Setup

The `data/` directory is not included in this repository due to PII and file size constraints.

You need to provide:
- `data/synthetic_complaints.csv` — complaint records with columns: `created_at`, `latitude`, `longitude`, `category_id`, `complaint_status_title`, etc.
- `data/festivals.csv` — (optional) local event calendar
- `data/zone_static_features.csv` — static zone features

To generate synthetic data:
```bash
python generate_synthetic_data.py
```

## Running the Pipeline

```bash
# Full training pipeline
python main.py

# API server
uvicorn api.app:app --reload --port 8000
```

## API Endpoints

| Method | Endpoint     | Description                        |
|--------|--------------|------------------------------------|
| GET    | /health      | System health check                |
| GET    | /zones       | Zone graph info                    |
| POST   | /predict     | Complaint surge forecast per zone  |
| GET    | /risk        | Current MSI risk scores            |
| GET    | /msi         | Full MSI breakdown                 |
| GET    | /hotspots    | DBSCAN hotspot regions             |
| GET    | /metrics     | Model evaluation metrics           |

## Configuration

All hyperparameters and paths are in `config.py`. Key settings:

```python
NUM_CLUSTERS = 20          # spatial zones
DEFAULT_SEQ_LEN = 14       # 14-day input window
MODEL_TYPE = "multi_task"  # multi-task GNN+LSTM
SKIP_TRAINING = True       # use saved model weights
```

## Running Tests

```bash
pytest tests/
```
