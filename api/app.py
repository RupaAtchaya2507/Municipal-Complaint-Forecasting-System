"""
Municipal Incident Prediction API
===================================
FastAPI backend exposing the spatiotemporal GNN+LSTM pipeline.

Endpoints:
  GET  /health          — system health check
  GET  /zones           — zone graph info (centroids, neighbours)
  POST /predict         — complaint surge forecast per zone
  GET  /risk            — current MSI risk scores per zone
  GET  /msi             — full Municipal Stress Index breakdown
  GET  /hotspots        — DBSCAN detected hotspot regions

Run with:
  uvicorn api.app:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import (
    ComplaintIn, ComplaintIngestResponse,
    HealthResponse,
    HotspotResponse, HotspotSummary,
    MSIResponse, MSIZone,
    MetricsResponse,
    PredictRequest, PredictResponse, ZoneForecast,
    RiskResponse, ZoneRisk,
    ZoneInfo, ZonesResponse,
)
from api.state import app_state
from api.scheduler import DataScheduler
from src.clustering import assign_zone
from src.dataset import LAST_MSI_COMPONENTS

logger = logging.getLogger(__name__)

scheduler = DataScheduler(app_state, interval_hours=24)


# ──────────────────────────────────────────────
# Lifespan: load pipeline once on startup
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading ML pipeline...")
    app_state.load()
    scheduler.start()          # start 24h refresh cycle
    logger.info("Pipeline loaded. Scheduler started. API ready.")
    yield
    scheduler.stop()
    logger.info("Shutting down.")


# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────

app = FastAPI(
    title="Spatiotemporal Incident Prediction API",
    description="City Risk Forecasting Engine for Municipal Infrastructure Stress",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_ready():
    if not app_state.ready:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet. Try again shortly.")


# ──────────────────────────────────────────────
# POST /complaints
# ──────────────────────────────────────────────

@app.post("/complaints", response_model=ComplaintIngestResponse, tags=["Data Ingestion"])
def ingest_complaint(complaint: ComplaintIn):
    """
    Accept a real-time complaint and append it to the in-memory dataset.

    In production this endpoint is called by:
      - BBMP Sahaaya portal on each new citizen complaint
      - A mobile app built on top of this API
      - The simulate_realtime.py replay script (for demo)

    After ingestion the pipeline refresh is triggered automatically
    so risk scores and predictions reflect the new complaint.
    """
    _require_ready()

    import pandas as pd

    new_row = pd.DataFrame([{
        "created_at":              pd.to_datetime(complaint.created_at, format="mixed", dayfirst=False),
        "latitude":                complaint.latitude,
        "longitude":               complaint.longitude,
        "category_id":             complaint.category_id,
        "category_title":          complaint.category_title,
        "sub_category_title":      complaint.sub_category_title,
        "complaint_status_title":  complaint.complaint_status_title,
        "ward_id":                 complaint.ward_id,
        "ward_title":              complaint.ward_title,
        "civic_agency_title":      complaint.civic_agency_title,
        "comment_count":           complaint.comment_count,
        "description":             complaint.description,
    }])
    new_row["date"] = new_row["created_at"].dt.date

    # Assign to nearest zone using existing centroids
    zone_id = int(assign_zone(
        new_row[["latitude", "longitude"]].values,
        app_state.centroids
    )[0])
    new_row["zone_id"] = zone_id

    # Append to in-memory complaints DataFrame
    app_state.df_complaints = pd.concat(
        [app_state.df_complaints, new_row], ignore_index=True
    )

    total = len(app_state.df_complaints)
    logger.info(f"New complaint ingested → zone {zone_id} | total={total}")

    # Trigger background refresh so predictions update
    scheduler.trigger_now()

    return ComplaintIngestResponse(
        status="accepted",
        zone_id=zone_id,
        total_complaints=total,
        message=f"Complaint assigned to zone {zone_id}. Pipeline refresh triggered.",
    )


@app.post("/refresh", tags=["System"])
def trigger_refresh():
    """
    Manually trigger a real-time data refresh.
    Fetches latest weather, re-runs feature engineering, updates predictions.
    """
    _require_ready()
    scheduler.trigger_now()
    return {"status": "refresh triggered", "message": "Running in background. Check /risk in ~30 seconds."}


# ──────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Check whether the API and ML pipeline are ready."""
    return HealthResponse(
        status="ok" if app_state.ready else "loading",
        model_loaded=app_state.model is not None,
        num_zones=app_state.num_zones,
        num_features=app_state.num_features,
        pipeline_ready=app_state.ready,
    )


# ──────────────────────────────────────────────
# GET /zones
# ──────────────────────────────────────────────

@app.get("/zones", response_model=ZonesResponse, tags=["Spatial"])
def get_zones():
    """
    Return zone graph info — centroid coordinates and neighbour count per zone.
    Used by the frontend to render the GIS map.
    """
    _require_ready()

    adj = app_state.adj_matrix
    centroids = app_state.centroids
    zones = []

    for z in range(app_state.num_zones):
        num_neighbours = int((adj[z] > 0).sum())
        zones.append(ZoneInfo(
            zone_id=z,
            centroid_lat=round(float(centroids[z, 0]), 6),
            centroid_lon=round(float(centroids[z, 1]), 6),
            num_neighbors=num_neighbours,
        ))

    return ZonesResponse(num_zones=app_state.num_zones, zones=zones)


# ──────────────────────────────────────────────
# POST /predict
# ──────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse, tags=["Forecasting"])
def predict(body: PredictRequest):
    """
    Forecast complaint surge and MSI for each zone.

    Answers:
      - Which ward is likely to have high complaint volume?
      - When will the spike occur? (horizon)
      - Why is the spike expected?
      - What action should the municipality take?
    """
    _require_ready()

    # Re-run inference with requested horizon
    app_state._run_inference(horizon=body.horizon)

    predictions   = app_state.last_predictions        # [N]
    risk_results  = app_state.last_risk_results        # list[dict]
    cnt_preds     = app_state._last_cnt                # [N] or None
    unres_preds   = app_state._last_unres              # [N] or None

    # Map zone_id → dominant complaint category from recent data
    dominant_cats = _dominant_categories_per_zone(app_state.df_complaints,
                                                   app_state.num_zones)

    forecasts = []
    for z in range(app_state.num_zones):
        if body.zone_ids and z not in body.zone_ids:
            continue

        risk = risk_results[z]
        pred_msi = float(predictions[z])
        risk_level = risk["risk_level"]
        primary_driver = risk["explanation"]["primary_driver"]
        explanation_text = risk["explanation"]["explanation"]

        # Action recommendation based on risk level and primary driver
        action = _recommend_action(risk_level, primary_driver)
        why = f"{explanation_text}. {action}"

        forecasts.append(ZoneForecast(
            zone_id=z,
            predicted_msi=round(pred_msi, 4),
            predicted_complaint_count=round(float(cnt_preds[z]), 2) if cnt_preds is not None else None,
            predicted_unresolved_ratio=round(float(unres_preds[z]), 4) if unres_preds is not None else None,
            risk_level=risk_level,
            dominant_complaint_type=dominant_cats.get(z),
            why=why,
        ))

    return PredictResponse(
        horizon=body.horizon,
        num_zones=app_state.num_zones,
        forecasts=forecasts,
    )


# ──────────────────────────────────────────────
# GET /risk
# ──────────────────────────────────────────────

@app.get("/risk", response_model=RiskResponse, tags=["Risk"])
def get_risk(
    risk_level: Optional[str] = Query(default=None, description="Filter by risk level: Low | Medium | High"),
    zone_id: Optional[int] = Query(default=None, description="Filter by specific zone ID"),
):
    """
    Return current MSI risk scores for all zones (Module 6 — Green/Yellow/Red).

    Supports filtering by risk level or zone ID.
    """
    _require_ready()

    results = app_state.last_risk_results
    zones = []

    for r in results:
        if zone_id is not None and r["zone_id"] != zone_id:
            continue
        if risk_level and r["risk_level"].lower() != risk_level.lower():
            continue

        zones.append(ZoneRisk(
            zone_id=r["zone_id"],
            risk_score=r["risk_score"],
            risk_level=r["risk_level"],
            msi_components=r.get("msi_components", {}),
            weights=r["weights"],
            explanation=r["explanation"],
        ))

    all_levels = [r["risk_level"] for r in results]
    return RiskResponse(
        num_zones=app_state.num_zones,
        high_risk_count=all_levels.count("High"),
        medium_risk_count=all_levels.count("Medium"),
        low_risk_count=all_levels.count("Low"),
        zones=zones,
    )


# ──────────────────────────────────────────────
# GET /msi
# ──────────────────────────────────────────────

@app.get("/msi", response_model=MSIResponse, tags=["MSI"])
def get_msi():
    """
    Return the full Municipal Stress Index breakdown per zone (Module 5).

    Includes all 6 components:
      C — Complaint Deviation, U — Unresolved Ratio, G — Growth Rate,
      N — Neighbor Pressure, W — Weather Anomaly, V — Road Vulnerability
    """
    _require_ready()

    components = LAST_MSI_COMPONENTS
    y_msi = app_state.y_msi           # [samples, N]
    predictions = app_state.last_predictions   # [N]

    if y_msi is None or len(y_msi) == 0:
        raise HTTPException(status_code=500, detail="MSI targets not computed yet.")

    p50 = float(np.percentile(y_msi, 50))
    p80 = float(np.percentile(y_msi, 80))

    zones = []
    for z in range(app_state.num_zones):
        avg_msi = float(y_msi[:, z].mean())
        max_msi = float(y_msi[:, z].max())
        pred_msi = float(predictions[z])

        risk_class = "HIGH" if avg_msi >= p80 else ("MEDIUM" if avg_msi >= p50 else "LOW")

        # Extract latest step component values
        def _last(key):
            arr = components.get(key)
            if arr is None:
                return 0.0
            return float(arr[-1, z]) if arr.ndim == 2 else 0.0

        zones.append(MSIZone(
            zone_id=z,
            avg_msi=round(avg_msi, 4),
            max_msi=round(max_msi, 4),
            predicted_msi=round(pred_msi, 4),
            risk_class=risk_class,
            complaint_deviation=round(_last("C_norm"), 4),
            unresolved_ratio=round(_last("U_norm"), 4),
            growth_rate=round(_last("G_norm"), 4),
            neighbor_pressure=round(_last("N_norm"), 4),
            weather_anomaly=round(_last("W_norm"), 4),
            road_vulnerability=round(_last("V_norm"), 4),
        ))

    return MSIResponse(
        num_zones=app_state.num_zones,
        p50_msi=round(p50, 4),
        p80_msi=round(p80, 4),
        zones=zones,
    )


# ──────────────────────────────────────────────
# GET /hotspots
# ──────────────────────────────────────────────

@app.get("/hotspots", response_model=HotspotResponse, tags=["Hotspots"])
def get_hotspots(
    risk_level: Optional[str] = Query(default=None, description="Filter: Low | Medium | High"),
    refresh: bool = Query(default=False, description="Re-run DBSCAN before returning"),
):
    """
    Return DBSCAN-detected complaint hotspot regions (Module 4).

    Each hotspot includes centroid, dominant complaint type, risk level,
    density score, and wards covered.
    """
    _require_ready()

    if refresh:
        app_state._run_hotspots()

    result = app_state.last_hotspot_result
    if not result:
        raise HTTPException(status_code=500, detail="Hotspot detection has not run yet.")

    all_hotspots = result.get("all_hotspots", [])
    summary = result.get("summary", {})

    hotspots_out = []
    for h in all_hotspots:
        if risk_level and h.get("risk_level", "").lower() != risk_level.lower():
            continue
        hotspots_out.append(HotspotSummary(
            hotspot_id=h["hotspot_id"],
            centroid_lat=h["centroid_lat"],
            centroid_lon=h["centroid_lon"],
            complaint_count=h["complaint_count"],
            dominant_category=h["dominant_category"],
            unresolved_ratio=h["unresolved_ratio"],
            risk_level=h.get("risk_level", "Unknown"),
            risk_score=h.get("risk_score", 0.0),
            density_score=h["density_score"],
            wards_covered=[int(w) for w in h.get("wards_covered", []) if str(w).isdigit()],
        ))

    return HotspotResponse(
        total_hotspots=summary.get("total_hotspots", len(hotspots_out)),
        high_risk=summary.get("high_risk", 0),
        medium_risk=summary.get("medium_risk", 0),
        low_risk=summary.get("low_risk", 0),
        emerging_count=summary.get("emerging_count", 0),
        hotspots=hotspots_out,
    )


# ──────────────────────────────────────────────
# GET /metrics
# ──────────────────────────────────────────────

@app.get("/metrics", response_model=MetricsResponse, tags=["Evaluation"])
def get_metrics():
    """
    Return model evaluation metrics on the test set (Module 8).

    Includes MAE, RMSE, R², F1 Score (Low/Medium/High/Macro),
    and Lead Time Accuracy.
    """
    _require_ready()

    from src.utils import compute_risk_classification_metrics, compute_lead_time_accuracy
    import config as cfg

    y_msi = app_state.y_msi           # [samples, N]
    predictions = app_state.last_predictions   # [N] latest predictions

    if y_msi is None or len(y_msi) < 2:
        raise HTTPException(status_code=500, detail="MSI targets not available.")

    n = len(y_msi)
    val_end = int(n * (cfg.TRAIN_RATIO + cfg.VAL_RATIO))

    # Test split: absolute MSI
    y_test_abs = y_msi[val_end:]          # [T_test, N]
    if len(y_test_abs) < 2:
        raise HTTPException(status_code=500, detail="Not enough test samples.")

    # Use predictions repeated across test steps as a proxy
    # (actual rolling predictions would require re-running inference for each step)
    T_test = len(y_test_abs)
    preds_tiled = np.tile(predictions, (T_test, 1))   # [T_test, N]

    # Regression metrics
    reg = compute_metrics(
        y_test_abs.flatten(), preds_tiled.flatten(), regression=True
    )

    # F1 risk classification metrics
    f1 = compute_risk_classification_metrics(
        y_test_abs.flatten(), preds_tiled.flatten(),
        thresholds=cfg.RISK_THRESHOLDS,
    )

    # Lead Time Accuracy
    timestamps = np.arange(T_test)
    lead = compute_lead_time_accuracy(
        y_test_abs, preds_tiled, timestamps,
        high_thresh=cfg.RISK_THRESHOLDS[1],
        tolerance_steps=1,
    )

    return MetricsResponse(
        mae=round(reg["mae"], 6),
        rmse=round(reg["rmse"], 6),
        r2=round(reg["r2"], 6),
        f1_macro=round(f1["f1_macro"], 4),
        f1_low=round(f1["f1_low"], 4),
        f1_medium=round(f1["f1_medium"], 4),
        f1_high=round(f1["f1_high"], 4),
        lead_time_accuracy=round(lead["lead_time_accuracy"], 4),
        total_spike_zones=lead["total_spike_zones"],
        mean_lead_error_steps=round(lead["mean_lead_error_steps"], 2)
        if not np.isnan(lead["mean_lead_error_steps"]) else -1.0,
    )


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────

def _dominant_categories_per_zone(df, num_zones: int) -> dict:
    """Return {zone_id: dominant_complaint_category} from complaint DataFrame."""
    if df is None or "zone_id" not in df.columns:
        return {}
    cat_col = "complaint_category" if "complaint_category" in df.columns else "category_title"
    if cat_col not in df.columns:
        return {}
    result = {}
    for z in range(num_zones):
        zone_df = df[df["zone_id"] == z]
        if len(zone_df) == 0:
            result[z] = None
            continue
        result[z] = zone_df[cat_col].value_counts().index[0]
    return result


def _recommend_action(risk_level: str, primary_driver: str) -> str:
    """Generate a municipality action recommendation based on risk level and driver."""
    actions = {
        ("High", "prediction"):         "Deploy emergency response team immediately.",
        ("High", "unresolved"):         "Escalate unresolved complaints. Assign additional staff.",
        ("High", "density"):            "Surge in complaints detected. Increase patrol frequency.",
        ("High", "weather_anomaly"):    "Heavy rain alert. Pre-position drainage crews.",
        ("High", "road_vulnerability"): "Critical road infrastructure risk. Schedule urgent inspection.",
        ("Medium", "prediction"):       "Monitor closely. Pre-position maintenance crew.",
        ("Medium", "unresolved"):       "Clear complaint backlog. Prioritise older open tickets.",
        ("Medium", "density"):          "Moderate surge. Schedule preventive maintenance.",
        ("Medium", "weather_anomaly"):  "Rain forecast. Check drain clearance status.",
        ("Medium", "road_vulnerability"): "Road condition review recommended within 48 hours.",
        ("Low", "prediction"):          "No immediate action required. Continue routine monitoring.",
        ("Low", "unresolved"):          "Routine complaint resolution. No escalation needed.",
        ("Low", "density"):             "Normal complaint volume. Standard operations.",
        ("Low", "weather_anomaly"):     "Mild weather impact. Standard monitoring.",
        ("Low", "road_vulnerability"):  "Road infrastructure stable. Routine inspection cycle.",
    }
    return actions.get((risk_level, primary_driver),
                       "Monitor situation and follow standard municipal protocols.")
