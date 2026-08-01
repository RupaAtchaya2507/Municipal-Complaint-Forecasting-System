"""
API Schemas
===========
Pydantic models for all request bodies and response payloads.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Shared
# ──────────────────────────────────────────────

class ZoneRisk(BaseModel):
    zone_id: int
    risk_score: float
    risk_level: str                     # "Low" | "Medium" | "High"
    msi_components: dict
    weights: dict
    explanation: dict


class ZoneForecast(BaseModel):
    zone_id: int
    predicted_msi: float
    predicted_complaint_count: Optional[float]
    predicted_unresolved_ratio: Optional[float]
    risk_level: str
    dominant_complaint_type: Optional[str] = None
    why: str                            # human-readable explanation


class HotspotSummary(BaseModel):
    hotspot_id: int
    centroid_lat: float
    centroid_lon: float
    complaint_count: int
    dominant_category: str
    unresolved_ratio: float
    risk_level: str
    risk_score: float
    density_score: float
    wards_covered: list[int]


# ──────────────────────────────────────────────
# /predict  — POST
# ──────────────────────────────────────────────

class PredictRequest(BaseModel):
    horizon: int = Field(default=1, ge=1, le=7, description="Steps ahead to forecast (1–7)")
    zone_ids: Optional[list[int]] = Field(default=None, description="Subset of zone IDs to return. None = all zones.")


class PredictResponse(BaseModel):
    horizon: int
    num_zones: int
    forecasts: list[ZoneForecast]


# ──────────────────────────────────────────────
# /risk  — GET
# ──────────────────────────────────────────────

class RiskResponse(BaseModel):
    num_zones: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    zones: list[ZoneRisk]


# ──────────────────────────────────────────────
# /hotspots  — GET
# ──────────────────────────────────────────────

class HotspotResponse(BaseModel):
    total_hotspots: int
    high_risk: int
    medium_risk: int
    low_risk: int
    emerging_count: int
    hotspots: list[HotspotSummary]


# ──────────────────────────────────────────────
# /msi  — GET
# ──────────────────────────────────────────────

class MSIZone(BaseModel):
    zone_id: int
    avg_msi: float
    max_msi: float
    predicted_msi: float
    risk_class: str
    complaint_deviation: float          # C component
    unresolved_ratio: float             # U component
    growth_rate: float                  # G component
    neighbor_pressure: float            # N component
    weather_anomaly: float              # W component
    road_vulnerability: float           # V component


class MSIResponse(BaseModel):
    num_zones: int
    p50_msi: float
    p80_msi: float
    zones: list[MSIZone]


# ──────────────────────────────────────────────
# /zones  — GET
# ──────────────────────────────────────────────

class ZoneInfo(BaseModel):
    zone_id: int
    centroid_lat: float
    centroid_lon: float
    num_neighbors: int


class ZonesResponse(BaseModel):
    num_zones: int
    zones: list[ZoneInfo]


# ──────────────────────────────────────────────
# /health  — GET
# ──────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_zones: int
    num_features: int
    pipeline_ready: bool


# ──────────────────────────────────────────────
# /complaints  — POST
# ──────────────────────────────────────────────

class ComplaintIn(BaseModel):
    created_at: str = Field(description="Timestamp e.g. '01/15/2024 10:30'")
    latitude: float = Field(ge=12.0, le=14.0, description="Latitude (Bangalore range)")
    longitude: float = Field(ge=77.0, le=78.0, description="Longitude (Bangalore range)")
    category_id: int = Field(default=0)
    category_title: str = Field(default="Others")
    sub_category_title: str = Field(default="Others")
    complaint_status_title: str = Field(default="Open")
    ward_id: int = Field(default=0)
    ward_title: str = Field(default="Unknown")
    civic_agency_title: str = Field(default="BBMP")
    comment_count: int = Field(default=0)
    description: str = Field(default="")


class ComplaintIngestResponse(BaseModel):
    status: str
    zone_id: int
    total_complaints: int
    message: str


# ──────────────────────────────────────────────
# /metrics  — GET
# ──────────────────────────────────────────────

class MetricsResponse(BaseModel):
    mae: float
    rmse: float
    r2: float
    f1_macro: float
    f1_low: float
    f1_medium: float
    f1_high: float
    lead_time_accuracy: float
    total_spike_zones: int
    mean_lead_error_steps: float
