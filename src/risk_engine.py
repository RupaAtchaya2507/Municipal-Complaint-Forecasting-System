"""
Dynamic Risk Engine Module
==========================
Softmax-based weighting, EMA smoothing, risk classification,
and component-level explanations.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def softmax_weights(U: float, D: float, P: float, W: float = 0.0, V: float = 0.0) -> tuple:
    """
    Compute dynamic weights using softmax over all 5 MSI components.

    Guarantees all weights sum to 1 and are in (0, 1).

    Args:
        U: unresolved ratio ∈ [0, 1]
        D: complaint density/surge ∈ [0, 1]
        P: model prediction ∈ [0, 1]
        W: weather anomaly score ∈ [0, 1]
        V: road vulnerability score ∈ [0, 1]

    Returns:
        (w_u, w_d, w_p, w_w, w_v)
    """
    scores = np.array([U, D, P, W, V], dtype=np.float64)
    weights = softmax(scores)
    return tuple(float(w) for w in weights)


def compute_risk_raw(
    U: float, D: float, P: float,
    w_u: float, w_d: float, w_p: float,
    W: float = 0.0, V: float = 0.0,
    w_w: float = 0.0, w_v: float = 0.0,
) -> float:
    """
    Compute raw risk score as weighted sum of all MSI components.

    Risk_raw = w_u*U + w_d*D + w_p*P + w_w*W + w_v*V
    """
    return w_u * U + w_d * D + w_p * P + w_w * W + w_v * V


def apply_ema(
    risk_raw: float,
    risk_prev: float,
    alpha: float = 0.3,
) -> float:
    """
    Apply Exponential Moving Average for temporal smoothing.
    
    Risk_t = α * Risk_raw + (1 - α) * Risk_{t-1}
    
    Args:
        risk_raw: current raw risk score
        risk_prev: previous smoothed risk score (None for first step)
        alpha: smoothing factor (higher = more responsive to current)
    """
    if risk_prev is None:
        return risk_raw
    return alpha * risk_raw + (1 - alpha) * risk_prev


def classify_risk(
    score: float,
    thresholds: tuple = (0.3, 0.7),
) -> str:
    """
    Classify risk score into levels.
    
    Args:
        score: risk score ∈ [0, 1]
        thresholds: (low/medium boundary, medium/high boundary)
    
    Returns:
        'Low', 'Medium', or 'High'
    """
    low_thresh, high_thresh = thresholds
    if score < low_thresh:
        return "Low"
    elif score < high_thresh:
        return "Medium"
    else:
        return "High"


def explain_risk(
    w_u: float, w_d: float, w_p: float,
    U: float, D: float, P: float,
    W: float = 0.0, V: float = 0.0,
    w_w: float = 0.0, w_v: float = 0.0,
) -> dict:
    """
    Generate component-level contribution explanations for all 5 MSI components.

    Returns dict with:
      - contribution_pct: percentage each component contributes
      - primary_driver: the dominant factor
      - explanation: human-readable text
    """
    contributions = {
        "unresolved":        w_u * U,
        "density":           w_d * D,
        "prediction":        w_p * P,
        "weather_anomaly":   w_w * W,
        "road_vulnerability": w_v * V,
    }

    total = sum(contributions.values())
    if total == 0:
        pct = {k: 0.0 for k in contributions}
    else:
        pct = {k: round(v / total * 100, 1) for k, v in contributions.items()}

    primary = max(contributions, key=contributions.get)

    explanations = {
        "unresolved":         "High unresolved complaints driving risk",
        "density":            "Recent complaint surge driving risk",
        "prediction":         "Model predicts elevated future risk",
        "weather_anomaly":    "Heavy rainfall / weather anomaly elevating risk",
        "road_vulnerability": "Poor road quality and infrastructure vulnerability",
    }

    return {
        "contribution_pct":  pct,
        "primary_driver":    primary,
        "explanation":       explanations[primary],
        "raw_contributions": contributions,
    }


class RiskEngine:
    """
    Stateful Dynamic Risk Engine.

    Tracks previous risk scores per zone for EMA smoothing.
    Supports all 5 MSI components:
      U  — Unresolved Ratio
      D  — Complaint Density / Surge
      P  — Model Prediction
      W  — Weather Anomaly Score  (Module 5)
      V  — Road Vulnerability Score (Module 5)
    """

    def __init__(
        self,
        num_zones: int,
        alpha: float = 0.3,
        thresholds: tuple = (0.3, 0.7),
        weighting_method: str = "dynamic",
        static_weights: tuple = (0.25, 0.20, 0.20, 0.15, 0.10, 0.10),
    ):
        """
        Args:
            static_weights: (w_u, w_d, w_p, w_g, w_w, w_v) used when weighting_method='static'.
                            Defaults match the Module 5 MSI formula weights.
        """
        self.num_zones = num_zones
        self.alpha = alpha
        self.thresholds = thresholds
        self.weighting_method = weighting_method
        self.static_weights = static_weights

        self.prev_risks = {z: None for z in range(num_zones)}
        self.prev_preds = {z: None for z in range(num_zones)}

        logger.info(
            f"RiskEngine initialised: {num_zones} zones, "
            f"α={alpha}, thresholds={thresholds}, weighting={weighting_method}"
        )

    def compute_zone_risk(
        self,
        zone_id: int,
        U: float,
        D: float,
        P: float,
        W: float = 0.0,
        V: float = 0.0,
    ) -> dict:
        """
        Compute risk for a single zone.

        Args:
            zone_id: integer zone index
            U: unresolved ratio ∈ [0,1]
            D: complaint density/surge ∈ [0,1]
            P: model MSI prediction ∈ [0,1]
            W: weather anomaly score ∈ [0,1]  (default 0)
            V: road vulnerability score ∈ [0,1] (default 0)

        Returns dict with: zone_id, risk_score, risk_raw, risk_level,
                           weights, msi_components, explanation
        """
        # EMA smoothing on prediction
        P_smoothed = apply_ema(P, self.prev_preds[zone_id], self.alpha)
        self.prev_preds[zone_id] = P_smoothed

        # Weights
        if self.weighting_method == "dynamic":
            w_u, w_d, w_p, w_w, w_v = softmax_weights(U, D, P_smoothed, W, V)
        else:
            # Static weights tuned to MSI formula: U=0.25, D=0.20, P=0.30, W=0.05, V=0.20
            sw = self.static_weights
            if len(sw) >= 5:
                w_u, w_d, w_p, w_w, w_v = sw[0], sw[1], sw[2], sw[3], sw[4]
            else:
                w_u, w_d, w_p = sw[0], sw[1], sw[2]
                w_w = w_v = 0.0
            # renormalise so they sum to 1
            total_w = w_u + w_d + w_p + w_w + w_v
            if total_w > 0:
                w_u /= total_w; w_d /= total_w; w_p /= total_w
                w_w /= total_w; w_v /= total_w

        risk_raw = compute_risk_raw(U, D, P_smoothed, w_u, w_d, w_p, W, V, w_w, w_v)
        risk_smoothed = apply_ema(risk_raw, self.prev_risks[zone_id], self.alpha)
        self.prev_risks[zone_id] = risk_smoothed

        risk_level = classify_risk(risk_smoothed, self.thresholds)
        explanation = explain_risk(w_u, w_d, w_p, U, D, P_smoothed, W, V, w_w, w_v)

        return {
            "zone_id":   zone_id,
            "risk_score": round(risk_smoothed, 4),
            "risk_raw":   round(risk_raw, 4),
            "risk_level": risk_level,
            "weights": {
                "w_u": round(w_u, 4), "w_d": round(w_d, 4), "w_p": round(w_p, 4),
                "w_w": round(w_w, 4), "w_v": round(w_v, 4),
            },
            "msi_components": {
                "U_unresolved":        round(U, 4),
                "D_density":           round(D, 4),
                "P_prediction":        round(float(P_smoothed), 4),
                "W_weather_anomaly":   round(W, 4),
                "V_road_vulnerability": round(V, 4),
            },
            "explanation": explanation,
        }

    def compute_all_zones(
        self,
        U_values: np.ndarray,
        D_values: np.ndarray,
        P_values: np.ndarray,
        W_values: np.ndarray = None,
        V_values: np.ndarray = None,
    ) -> list:
        """
        Compute risk for all zones at a single time step.

        Args:
            U_values: [N] unresolved ratios
            D_values: [N] densities
            P_values: [N] model predictions
            W_values: [N] weather anomaly scores (optional, defaults to zeros)
            V_values: [N] road vulnerability scores (optional, defaults to zeros)

        Returns:
            List of risk dicts (one per zone)
        """
        if W_values is None:
            W_values = np.zeros(self.num_zones)
        if V_values is None:
            V_values = np.zeros(self.num_zones)

        results = []
        for z in range(self.num_zones):
            result = self.compute_zone_risk(
                zone_id=z,
                U=float(U_values[z]),
                D=float(D_values[z]),
                P=float(P_values[z]),
                W=float(W_values[z]),
                V=float(V_values[z]),
            )
            results.append(result)

        scores = [r["risk_score"] for r in results]
        levels = [r["risk_level"] for r in results]
        logger.info(
            f"Risk computed: High={levels.count('High')}, "
            f"Medium={levels.count('Medium')}, Low={levels.count('Low')}, "
            f"mean_score={np.mean(scores):.4f}"
        )
        return results

    def reset(self):
        """Reset EMA history."""
        self.prev_risks = {z: None for z in range(self.num_zones)}
        self.prev_preds = {z: None for z in range(self.num_zones)}
        logger.info("RiskEngine reset")
