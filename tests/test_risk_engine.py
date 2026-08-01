"""
Tests for Dynamic Risk Engine Module
"""

import pytest
import numpy as np
from src.risk_engine import (
    softmax_weights,
    compute_risk_raw,
    apply_ema,
    classify_risk,
    explain_risk,
    RiskEngine,
)


class TestSoftmaxWeights:
    def test_weights_sum_to_one(self):
        w_u, w_d, w_p, w_w, w_v = softmax_weights(0.8, 0.5, 0.3)
        assert abs(w_u + w_d + w_p + w_w + w_v - 1.0) < 1e-6

    def test_all_weights_positive(self):
        w_u, w_d, w_p, w_w, w_v = softmax_weights(0.0, 0.0, 0.0)
        assert w_u > 0
        assert w_d > 0
        assert w_p > 0
        assert w_w > 0
        assert w_v > 0

    def test_higher_input_gets_higher_weight(self):
        w_u, w_d, w_p, w_w, w_v = softmax_weights(0.9, 0.1, 0.1)
        assert w_u > w_d
        assert w_u > w_p

    def test_equal_inputs_equal_weights(self):
        w_u, w_d, w_p, w_w, w_v = softmax_weights(0.5, 0.5, 0.5)
        # For equal inputs under standard softmax, all weights should be mathematically equal
        assert abs(w_u - w_d) < 1e-6
        assert abs(w_p - w_u) < 1e-6


class TestRiskRaw:
    def test_basic_computation(self):
        risk = compute_risk_raw(0.5, 0.5, 0.5, 0.33, 0.33, 0.34)
        assert 0 <= risk <= 1

    def test_zero_inputs(self):
        risk = compute_risk_raw(0, 0, 0, 0.33, 0.33, 0.34)
        assert risk == 0.0


class TestEMA:
    def test_first_step_returns_raw(self):
        result = apply_ema(0.7, None, alpha=0.3)
        assert result == 0.7

    def test_smoothing_effect(self):
        """EMA should be between raw and previous."""
        result = apply_ema(0.8, 0.2, alpha=0.3)
        assert 0.2 < result < 0.8

    def test_alpha_formula(self):
        result = apply_ema(0.8, 0.2, alpha=0.3)
        expected = 0.3 * 0.8 + 0.7 * 0.2
        assert abs(result - expected) < 1e-6

    def test_converges_to_raw(self):
        """Repeated EMA with same value should converge."""
        risk = 0.0
        for _ in range(100):
            risk = apply_ema(0.7, risk, alpha=0.3)
        assert abs(risk - 0.7) < 0.01


class TestClassifyRisk:
    def test_low_risk(self):
        assert classify_risk(0.1) == "Low"
        assert classify_risk(0.29) == "Low"

    def test_medium_risk(self):
        assert classify_risk(0.3) == "Medium"
        assert classify_risk(0.5) == "Medium"
        assert classify_risk(0.69) == "Medium"

    def test_high_risk(self):
        assert classify_risk(0.7) == "High"
        assert classify_risk(1.0) == "High"

    def test_custom_thresholds(self):
        assert classify_risk(0.4, thresholds=(0.5, 0.8)) == "Low"
        assert classify_risk(0.6, thresholds=(0.5, 0.8)) == "Medium"
        assert classify_risk(0.9, thresholds=(0.5, 0.8)) == "High"


class TestExplainRisk:
    def test_contributions_sum_to_100(self):
        result = explain_risk(0.4, 0.3, 0.3, 0.8, 0.5, 0.2)
        pct = result["contribution_pct"]
        total = sum(pct.values())
        assert abs(total - 100.0) < 0.5  # rounding tolerance

    def test_primary_driver(self):
        result = explain_risk(0.5, 0.3, 0.2, 0.9, 0.1, 0.1)
        assert result["primary_driver"] == "unresolved"

    def test_has_explanation_text(self):
        result = explain_risk(0.33, 0.33, 0.34, 0.5, 0.5, 0.5)
        assert isinstance(result["explanation"], str)


class TestRiskEngine:
    def test_compute_zone_risk(self):
        engine = RiskEngine(num_zones=3)
        result = engine.compute_zone_risk(0, U=0.8, D=0.5, P=0.3)
        assert "risk_score" in result
        assert "risk_level" in result
        assert "explanation" in result
        assert 0 <= result["risk_score"] <= 1

    def test_ema_tracking(self):
        engine = RiskEngine(num_zones=1, alpha=0.3)
        r1 = engine.compute_zone_risk(0, U=0.9, D=0.9, P=0.9)
        r2 = engine.compute_zone_risk(0, U=0.1, D=0.1, P=0.1)
        # Second score should be lower than first but still affected by first
        assert r2["risk_score"] < r1["risk_score"]
        assert r2["risk_score"] > 0  # EMA keeps some history

    def test_compute_all_zones(self):
        engine = RiskEngine(num_zones=3)
        results = engine.compute_all_zones(
            U_values=np.array([0.8, 0.2, 0.5]),
            D_values=np.array([0.6, 0.3, 0.4]),
            P_values=np.array([0.7, 0.1, 0.3]),
        )
        assert len(results) == 3
        for r in results:
            assert 0 <= r["risk_score"] <= 1

    def test_reset(self):
        engine = RiskEngine(num_zones=2)
        engine.compute_zone_risk(0, U=0.9, D=0.9, P=0.9)
        engine.reset()
        assert engine.prev_risks[0] is None
