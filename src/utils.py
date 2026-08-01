"""
Shared Utilities
================
Logging, seed setting, model save/load, and metric helpers.
"""

import os
import random
import logging
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def setup_logging(level=logging.INFO):
    """Configure logging for the project."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_model(model: torch.nn.Module, path: str):
    """Save model state dict."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logging.getLogger(__name__).info(f"Model saved to: {path}")


def load_model(model: torch.nn.Module, path: str, device: torch.device = None):
    """Load model state dict."""
    if device is None:
        device = get_device()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    logging.getLogger(__name__).info(f"Model loaded from: {path}")
    return model


def compute_metrics(y_true, y_pred, y_prob=None, regression=False) -> dict:
    """
    Compute classification or regression metrics.

    Args:
        y_true: ground truth labels
        y_pred: predicted labels / continuous values
        y_prob: predicted probabilities (for AUC-ROC in classification)
        regression: if True, computes regression metrics (MAE, RMSE, R2)

    Returns:
        dict with computed metrics
    """
    if regression:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))
        return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["auc_roc"] = 0.0

    return metrics


def msi_to_risk_label(msi_scores: np.ndarray, thresholds: tuple = (0.3, 0.7)) -> np.ndarray:
    """
    Convert continuous MSI scores to discrete risk labels.

    Labels:
      0 = Low   (MSI < low_thresh)
      1 = Medium (low_thresh <= MSI < high_thresh)
      2 = High  (MSI >= high_thresh)

    Args:
        msi_scores: array of MSI values in [0, 1]
        thresholds: (low_thresh, high_thresh)

    Returns:
        integer label array (0, 1, or 2)
    """
    low_thresh, high_thresh = thresholds
    labels = np.zeros(len(msi_scores), dtype=int)
    labels[msi_scores >= low_thresh] = 1
    labels[msi_scores >= high_thresh] = 2
    return labels


def compute_risk_classification_metrics(
    y_true_msi: np.ndarray,
    y_pred_msi: np.ndarray,
    thresholds: tuple = (0.3, 0.7),
) -> dict:
    """
    Compute real F1 score for the 3-class risk classification task.

    Converts continuous MSI predictions and ground truth into
    Low / Medium / High labels, then computes per-class and
    macro-averaged F1.

    Args:
        y_true_msi: ground truth MSI values [N]
        y_pred_msi: predicted MSI values [N]
        thresholds: (low_thresh, high_thresh) matching config.RISK_THRESHOLDS

    Returns:
        dict with:
          - f1_macro: macro-averaged F1 across 3 classes
          - f1_low, f1_medium, f1_high: per-class F1
          - precision_macro, recall_macro
          - confusion_matrix: 3x3 numpy array
    """
    y_true_msi = np.array(y_true_msi).flatten()
    y_pred_msi = np.array(y_pred_msi).flatten()

    y_true_labels = msi_to_risk_label(y_true_msi, thresholds)
    y_pred_labels = msi_to_risk_label(y_pred_msi, thresholds)

    f1_per_class = f1_score(y_true_labels, y_pred_labels, labels=[0, 1, 2],
                            average=None, zero_division=0)
    f1_macro = f1_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)
    precision_macro = precision_score(y_true_labels, y_pred_labels,
                                      average="macro", zero_division=0)
    recall_macro = recall_score(y_true_labels, y_pred_labels,
                                average="macro", zero_division=0)
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=[0, 1, 2])

    logger = logging.getLogger(__name__)
    logger.info(
        f"Risk Classification F1 — Low: {f1_per_class[0]:.4f}, "
        f"Medium: {f1_per_class[1]:.4f}, High: {f1_per_class[2]:.4f}, "
        f"Macro: {f1_macro:.4f}"
    )

    return {
        "f1_macro": float(f1_macro),
        "f1_low": float(f1_per_class[0]),
        "f1_medium": float(f1_per_class[1]),
        "f1_high": float(f1_per_class[2]),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "confusion_matrix": cm,
    }


def compute_lead_time_accuracy(
    y_true_msi: np.ndarray,
    y_pred_msi: np.ndarray,
    timestamps: np.ndarray,
    high_thresh: float = 0.7,
    tolerance_steps: int = 1,
) -> dict:
    """
    Measure how accurately the model predicts the TIMING of high-risk spikes.

    For each zone, finds the first time step where actual MSI crosses
    high_thresh (the spike) and checks if the model predicted it within
    +/- tolerance_steps windows.

    Args:
        y_true_msi: actual MSI array [T, N] (time x zones)
        y_pred_msi: predicted MSI array [T, N]
        timestamps: array of time labels [T] (for reporting)
        high_thresh: threshold above which a zone is considered high-risk
        tolerance_steps: allowed window offset (default ±1 step)

    Returns:
        dict with:
          - lead_time_accuracy: fraction of spikes correctly detected within tolerance
          - detected_zones: list of zone ids where spike was caught in time
          - missed_zones: list of zone ids where spike was missed
          - mean_lead_error_steps: mean absolute step error for detected spikes
    """
    y_true_msi = np.array(y_true_msi)   # [T, N]
    y_pred_msi = np.array(y_pred_msi)   # [T, N]
    T, N = y_true_msi.shape

    detected, missed = [], []
    lead_errors = []

    for z in range(N):
        # Find first actual spike
        true_spike_steps = np.where(y_true_msi[:, z] >= high_thresh)[0]
        if len(true_spike_steps) == 0:
            continue  # zone never went high-risk — skip

        first_true_spike = true_spike_steps[0]

        # Find first predicted spike
        pred_spike_steps = np.where(y_pred_msi[:, z] >= high_thresh)[0]
        if len(pred_spike_steps) == 0:
            missed.append(z)
            continue

        first_pred_spike = pred_spike_steps[0]
        error = abs(int(first_pred_spike) - int(first_true_spike))

        if error <= tolerance_steps:
            detected.append(z)
            lead_errors.append(error)
        else:
            missed.append(z)

    total_spikes = len(detected) + len(missed)
    accuracy = len(detected) / total_spikes if total_spikes > 0 else 0.0
    mean_error = float(np.mean(lead_errors)) if lead_errors else float("nan")

    logger = logging.getLogger(__name__)
    logger.info(
        f"Lead Time Accuracy: {accuracy:.4f} "
        f"({len(detected)}/{total_spikes} spikes detected within ±{tolerance_steps} steps), "
        f"mean lead error={mean_error:.2f} steps"
    )

    return {
        "lead_time_accuracy": float(accuracy),
        "detected_zones": detected,
        "missed_zones": missed,
        "mean_lead_error_steps": mean_error,
        "total_spike_zones": total_spikes,
    }
