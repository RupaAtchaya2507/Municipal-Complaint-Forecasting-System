"""
Production Spatiotemporal Inference Module
===========================================
Handles live spatiotemporal forecasting inference, model loading,
Delta MSI reconstruction, and Risk Engine integration.
"""

import os
import torch
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Union, Dict

import config
from src.model import SpatioTemporalModel, MultiTaskSpatioTemporalModel
from src.risk_engine import RiskEngine

logger = logging.getLogger(__name__)

def load_production_model(
    model_path: str,
    num_features: int,
    num_zones: int,
    model_type: str = "multi_task",
    use_sigmoid: bool = False,
    device: torch.device = None
) -> torch.nn.Module:
    """
    Load a trained model (either single-task or multi-task) from disk.
    
    Args:
        model_path: path to the saved model checkpoint (.pt or .pth)
        num_features: number of features in the input sequence
        num_zones: number of zones in the graph
        model_type: "single_task" or "multi_task"
        use_sigmoid: whether the model uses a Sigmoid output layer
        device: torch.device to load onto
        
    Returns:
        Loaded PyTorch model in eval mode
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    logger.info(f"Loading {model_type} model from {model_path} onto {device}...")
    
    base_model = SpatioTemporalModel(
        num_features=num_features,
        num_zones=num_zones,
        gcn_hidden=config.GCN_HIDDEN_DIM,
        lstm_hidden=config.LSTM_HIDDEN_DIM,
        lstm_layers=config.LSTM_NUM_LAYERS,
        dropout=config.DROPOUT_RATE,
        use_sigmoid=use_sigmoid
    )
    
    if model_type == "multi_task":
        model = MultiTaskSpatioTemporalModel(base_model)
    else:
        model = base_model
        
    checkpoint = torch.load(model_path, map_location=device)
    # Support loading both state dicts and full model objects
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint
        
    model = model.to(device)
    model.eval()
    logger.info("Model loaded successfully and set to evaluation mode.")
    return model

def run_inference(
    model: torch.nn.Module,
    X_seq: np.ndarray,
    adj_matrix: np.ndarray,
    device: torch.device = None,
    category_ids: torch.Tensor = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run raw forward pass on spatiotemporal sequence.
    
    Args:
        model: loaded SpatioTemporalModel or MultiTaskSpatioTemporalModel
        X_seq: input sequence of shape [batch, seq_len, N, F]
        adj_matrix: graph adjacency matrix of shape [N, N]
        device: torch.device to execute on
        category_ids: optional category embeddings
        
    Returns:
        preds_msi: predicted MSI or Delta MSI values [batch, N]
        preds_count: predicted future complaint counts [batch, N] (None for single-task)
        preds_unres: predicted future unresolved ratio [batch, N] (None for single-task)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Convert inputs to tensors
    X_tensor = torch.FloatTensor(X_seq).to(device)
    adj_tensor = torch.FloatTensor(adj_matrix).to(device)
    
    is_multitask = isinstance(model, MultiTaskSpatioTemporalModel)
    
    with torch.no_grad():
        if is_multitask:
            pred_msi, pred_cnt, pred_unres = model(X_tensor, adj_tensor, category_ids)
            return (
                pred_msi.cpu().numpy(),
                pred_cnt.cpu().numpy(),
                pred_unres.cpu().numpy()
            )
        else:
            pred_msi = model(X_tensor, adj_tensor, category_ids)
            return pred_msi.cpu().numpy(), None, None

def reconstruct_future_msi(
    pred_delta: np.ndarray,
    msi_prev: np.ndarray
) -> np.ndarray:
    """
    Reconstruct Future MSI from predicted Delta MSI and previous step MSI.
    Formula: MSI_t = Predicted_Delta_t + MSI_{t-1}
    
    Args:
        pred_delta: predicted Delta MSI values, shape [batch, N]
        msi_prev: actual/derived previous MSI values, shape [batch, N]
        
    Returns:
        reconstructed_msi: reconstructed Future MSI values, shape [batch, N]
    """
    reconstructed = pred_delta + msi_prev
    return reconstructed

def inference_pipeline(
    model_path: str,
    X_seq: np.ndarray,
    adj_matrix: np.ndarray,
    msi_prev: np.ndarray,
    U_values: np.ndarray,
    D_values: np.ndarray,
    model_type: str = "multi_task",
    predict_delta: bool = True,
    use_sigmoid: bool = False,
    risk_engine: RiskEngine = None
) -> Dict:
    """
    Run full end-to-end production inference pipeline.
    
    Args:
        model_path: path to trained model weights
        X_seq: input sequence of shape [batch, seq_len, N, F]
        adj_matrix: adjacency matrix [N, N]
        msi_prev: previous step actual/derived MSI [N] or [batch, N]
        U_values: current unresolved ratios [N]
        D_values: current surge/density values [N]
        model_type: "single_task" or "multi_task"
        predict_delta: whether the loaded model targets Delta MSI
        use_sigmoid: whether to include output sigmoid layer
        risk_engine: initialized stateful RiskEngine
        
    Returns:
        dictionary containing forecasts, reconstructed MSIs, and risk scores
    """
    num_zones = adj_matrix.shape[0]
    num_features = X_seq.shape[3]
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_production_model(
        model_path=model_path,
        num_features=num_features,
        num_zones=num_zones,
        model_type=model_type,
        use_sigmoid=use_sigmoid,
        device=device
    )
    
    # Run forward pass
    preds_msi_raw, preds_cnt, preds_unres = run_inference(
        model=model,
        X_seq=X_seq,
        adj_matrix=adj_matrix,
        device=device
    )
    
    # Take latest sequence projection for final forecasting steps
    latest_pred_raw = preds_msi_raw[-1]  # [N]
    
    # Delta to MSI reconstruction
    if predict_delta:
        # Align previous step MSI values
        prev_msi_val = msi_prev[-1] if len(msi_prev.shape) > 1 else msi_prev
        predicted_msi = reconstruct_future_msi(latest_pred_raw, prev_msi_val)
        logger.info("Delta MSI successfully reconstructed to Future MSI.")
    else:
        predicted_msi = latest_pred_raw
        logger.info("Direct Future MSI predictions returned.")
        
    # Risk Engine calculation
    if risk_engine is None:
        risk_engine = RiskEngine(
            num_zones=num_zones,
            alpha=config.EMA_ALPHA,
            thresholds=config.RISK_THRESHOLDS
        )
        
    risk_results = risk_engine.compute_all_zones(
        U_values=U_values,
        D_values=D_values,
        P_values=predicted_msi
    )
    
    return {
        "raw_model_predictions": latest_pred_raw,
        "predicted_msi": predicted_msi,
        "predicted_counts": preds_cnt[-1] if preds_cnt is not None else None,
        "predicted_unresolved_ratios": preds_unres[-1] if preds_unres is not None else None,
        "risk_assessments": risk_results
    }
