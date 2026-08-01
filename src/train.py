"""
Training Module
===============
Focal Loss, training loop, evaluation, early stopping,
and sequence length experimentation.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import os
from tqdm import tqdm

from src.utils import compute_metrics, save_model

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    When γ > 0, reduces loss contribution from easy (well-classified) examples,
    focusing training on hard examples.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: predictions ∈ [0,1], shape [batch, N]
            target: binary labels, shape [batch, N]
        """
        # Clamp to avoid log(0)
        pred = pred.clamp(1e-7, 1 - 1e-7)

        # Binary cross entropy components
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)

        # p_t
        p_t = target * pred + (1 - target) * (1 - pred)

        # Alpha weighting
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)

        # Focal modulation
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)

        loss = focal_weight * bce
        return loss.mean()


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    loss_fn,
    adj: torch.Tensor,
    device: torch.device,
    category_ids: torch.Tensor = None,
    threshold: float = 0.5,
) -> dict:
    """
    Train for one epoch in regression mode.
    
    Returns dict with: loss, mae, rmse, r2
    """
    from src.model import MultiTaskSpatioTemporalModel
    is_multitask = isinstance(model, MultiTaskSpatioTemporalModel)
    
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        
        if is_multitask:
            pred_msi, pred_cnt, pred_unres = model(X_batch, adj, category_ids)
            # Auxiliary targets extracted from last step of feature sequences
            target_cnt = X_batch[:, -1, :, 0]
            target_unres = X_batch[:, -1, :, 1] / torch.clamp(X_batch[:, -1, :, 0], min=1.0)
            
            l_msi = loss_fn(pred_msi, y_batch)
            l_cnt = loss_fn(pred_cnt, target_cnt)
            l_unres = loss_fn(pred_unres, target_unres)

            try:
                import config as _cfg
                w_msi  = getattr(_cfg, "MSI_LOSS_WEIGHT",   0.6)
                w_cnt  = getattr(_cfg, "COUNT_LOSS_WEIGHT", 0.2)
                w_unres = getattr(_cfg, "UNRES_LOSS_WEIGHT", 0.2)
            except Exception:
                w_msi, w_cnt, w_unres = 0.6, 0.2, 0.2

            loss = w_msi * l_msi + w_cnt * l_cnt + w_unres * l_unres
            pred = pred_msi
        else:
            pred = model(X_batch, adj, category_ids)
            loss = loss_fn(pred, y_batch)
            
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X_batch)

        # Collect predictions
        preds = pred.detach().cpu().numpy()
        targets = y_batch.detach().cpu().numpy()

        all_preds.append(preds.flatten())
        all_targets.append(targets.flatten())

    # Aggregate
    avg_loss = total_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    metrics = compute_metrics(all_targets, all_preds, regression=True)
    metrics["loss"] = avg_loss
    # Backwards compatibility: Map R2 to F1 for learning curves plotter
    metrics["f1"] = metrics["r2"]

    return metrics


def evaluate(
    model: nn.Module,
    loader,
    loss_fn,
    adj: torch.Tensor,
    device: torch.device,
    category_ids: torch.Tensor = None,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate model on a dataset in regression mode.
    
    Returns dict with: loss, mae, rmse, r2, targets, probs (predictions)
    """
    from src.model import MultiTaskSpatioTemporalModel
    is_multitask = isinstance(model, MultiTaskSpatioTemporalModel)
    
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            if is_multitask:
                pred_msi, pred_cnt, pred_unres = model(X_batch, adj, category_ids)
                target_cnt = X_batch[:, -1, :, 0]
                target_unres = X_batch[:, -1, :, 1] / torch.clamp(X_batch[:, -1, :, 0], min=1.0)
                
                l_msi = loss_fn(pred_msi, y_batch)
                l_cnt = loss_fn(pred_cnt, target_cnt)
                l_unres = loss_fn(pred_unres, target_unres)

                try:
                    import config as _cfg
                    w_msi  = getattr(_cfg, "MSI_LOSS_WEIGHT",   0.6)
                    w_cnt  = getattr(_cfg, "COUNT_LOSS_WEIGHT", 0.2)
                    w_unres = getattr(_cfg, "UNRES_LOSS_WEIGHT", 0.2)
                except Exception:
                    w_msi, w_cnt, w_unres = 0.6, 0.2, 0.2

                loss = w_msi * l_msi + w_cnt * l_cnt + w_unres * l_unres
                pred = pred_msi
            else:
                pred = model(X_batch, adj, category_ids)
                loss = loss_fn(pred, y_batch)

            total_loss += loss.item() * len(X_batch)

            preds = pred.cpu().numpy()
            targets = y_batch.cpu().numpy()

            all_preds.append(preds.flatten())
            all_targets.append(targets.flatten())

    avg_loss = total_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    metrics = compute_metrics(all_targets, all_preds, regression=True)
    metrics["loss"] = avg_loss
    metrics["targets"] = all_targets
    metrics["probs"] = all_preds  # Return predictions as probs for backward compatibility
    # Backwards compatibility: Map R2 to F1 for learning curves plotter
    metrics["f1"] = metrics["r2"]

    return metrics


def find_optimal_threshold(targets: np.ndarray, probs: np.ndarray) -> tuple:
    """Unused for regression, return dummy values for pipeline compatibility."""
    return 0.5, 0.0


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    adj: torch.Tensor,
    device: torch.device,
    config: dict,
    category_ids: torch.Tensor = None,
    save_path: str = None,
) -> dict:
    """
    Full training loop for spatiotemporal MSI regression.
    
    Optimizes MSE Loss, scheduling on validation loss reduction and early stopping.
    """
    # Use dynamic loss function based on configuration
    loss_type = config.get("loss_type", "mse")
    if loss_type == "huber":
        loss_fn = nn.HuberLoss()
    elif loss_type == "smooth_l1":
        loss_fn = nn.SmoothL1Loss()
    else:
        loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("lr", 1e-3),
        weight_decay=config.get("weight_decay", 1e-4),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",  # Minimize validation MSE loss
        factor=config.get("lr_factor", 0.5),
        patience=config.get("lr_patience", 5),
    )

    max_epochs = config.get("max_epochs", 200)
    early_stop_patience = config.get("early_stop_patience", 15)

    # Training history
    history = {
        "train_loss": [], "train_f1": [],
        "val_loss": [], "val_f1": [], "val_auc_roc": [],
        "lr": [],
    }

    best_val_loss = float('inf')
    patience_counter = 0

    logger.info(f"Starting spatiotemporal regression training: {max_epochs} max epochs, "
                f"early stop patience={early_stop_patience}")

    for epoch in range(1, max_epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, adj, device, category_ids
        )
        val_metrics = evaluate(
            model, val_loader, loss_fn, adj, device, category_ids
        )

        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history (val_auc_roc gets mapped to RMSE for plotter visual separation)
        history["train_loss"].append(train_metrics["loss"])
        history["train_f1"].append(train_metrics["f1"]) # R2 score
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"]) # R2 score
        history["val_auc_roc"].append(val_metrics.get("rmse", 0.0))
        history["lr"].append(current_lr)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:3d} | "
                f"Train Loss={train_metrics['loss']:.4f} R2={train_metrics['r2']:.4f} | "
                f"Val Loss={val_metrics['loss']:.4f} R2={val_metrics['r2']:.4f} "
                f"RMSE={val_metrics.get('rmse', 0):.4f} | "
                f"LR={current_lr:.6f}"
              )

        # Early stopping check based on minimizing validation MSE loss
        is_best = val_metrics["loss"] < best_val_loss

        if is_best:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            if save_path:
                save_model(model, save_path)
                logger.info(f"  ★ New best model saved (Val Loss={best_val_loss:.4f}, R2={val_metrics['r2']:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch} "
                        f"(no improvement for {early_stop_patience} epochs)")
            break

    logger.info(f"Training complete. Best val Loss={best_val_loss:.4f}")
    
    # Store best_val_f1 as best loss for compatibility (in sequence experiment)
    history["best_val_f1"] = best_val_loss
    history["optimal_threshold"] = 0.5
    history["optimal_threshold_f1"] = 0.0

    return history


def experiment_seq_lengths(
    feature_tensor: np.ndarray,
    adj: torch.Tensor,
    device: torch.device,
    model_class,
    model_kwargs: dict,
    config: dict,
    seq_lengths: list = None,
    adjacency_matrix: np.ndarray = None,
) -> dict:
    """
    Compare different sequence lengths in regression mode.
    
    Ranks sequence lengths by minimizing validation loss.
    """
    from src.dataset import create_sequences, get_dataloaders

    if seq_lengths is None:
        seq_lengths = [3, 5, 7]

    results = {}

    for seq_len in seq_lengths:
        logger.info(f"\n{'='*50}")
        logger.info(f"Experimenting with seq_len = {seq_len}")
        logger.info(f"{'='*50}")

        try:
            X, y = create_sequences(
                feature_tensor, seq_len,
                adjacency_matrix=adjacency_matrix
            )
            train_loader, val_loader, test_loader = get_dataloaders(
                X, y, batch_size=config.get("batch_size", 32)
            )

            model = model_class(**model_kwargs).to(device)

            history = train_model(
                model, train_loader, val_loader, adj, device, config,
                save_path=None
            )

            # Map best_val_f1 to -loss so that the max() selector in main.py 
            # correctly identifies the MINIMUM validation loss sequence length.
            results[seq_len] = {
                "best_val_f1": -history["best_val_f1"],
                "final_val_loss": history["val_loss"][-1],
                "epochs_trained": len(history["train_loss"]),
            }

            logger.info(f"seq_len={seq_len}: best_val_loss={history['best_val_f1']:.4f}")

        except ValueError as e:
            logger.warning(f"seq_len={seq_len} failed: {e}")
            results[seq_len] = {"error": str(e)}

    # Report best (minimum raw loss)
    valid_results = {k: v for k, v in results.items() if "best_val_f1" in v}
    if valid_results:
        best_seq = max(valid_results, key=lambda k: valid_results[k]["best_val_f1"])
        logger.info(f"\n★ Best sequence length: {best_seq} "
                    f"(Val Loss={-valid_results[best_seq]['best_val_f1']:.4f})")

    return results
