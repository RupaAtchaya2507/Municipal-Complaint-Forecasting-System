import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def ensure_dir(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

def plot_spatial_clusters(df: pd.DataFrame, centroids: np.ndarray, save_path: str = "images/spatial_clusters.png"):
    """
    Visualize zone clusters on a scatter plot to confirm geographical meaningfulness.
    """
    ensure_dir(save_path)
    plt.figure(figsize=(10, 8))
    
    # Scatter plot of all complaints colored by zone_id
    sns.scatterplot(
        x='longitude', 
        y='latitude', 
        hue='zone_id', 
        palette='tab20', 
        data=df, 
        s=10, 
        alpha=0.6, 
        edgecolor=None
    )
    
    # Plot centroids
    plt.scatter(
        centroids[:, 1], 
        centroids[:, 0], 
        c='red', 
        s=200, 
        marker='X', 
        label='Centroids',
        edgecolors='black'
    )
    
    plt.title('Phase 3: Spatial Clustering of Incidents into Zones')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Zone ID')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved spatial cluster visualization to {save_path}")

def plot_learning_curves(history: dict, save_path: str = "images/learning_curves.png"):
    """
    Plot training/validation loss and F1 curves to confirm convergence.
    """
    ensure_dir(save_path)
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss Curve
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Phase 7: Training & Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (Focal Loss)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # F1 Score Curve
    ax2.plot(epochs, history['train_f1'], 'b-', label='Training F1')
    ax2.plot(epochs, history['val_f1'], 'r-', label='Validation F1')
    ax2.set_title('Phase 7: Training & Validation F1 Score')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('F1 Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved learning curves visualization to {save_path}")

def plot_risk_assessment(risk_results: list, save_path: str = "images/risk_assessment.png"):
    """
    Plot bar chart of risk scores per zone to manually verify risk distribution.
    """
    ensure_dir(save_path)
    
    zones = [str(r['zone_id']) for r in risk_results]
    scores = [r['risk_score'] for r in risk_results]
    levels = [r['risk_level'] for r in risk_results]
    
    # Color code by risk level
    color_map = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
    colors = [color_map.get(lvl, 'blue') for lvl in levels]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(zones, scores, color=colors)
    
    # Add a horizontal line for thresholds if they exist
    plt.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Medium Risk Threshold')
    plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='High Risk Threshold')
    
    plt.title('Phase 8: Dynamic Risk Assessment per Zone')
    plt.xlabel('Zone ID')
    plt.ylabel('Risk Score')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved risk assessment visualization to {save_path}")
