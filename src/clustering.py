"""
Spatial Clustering & Graph Construction Module
===============================================
Create zones via KMeans, build k-NN graph with distance-weighted edges.
"""

import numpy as np
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


def find_optimal_clusters(
    coords: np.ndarray,
    k_range: tuple = (5, 20),
) -> int:
    """
    Determine optimal number of clusters using Elbow + Silhouette method.
    
    Args:
        coords: array of shape [N, 2] (latitude, longitude)
        k_range: (min_k, max_k) inclusive range to search
    
    Returns:
        Optimal number of clusters
    """
    k_min, k_max = k_range
    k_values = range(k_min, k_max + 1)

    inertias = []
    silhouettes = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)
        inertias.append(kmeans.inertia_)

        if k >= 2:
            sil = silhouette_score(coords, labels, sample_size=min(5000, len(coords)))
            silhouettes.append(sil)
        else:
            silhouettes.append(-1)

    # Pick k with highest silhouette score
    best_idx = int(np.argmax(silhouettes))
    best_k = list(k_values)[best_idx]

    logger.info(f"Optimal clusters: {best_k} "
                f"(silhouette={silhouettes[best_idx]:.4f})")
    return best_k


def create_zones(
    df,
    n_clusters: int,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
):
    """
    Cluster complaints into spatial zones using KMeans.
    
    Args:
        df: DataFrame with lat/lon columns
        n_clusters: number of zones
    
    Returns:
        (df_with_zone_id, centroids)
        - df_with_zone_id: original df + 'zone_id' column
        - centroids: array of shape [n_clusters, 2]
    """
    coords = df[[lat_col, lon_col]].values

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["zone_id"] = kmeans.fit_predict(coords)
    centroids = kmeans.cluster_centers_

    logger.info(f"Created {n_clusters} zones from {len(df)} complaints")
    for z in range(n_clusters):
        count = (df["zone_id"] == z).sum()
        logger.debug(f"  Zone {z}: {count} complaints, "
                     f"centroid=({centroids[z][0]:.4f}, {centroids[z][1]:.4f})")

    return df, centroids


def build_adjacency_matrix(
    centroids: np.ndarray,
    k: int = 3,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Build a weighted adjacency matrix using k-nearest neighbors.
    
    Edge weight = 1 / (distance + ε)
    
    Args:
        centroids: array of shape [N, 2] (zone centroids)
        k: number of nearest neighbors per node
        epsilon: small constant to avoid division by zero
    
    Returns:
        Weighted adjacency matrix of shape [N, N]
    """
    n = len(centroids)
    k_actual = min(k, n - 1)  # can't have more neighbors than nodes - 1

    nn = NearestNeighbors(n_neighbors=k_actual + 1, metric="euclidean")
    nn.fit(centroids)
    distances, indices = nn.kneighbors(centroids)

    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j_idx in range(1, k_actual + 1):  # skip self (index 0)
            j = indices[i, j_idx]
            dist = distances[i, j_idx]
            weight = 1.0 / (dist + epsilon)
            # Make symmetric
            adj[i, j] = weight
            adj[j, i] = weight

    logger.info(f"Built adjacency matrix: {n}×{n}, "
                f"{int(np.count_nonzero(adj))} non-zero edges")
    return adj


def assign_zone(coords: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign each coordinate to the nearest zone centroid.

    Args:
        coords: array of shape [N, 2] (latitude, longitude)
        centroids: array of shape [K, 2] (zone centroids)

    Returns:
        Array of shape [N] with zone IDs
    """
    diffs = coords[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # [N, K, 2]
    dists = np.linalg.norm(diffs, axis=2)                           # [N, K]
    return dists.argmin(axis=1)                                      # [N]


def build_edge_index(adj_matrix: np.ndarray):
    """
    Convert adjacency matrix to PyTorch Geometric edge_index + edge_weight.
    
    Returns:
        (edge_index, edge_weight) as numpy arrays
        - edge_index: [2, num_edges]
        - edge_weight: [num_edges]
    """
    rows, cols = np.where(adj_matrix > 0)
    edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
    edge_weight = adj_matrix[rows, cols].astype(np.float32)

    logger.info(f"Edge index: {edge_index.shape[1]} directed edges")
    return edge_index, edge_weight
