"""
Tests for Spatial Clustering & Graph Module
"""

import pytest
import numpy as np
from src.clustering import (
    find_optimal_clusters,
    create_zones,
    build_adjacency_matrix,
    build_edge_index,
)
import pandas as pd


@pytest.fixture
def sample_coords():
    """Generate clustered coordinates for testing."""
    np.random.seed(42)
    # 3 clear clusters
    c1 = np.random.randn(50, 2) * 0.1 + [12.97, 77.59]
    c2 = np.random.randn(50, 2) * 0.1 + [13.05, 77.65]
    c3 = np.random.randn(50, 2) * 0.1 + [12.90, 77.50]
    return np.vstack([c1, c2, c3])


@pytest.fixture
def sample_df(sample_coords):
    """Create DataFrame from sample coordinates."""
    return pd.DataFrame({
        "latitude": sample_coords[:, 0],
        "longitude": sample_coords[:, 1],
        "category_id": np.random.randint(1, 5, len(sample_coords)),
    })


@pytest.fixture
def centroids():
    """Simple centroids for graph tests."""
    return np.array([
        [12.97, 77.59],
        [13.05, 77.65],
        [12.90, 77.50],
        [13.00, 77.55],
        [12.95, 77.62],
    ], dtype=np.float32)


class TestCreateZones:
    def test_zone_count_matches(self, sample_df):
        n_clusters = 5
        df_result, centroids = create_zones(sample_df, n_clusters)
        assert df_result["zone_id"].nunique() <= n_clusters
        assert len(centroids) == n_clusters

    def test_zone_id_column_added(self, sample_df):
        df_result, _ = create_zones(sample_df, 3)
        assert "zone_id" in df_result.columns

    def test_centroid_shape(self, sample_df):
        n = 4
        _, centroids = create_zones(sample_df, n)
        assert centroids.shape == (n, 2)


class TestAdjacencyMatrix:
    def test_shape_is_square(self, centroids):
        adj = build_adjacency_matrix(centroids, k=3)
        n = len(centroids)
        assert adj.shape == (n, n)

    def test_symmetric(self, centroids):
        adj = build_adjacency_matrix(centroids, k=3)
        np.testing.assert_array_almost_equal(adj, adj.T)

    def test_all_weights_positive(self, centroids):
        adj = build_adjacency_matrix(centroids, k=3)
        assert (adj[adj != 0] > 0).all()

    def test_no_self_loops(self, centroids):
        adj = build_adjacency_matrix(centroids, k=3)
        np.testing.assert_array_equal(np.diag(adj), np.zeros(len(centroids)))

    def test_max_neighbors(self, centroids):
        k = 2
        adj = build_adjacency_matrix(centroids, k=k)
        # Due to symmetry, a node may have more than k connections
        # but each row should have at least k non-zero entries
        for i in range(len(centroids)):
            assert np.count_nonzero(adj[i]) >= k


class TestEdgeIndex:
    def test_edge_index_shape(self, centroids):
        adj = build_adjacency_matrix(centroids, k=2)
        edge_index, edge_weight = build_edge_index(adj)
        assert edge_index.shape[0] == 2
        assert len(edge_weight) == edge_index.shape[1]

    def test_edge_weights_positive(self, centroids):
        adj = build_adjacency_matrix(centroids, k=2)
        _, edge_weight = build_edge_index(adj)
        assert (edge_weight > 0).all()
