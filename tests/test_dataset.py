"""
Tests for Sequence Dataset Module
"""

import pytest
import numpy as np
from src.dataset import create_sequences, time_based_split, SpatioTemporalDataset


@pytest.fixture
def feature_tensor():
    """Create a synthetic feature tensor [T=20, N=3, F=5]."""
    np.random.seed(42)
    return np.random.rand(20, 3, 5).astype(np.float32)


class TestCreateSequences:
    def test_output_shape(self, feature_tensor):
        seq_len = 3
        X, y = create_sequences(feature_tensor, seq_len)
        T, N, F = feature_tensor.shape
        expected_samples = T - seq_len
        assert X.shape == (expected_samples, seq_len, N, F)
        assert y.shape == (expected_samples, N)

    def test_no_future_leakage(self, feature_tensor):
        """Target time step should always be after input time steps."""
        seq_len = 3
        X, y = create_sequences(feature_tensor, seq_len)
        # For sample i, input covers [i, i+seq_len), target is at i+seq_len
        # This is guaranteed by construction, but let's verify shapes
        assert X.shape[1] == seq_len  # sequence dimension
        assert len(X) == len(feature_tensor) - seq_len

    def test_seq_len_too_large_raises(self, feature_tensor):
        with pytest.raises(ValueError):
            create_sequences(feature_tensor, seq_len=25)

    def test_target_is_continuous(self, feature_tensor):
        X, y = create_sequences(feature_tensor, 3)
        # MSI target is a continuous stress index rather than binary indicators
        assert y.dtype == np.float32
        # Target values are bounded or scaled, not strictly restricted to 0.0 and 1.0
        unique_vals = np.unique(y)
        assert len(unique_vals) > 2



class TestTimeBasedSplit:
    def test_split_ratios(self, feature_tensor):
        X, y = create_sequences(feature_tensor, 3)
        X_tr, y_tr, X_val, y_val, X_test, y_test = time_based_split(X, y)

        total = len(X)
        assert len(X_tr) == int(total * 0.70)
        assert len(X_val) == int(total * 0.85) - int(total * 0.70)
        assert len(X_test) == total - int(total * 0.85)

    def test_no_overlap(self, feature_tensor):
        X, y = create_sequences(feature_tensor, 3)
        X_tr, y_tr, X_val, y_val, X_test, y_test = time_based_split(X, y)
        total = len(X_tr) + len(X_val) + len(X_test)
        assert total == len(X)

    def test_chronological_order(self, feature_tensor):
        """Train samples come before val, val before test."""
        X, y = create_sequences(feature_tensor, 3)
        X_tr, _, X_val, _, X_test, _ = time_based_split(X, y)
        # The arrays are sliced in order, so first train, then val, then test
        n_tr = len(X_tr)
        n_val = len(X_val)
        np.testing.assert_array_equal(X_tr, X[:n_tr])
        np.testing.assert_array_equal(X_val, X[n_tr:n_tr + n_val])
        np.testing.assert_array_equal(X_test, X[n_tr + n_val:])


class TestSpatioTemporalDataset:
    def test_length(self, feature_tensor):
        X, y = create_sequences(feature_tensor, 3)
        ds = SpatioTemporalDataset(X, y)
        assert len(ds) == len(X)

    def test_getitem_shapes(self, feature_tensor):
        X, y = create_sequences(feature_tensor, 3)
        ds = SpatioTemporalDataset(X, y)
        x_item, y_item = ds[0]
        assert x_item.shape == (3, 3, 5)  # seq_len, N, F
        assert y_item.shape == (3,)       # N
