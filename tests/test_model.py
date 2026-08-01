"""
Tests for Model Architecture
"""

import pytest
import torch
import numpy as np
from src.model import SpatioTemporalModel, GCNBlock, GCNLayer


@pytest.fixture
def model_config():
    return {
        "num_features": 5,
        "num_zones": 3,
        "gcn_hidden": 16,
        "lstm_hidden": 32,
        "lstm_layers": 1,
        "dropout": 0.1,
    }


@pytest.fixture
def model(model_config):
    return SpatioTemporalModel(**model_config)


@pytest.fixture
def adj_matrix():
    """Simple 3-zone adjacency matrix."""
    adj = torch.FloatTensor([
        [0.0, 1.0, 0.5],
        [1.0, 0.0, 0.8],
        [0.5, 0.8, 0.0],
    ])
    return adj


@pytest.fixture
def sample_input():
    """Batch of 2 samples, seq_len=3, N=3 zones, F=5 features."""
    return torch.randn(2, 3, 3, 5)


class TestGCNLayer:
    def test_output_shape(self):
        layer = GCNLayer(5, 16)
        x = torch.randn(3, 5)
        adj = torch.eye(3)
        out = layer(x, adj)
        assert out.shape == (3, 16)


class TestGCNBlock:
    def test_output_shape(self):
        block = GCNBlock(5, 16)
        x = torch.randn(3, 5)
        adj = torch.eye(3)
        out = block(x, adj)
        assert out.shape == (3, 16)

    def test_residual_connection(self):
        """Output should differ from just GCN layers (residual adds input)."""
        block = GCNBlock(16, 16)  # same dim → identity residual
        x = torch.randn(3, 16)
        adj = torch.eye(3)
        out = block(x, adj)
        # If residual works, output should not be zero even with identity adj
        assert out.abs().sum() > 0


class TestSpatioTemporalModel:
    def test_output_shape(self, model, sample_input, adj_matrix):
        output = model(sample_input, adj_matrix)
        assert output.shape == (2, 3)  # [batch, N]

    def test_output_range(self, model, sample_input, adj_matrix):
        """All outputs should be ∈ [0, 1] due to sigmoid."""
        output = model(sample_input, adj_matrix)
        assert (output >= 0).all()
        assert (output <= 1).all()

    def test_gradient_flow(self, model, sample_input, adj_matrix):
        """Ensure gradients flow through all parameters."""
        output = model(sample_input, adj_matrix)
        loss = output.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                # Check gradient is not all zeros (dead layer)
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_different_batch_sizes(self, model, adj_matrix):
        """Model should handle different batch sizes."""
        for bs in [1, 4, 8]:
            x = torch.randn(bs, 3, 3, 5)
            output = model(x, adj_matrix)
            assert output.shape == (bs, 3)
