"""
Model Architecture Module
=========================
GNN (2-layer GCN with residual) + LSTM + BatchNorm + FC
for spatiotemporal incident prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class GCNLayer(nn.Module):
    """
    Single Graph Convolutional Network layer.
    
    Implements: H' = σ(D^{-1/2} A D^{-1/2} H W)
    Simplified: uses adjacency matrix with self-loops and symmetric normalization.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        """
        Args:
            x: node features [N, F_in]
            adj: adjacency matrix [N, N] (with self-loops and normalized)
        Returns:
            [N, F_out]
        """
        # Message passing: aggregate neighbor features
        support = self.linear(x)          # [N, F_out]
        output = torch.matmul(adj, support)  # [N, F_out]
        return output


class GCNBlock(nn.Module):
    """
    2-layer GCN block with residual connection.
    
    Architecture:
        Input [N, F_in]
        → GCNLayer1 [N, hidden_dim] + ReLU
        → GCNLayer2 [N, hidden_dim] + ReLU
        + Residual projection if F_in != hidden_dim
        → Output [N, hidden_dim]
    """

    def __init__(self, in_features: int, hidden_dim: int = 32):
        super().__init__()
        self.gcn1 = GCNLayer(in_features, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)

        # Residual projection if dimensions don't match
        if in_features != hidden_dim:
            self.residual_proj = nn.Linear(in_features, hidden_dim)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x, adj):
        """
        Args:
            x: [N, F_in]
            adj: [N, N]
        Returns:
            [N, hidden_dim]
        """
        residual = self.residual_proj(x)

        h = F.relu(self.gcn1(x, adj))
        h = F.relu(self.gcn2(h, adj))

        # Residual connection
        h = h + residual

        return h


class CategoryEmbedding(nn.Module):
    """
    Embedding layer for categorical features (category_id).
    """

    def __init__(self, num_categories: int, embed_dim: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(num_categories + 1, embed_dim)  # +1 for unknown

    def forward(self, category_ids):
        """
        Args:
            category_ids: [N] integer tensor
        Returns:
            [N, embed_dim]
        """
        return self.embedding(category_ids)


class SpatioTemporalModel(nn.Module):
    """
    GNN + LSTM model for spatiotemporal prediction.
    
    Architecture:
        For each time step:
            Input features [N, F] → GCN Block → [N, gcn_hidden]
        Stack T time steps → [T, N, gcn_hidden]
        Reshape for LSTM → per zone: [T, gcn_hidden]
        LSTM → [T, lstm_hidden] → take last hidden state
        BatchNorm → Dropout → FC → Sigmoid
    
    Output: P ∈ [0,1] per zone
    """

    def __init__(
        self,
        num_features: int,
        num_zones: int,
        gcn_hidden: int = 32,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        num_categories: int = 0,
        category_embed_dim: int = 8,
        use_sigmoid: bool = True,
    ):
        super().__init__()

        self.num_zones = num_zones
        self.num_features = num_features
        self.gcn_hidden = gcn_hidden
        self.use_category_embed = num_categories > 0
        self.use_sigmoid = use_sigmoid

        # Category embedding (optional)
        if self.use_category_embed:
            self.cat_embed = CategoryEmbedding(num_categories, category_embed_dim)
            gcn_input_dim = num_features + category_embed_dim
        else:
            gcn_input_dim = num_features

        # GCN block (2 layers + residual)
        self.gcn_block = GCNBlock(gcn_input_dim, gcn_hidden)

        # LSTM for temporal learning
        self.lstm = nn.LSTM(
            input_size=gcn_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Output layers — use LayerNorm (works with any batch size, unlike BatchNorm)
        self.layer_norm = nn.LayerNorm(lstm_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, 1)

    def normalize_adjacency(self, adj):
        """
        Add self-loops and apply symmetric normalization.
        D^{-1/2} (A + I) D^{-1/2}
        """
        # Add self-loops
        identity = torch.eye(adj.size(0), device=adj.device)
        adj_hat = adj + identity

        # Degree matrix
        degree = adj_hat.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag(degree_inv_sqrt)

        # Symmetric normalization
        adj_norm = D_inv_sqrt @ adj_hat @ D_inv_sqrt
        return adj_norm

    def forward(self, x, adj, category_ids=None):
        """
        Forward pass.
        
        Args:
            x: input features [batch, seq_len, N, F]
            adj: adjacency matrix [N, N]
            category_ids: (optional) [N] integer tensor
        
        Returns:
            predictions [batch, N] ∈ [0, 1]
        """
        batch_size, seq_len, N, F = x.shape

        # Normalize adjacency matrix
        adj_norm = self.normalize_adjacency(adj)

        # Process each time step through GCN
        gcn_outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]  # [batch, N, F]

            # Process each sample in the batch
            batch_gcn = []
            for b in range(batch_size):
                h = x_t[b]  # [N, F]

                # Concatenate category embeddings if available
                if self.use_category_embed and category_ids is not None:
                    cat_emb = self.cat_embed(category_ids)  # [N, embed_dim]
                    h = torch.cat([h, cat_emb], dim=-1)  # [N, F + embed_dim]

                h = self.gcn_block(h, adj_norm)  # [N, gcn_hidden]
                batch_gcn.append(h)

            batch_gcn = torch.stack(batch_gcn, dim=0)  # [batch, N, gcn_hidden]
            gcn_outputs.append(batch_gcn)

        # Stack time steps: [batch, seq_len, N, gcn_hidden]
        gcn_seq = torch.stack(gcn_outputs, dim=1)

        # Process each zone through LSTM
        predictions = []
        for z in range(N):
            zone_seq = gcn_seq[:, :, z, :]  # [batch, seq_len, gcn_hidden]
            lstm_out, _ = self.lstm(zone_seq)  # [batch, seq_len, lstm_hidden]
            last_hidden = lstm_out[:, -1, :]  # [batch, lstm_hidden]

            # BatchNorm → Dropout → FC
            h = self.layer_norm(last_hidden)
            h = self.dropout(h)
            h = self.fc(h)  # [batch, 1]
            if self.use_sigmoid:
                h = torch.sigmoid(h)
            predictions.append(h.squeeze(-1))  # [batch]

        # Stack zone predictions: [batch, N]
        predictions = torch.stack(predictions, dim=1)

        return predictions


class MultiTaskSpatioTemporalModel(nn.Module):
    """
    Multi-Head SpatioTemporal model sharing GNN+LSTM encoding representations
    to predict Count, Unresolved Ratio, and final MSI (or Delta MSI) simultaneously.
    """

    def __init__(self, base_model: SpatioTemporalModel):
        super().__init__()
        self.base_model = base_model
        # Count forecasting head
        self.fc_count = nn.Linear(base_model.lstm.hidden_size, 1)
        # Unresolved ratio forecasting head
        self.fc_unresolved = nn.Linear(base_model.lstm.hidden_size, 1)

    def forward(self, x, adj, category_ids=None):
        """
        Forward pass.
        
        Args:
            x: input features [batch, seq_len, N, F]
            adj: adjacency matrix [N, N]
            category_ids: (optional) [N] integer tensor
        
        Returns:
            preds_msi: predicted MSI or Delta MSI [batch, N]
            preds_count: predicted future complaint counts [batch, N]
            preds_unresolved: predicted future unresolved ratio [batch, N]
        """
        batch_size, seq_len, N, F = x.shape
        adj_norm = self.base_model.normalize_adjacency(adj)

        # Base GNN sequence encoding
        gcn_outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]
            batch_gcn = []
            for b in range(batch_size):
                h = x_t[b]
                # Concatenate category embeddings if available
                if self.base_model.use_category_embed and category_ids is not None:
                    cat_emb = self.base_model.cat_embed(category_ids)
                    h = torch.cat([h, cat_emb], dim=-1)
                h = self.base_model.gcn_block(h, adj_norm)
                batch_gcn.append(h)
            gcn_outputs.append(torch.stack(batch_gcn, dim=0))
        gcn_seq = torch.stack(gcn_outputs, dim=1)

        preds_msi, preds_count, preds_unresolved = [], [], []
        for z in range(N):
            zone_seq = gcn_seq[:, :, z, :]
            lstm_out, _ = self.base_model.lstm(zone_seq)
            last_hidden = lstm_out[:, -1, :]

            h = self.base_model.layer_norm(last_hidden)
            h = self.base_model.dropout(h)

            msi = self.base_model.fc(h).squeeze(-1)
            count = self.fc_count(h).squeeze(-1)
            unresolved = self.fc_unresolved(h).squeeze(-1)

            preds_msi.append(msi)
            preds_count.append(count)
            preds_unresolved.append(unresolved)

        return torch.stack(preds_msi, dim=1), torch.stack(preds_count, dim=1), torch.stack(preds_unresolved, dim=1)

