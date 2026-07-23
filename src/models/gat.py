# src/models/gat.py
# ──────────────────────────────────────────────────────────────────────────
# GAT (Graph Attention Network) model for node classification.
#
# Architecture (from aim.md §6):
#   - 2-layer GATConv
#   - Multi-head attention: 4 heads on first layer, 1 on output
#   - Attention weights are a free interpretability signal — extract
#     them during inference for comparison with GNNExplainer
#
# This is the SECONDARY GNN model. GraphSAGE is the production model
# unless GAT clearly outperforms on PR-AUC.
# ──────────────────────────────────────────────────────────────────────────

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    """
    Graph Attention Network for node classification.

    Parameters
    ----------
    in_channels : int
        Number of input features per node.
    hidden_channels : int
        Hidden dimension per attention head.
    out_channels : int
        Number of output classes (2: licit/illicit).
    heads : int
        Number of attention heads in the first layer (default: 4).
    dropout : float
        Dropout probability (default: 0.3).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.dropout = dropout
        self.heads = heads
        self.hidden_channels = hidden_channels

        # First layer: multi-head attention
        # Output shape: (N, hidden_channels * heads) because concat=True (default)
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
        )
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels * heads)

        # Output layer: single head, no concat
        # Output shape: (N, out_channels)
        self.conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )

    def forward(self, x, edge_index, return_attention_weights=False):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix, shape (N, in_channels).
        edge_index : torch.Tensor
            Edge index, shape (2, E).
        return_attention_weights : bool
            If True, also return attention weights from both layers
            (useful for interpretability — free signal per aim.md §6).

        Returns
        -------
        out : torch.Tensor
            Raw logits, shape (N, out_channels).
        attn_weights : tuple of (edge_index, alpha), optional
            Only returned if return_attention_weights=True.
            Contains attention coefficients from both layers.
        """
        # Layer 1
        x = F.dropout(x, p=self.dropout, training=self.training)

        if return_attention_weights:
            x, attn1 = self.conv1(
                x, edge_index, return_attention_weights=True
            )
        else:
            x = self.conv1(x, edge_index)

        x = self.bn1(x)
        x = F.elu(x)

        # Layer 2
        x = F.dropout(x, p=self.dropout, training=self.training)

        if return_attention_weights:
            x, attn2 = self.conv2(
                x, edge_index, return_attention_weights=True
            )
            return x, (attn1, attn2)
        else:
            x = self.conv2(x, edge_index)
            return x

    def get_embeddings(self, x, edge_index):
        """
        Extract embeddings after the first GAT layer (before output head).
        Shape: (N, hidden_channels * heads).
        """
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        return x

    def get_config(self):
        """Return a serializable config dict for model_config.json."""
        return {
            "model_type": "GAT",
            "in_channels": self.conv1.in_channels,
            "hidden_channels": self.hidden_channels,
            "out_channels": self.conv2.out_channels,
            "heads": self.heads,
            "dropout": self.dropout,
        }
