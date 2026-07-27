"""
Graph Attention Network (GAT) for on-chain fraud detection.

Architecture:
    GATConv(in -> hidden, heads=4) -> concatenation -> ELU -> Dropout
    GATConv(hidden*4 -> out, heads=1, concat=False)

Key advantage: attention weights are a free interpretability signal.
Attention weights from conv1 indicate which neighbors each node attends to.
Extract via return_attention=True.

Compliance Disclaimer:
    This system is a research and portfolio demonstration only. It is NOT
    a certified AML/CFT compliance tool, a regulated financial product, or
    a legally defensible fraud-detection system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    """
    Graph Attention Network for on-chain fraud detection.

    Architecture:
        GATConv(in -> hidden, heads=4) -> concatenation -> ELU -> Dropout
        GATConv(hidden*4 -> out, heads=1, concat=False)

    Key advantage: attention weights are a free interpretability signal.
    Attention weights from conv1 indicate which neighbors each node attends to.
    Extract via return_attention_weights=True.
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

        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True,
        )
        self.conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            dropout=dropout,
            concat=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Args:
            x: (N, in_channels)
            edge_index: (2, E)
            return_attention: if True, return (logits, attention_weights)
        Returns:
            logits: (N, out_channels)
            [optional] attention_weights: tuple (edge_index, alpha) from conv1
        """
        x = F.dropout(x, p=self.dropout, training=self.training)

        if return_attention:
            x, (edge_idx, alpha) = self.conv1(
                x, edge_index, return_attention_weights=True
            )
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            return x, (edge_idx, alpha)
        else:
            x = F.elu(self.conv1(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            return x
