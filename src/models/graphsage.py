"""
GraphSAGE node classifier for on-chain fraud detection.

Architecture:
    SAGEConv(in -> hidden) + BN + ReLU + Dropout
    [SAGEConv(hidden -> hidden) + BN + ReLU + Dropout] x (num_layers - 2)
    SAGEConv(hidden -> out)

Production model: mean aggregation (inductive, generalizes to unseen nodes).

Why GraphSAGE over GCN: GraphSAGE is inductive -- it learns aggregation
functions that generalize to unseen nodes. This is critical because new
wallet addresses appear constantly. GCN is transductive (requires all nodes
at training time).

Compliance Disclaimer:
    This system is a research and portfolio demonstration only. It is NOT
    a certified AML/CFT compliance tool, a regulated financial product, or
    a legally defensible fraud-detection system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    """
    GraphSAGE node classifier for on-chain fraud detection.

    Architecture:
        SAGEConv(in -> hidden) + BN + ReLU + Dropout
        [SAGEConv(hidden -> hidden) + BN + ReLU + Dropout] x (num_layers - 2)
        SAGEConv(hidden -> out)

    Production model: mean aggregation (inductive, generalizes to unseen nodes).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 2,
        num_layers: int = 3,
        dropout: float = 0.3,
        aggr: str = "mean",
    ):
        super().__init__()
        assert num_layers >= 2, "Need at least 2 layers"

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        self.num_layers = num_layers

        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Output layer (no BN, no activation -- logits for CrossEntropyLoss)
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Node feature matrix (N, in_channels) float32
            edge_index: Edge index (2, E) long
        Returns:
            logits: (N, out_channels) float32 -- NOT softmaxed
        """
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x

    def get_embedding(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Return final hidden layer output (before last SAGEConv) for embedding storage."""
        for i in range(self.num_layers - 2):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # Apply second-to-last conv (gets to hidden_channels space)
        x = self.convs[-2](x, edge_index)
        x = self.bns[-1](x)
        x = F.relu(x)
        return x  # shape: (N, hidden_channels)
