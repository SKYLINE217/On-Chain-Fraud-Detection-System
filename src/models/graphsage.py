# src/models/graphsage.py
# ──────────────────────────────────────────────────────────────────────────
# GraphSAGE model for node classification on the Elliptic transaction graph.
#
# Architecture (from aim.md §6):
#   - 2–3 SAGEConv layers (configurable via num_layers)
#   - Hidden dim: 128 (sweep range: 64–256)
#   - Dropout: 0.3 (sweep range: 0.2–0.5)
#   - Aggregation: mean (also try max via aggr parameter)
#   - BatchNorm between layers
#
# This is the PRIMARY production/serving model because GraphSAGE is
# inductive — it generalizes to unseen nodes at inference, which is
# critical since new wallets appear constantly.
# ──────────────────────────────────────────────────────────────────────────

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):
    """
    GraphSAGE model for node classification.

    Parameters
    ----------
    in_channels : int
        Number of input features per node (166 raw, or 171+ with engineered).
    hidden_channels : int
        Hidden layer dimension (default: 128, sweep: 64–256).
    out_channels : int
        Number of output classes (2: licit/illicit).
    num_layers : int
        Number of SAGEConv layers (default: 3, sweep: 2–3).
    dropout : float
        Dropout probability (default: 0.3, sweep: 0.2–0.5).
    aggr : str
        Aggregation function for neighbor messages ('mean' or 'max').
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

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        # Output layer (no BatchNorm, no activation — raw logits)
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))

    def forward(self, x, edge_index):
        """
        Forward pass — all nodes participate in message passing
        (including unknown-class nodes for structural signal).

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix, shape (N, in_channels).
        edge_index : torch.Tensor
            Edge index, shape (2, E).

        Returns
        -------
        torch.Tensor
            Raw logits, shape (N, out_channels).
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final layer — raw logits (no activation, no dropout)
        x = self.convs[-1](x, edge_index)
        return x

    def get_embeddings(self, x, edge_index):
        """
        Extract the final hidden layer embeddings (before the output layer).
        Used for UMAP visualization and as the `embedding` property in Neo4j.

        Returns
        -------
        torch.Tensor
            Node embeddings, shape (N, hidden_channels).
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            # No dropout during embedding extraction (eval mode)
        return x

    def get_config(self):
        """Return a serializable config dict for model_config.json (blend.md Contract 3)."""
        return {
            "model_type": "GraphSAGE",
            "in_channels": self.convs[0].in_channels,
            "hidden_channels": self.bns[0].num_features,
            "out_channels": self.convs[-1].out_channels,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }
