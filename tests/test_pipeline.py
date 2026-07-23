# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.

import torch
from src.models.graphsage import GraphSAGE
from src.models.gat import GAT

def test_graphsage_forward():
    model = GraphSAGE(in_channels=10, hidden_channels=16, out_channels=2)
    x = torch.randn(5, 10)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    out = model(x, edge_index)
    assert out.shape == (5, 2)

def test_gat_forward():
    model = GAT(in_channels=10, hidden_channels=16, out_channels=2, heads=2)
    x = torch.randn(5, 10)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    out = model(x, edge_index)
    assert out.shape == (5, 2)
