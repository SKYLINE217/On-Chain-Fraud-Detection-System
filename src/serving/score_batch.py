# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

import torch
from torch_geometric.data import Data

def score_batch(model, data: Data) -> torch.Tensor:
    """
    Computes risk score probabilities for input graph nodes.
    Returns probability of class 1 (Illicit/Fraud).
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.exp(out)[:, 1] # Probability of illicit class
    return probs
