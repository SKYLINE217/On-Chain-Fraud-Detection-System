# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

"""
src/models/gnn.py

BUG-06 Fix: This file previously contained full duplicate reimplementations
of GraphSAGE and GAT that were INFERIOR to graphsage.py and gat.py
(missing BatchNorm, aggr param, get_config(), get_embeddings(), etc.).

Those duplicate classes caused silent model quality regressions — any
import from src.models.gnn would get the broken stripped-down versions.

This file now re-exports the canonical implementations to prevent confusion.
"""

# Re-export canonical implementations — DO NOT redefine these classes here
from src.models.graphsage import GraphSAGE
from src.models.gat import GAT

__all__ = ["GraphSAGE", "GAT"]
