# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

"""
PyTorch Geometric graph builder — constructs Data objects with temporal splits.

Label encoding:
    class="1" → 1 (illicit)
    class="2" → 0 (licit)
    "unknown" → -1 (excluded from supervised loss)

Temporal split (NEVER random — per project constraints):
    Train:  time steps 1–34
    Val:    time steps 35–39
    Test:   time steps 40–49

Unknown nodes remain in the graph for structural signal,
but are masked out of the supervised loss.
"""

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data


def create_pyg_graph(df: pd.DataFrame, df_edges: pd.DataFrame) -> Data:
    """
    Constructs a PyTorch Geometric Data object from the processed feature DataFrame.

    Args:
        df: Combined features DataFrame (from engineer.build_features).
            Must contain: txId, timeStep, label, and feature columns.
        df_edges: Edge list DataFrame with columns txId1, txId2.

    Returns:
        PyG Data object with x, edge_index, y, train_mask, val_mask, test_mask.
    """
    # Map txId to contiguous node indices
    nodes = df["txId"].values
    node_map = {tx_id: idx for idx, tx_id in enumerate(nodes)}

    # Feature matrix X — exclude metadata columns
    exclude_cols = {"txId", "timeStep", "class", "label", "in_degree", "out_degree"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    x = torch.tensor(df[feature_cols].values, dtype=torch.float)

    # Target labels Y
    y = torch.tensor(df["label"].values, dtype=torch.long)

    # Edge index — only keep edges where both endpoints exist in the node map
    valid_edges = df_edges[
        df_edges["txId1"].isin(node_map) & df_edges["txId2"].isin(node_map)
    ]
    src = valid_edges["txId1"].map(node_map).values.astype(np.int64)
    dst = valid_edges["txId2"].map(node_map).values.astype(np.int64)
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)

    # Time steps for temporal splitting
    time_steps = df["timeStep"].values

    # ── Temporal split masks ──
    # Train: steps 1–34, Val: steps 35–39, Test: steps 40–49
    # Only include LABELED nodes (label != -1) in supervised masks
    labeled = (y != -1)

    train_mask = torch.tensor(
        (time_steps >= 1) & (time_steps <= 34) & labeled.numpy(),
        dtype=torch.bool,
    )
    val_mask = torch.tensor(
        (time_steps >= 35) & (time_steps <= 39) & labeled.numpy(),
        dtype=torch.bool,
    )
    test_mask = torch.tensor(
        (time_steps >= 40) & (time_steps <= 49) & labeled.numpy(),
        dtype=torch.bool,
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    # Store metadata for downstream use
    data.num_features_original = 166
    data.num_features_engineered = len(feature_cols) - 166
    data.num_classes = 2  # binary: illicit vs licit

    _print_summary(data, y, train_mask, val_mask, test_mask)
    return data


def _print_summary(data, y, train_mask, val_mask, test_mask):
    """Print a summary of the constructed graph."""
    print("── PyG Graph Summary ──")
    print(f"    Nodes:         {data.num_nodes}")
    print(f"    Edges:         {data.num_edges}")
    print(f"    Features/node: {data.num_node_features}")
    print(f"    Train nodes:   {train_mask.sum().item()} (labeled, steps 1–34)")
    print(f"    Val nodes:     {val_mask.sum().item()} (labeled, steps 35–39)")
    print(f"    Test nodes:    {test_mask.sum().item()} (labeled, steps 40–49)")
    print(f"    Illicit (y=1): {(y == 1).sum().item()}")
    print(f"    Licit (y=0):   {(y == 0).sum().item()}")
    print(f"    Unknown (y=-1):{(y == -1).sum().item()}")


def save_pyg_graph(data: Data, output_path: str):
    """Save PyG Data object to disk."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(data, output_path)
    print(f"[✓] PyG graph saved to {output_path}")


def load_pyg_graph(path: str) -> Data:
    """Load PyG Data object from disk."""
    data = torch.load(path)
    print(f"[✓] PyG graph loaded from {path}")
    return data
