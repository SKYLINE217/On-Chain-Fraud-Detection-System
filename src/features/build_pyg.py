"""
build_pyg.py -- Build PyG Data object from features_combined.parquet.

Converts the parquet + edgelist into a torch_geometric.data.Data object with:
    - x: (N, 174) feature matrix
    - edge_index: (2, E) edge tensor
    - y: (N,) labels -- 0=licit, 1=illicit, -1=unknown
    - train_mask, val_mask, test_mask: temporal split masks (labeled only)

Temporal split (non-negotiable):
    Train: time steps 1-34
    Val:   time steps 35-39
    Test:  time steps 40-49

Compliance Disclaimer:
    This system is a research and portfolio demonstration only. It is NOT
    a certified AML/CFT compliance tool, a regulated financial product, or
    a legally defensible fraud-detection system.
"""

import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import logging

logger = logging.getLogger(__name__)

FEATURE_COLS = [f"f{i}" for i in range(1, 167)] + [
    "tx_freq", "amount_mean", "amount_skew", "address_age",
    "clustering_coeff", "burst_score", "pageRank", "communityId",
]

def load_pyg_data(
    parquet_path: str = "data/processed/features_combined.parquet",
    edgelist_path: str = "data/raw/elliptic_txs_edgelist.csv",
    mock_edges: bool = False,
) -> Data:
    """
    Build PyG Data object from parquet and edgelist.

    Args:
        parquet_path: Path to features_combined.parquet
        edgelist_path: Path to elliptic_txs_edgelist.csv
        mock_edges: If True, generate random edges (for mock data testing)

    Returns:
        PyG Data object with x, edge_index, y, train_mask, val_mask, test_mask
    """
    logger.info(f"Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    tx_ids = df["txId"].values
    id_to_idx = {str(tx_id): idx for idx, tx_id in enumerate(tx_ids)}
    n_nodes = len(df)

    x = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32)
    logger.info(f"Feature matrix: {x.shape}")

    y = torch.full((n_nodes,), -1, dtype=torch.long)
    y[df["class"] == "1"] = 1    
    y[df["class"] == "2"] = 0    

    logger.info(
        f"Labels: illicit={int((y == 1).sum())}, "
        f"licit={int((y == 0).sum())}, unknown={int((y == -1).sum())}"
    )

    if mock_edges or not os.path.exists(edgelist_path):
        if not mock_edges:
            logger.warning(
                f"Edgelist not found at {edgelist_path} -- generating mock edges"
            )

        rng = np.random.default_rng(42)
        n_edges = 234355  
        src = rng.integers(0, n_nodes, size=n_edges)
        dst = rng.integers(0, n_nodes, size=n_edges)
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    else:
        logger.info(f"Loading edgelist: {edgelist_path}")
        edges_df = pd.read_csv(edgelist_path)

        src_col = edges_df.columns[0]
        dst_col = edges_df.columns[1]

        valid_edges = []
        for _, row in edges_df.iterrows():
            src_id = str(int(row[src_col]))
            dst_id = str(int(row[dst_col]))
            if src_id in id_to_idx and dst_id in id_to_idx:
                valid_edges.append((id_to_idx[src_id], id_to_idx[dst_id]))

        if len(valid_edges) == 0:
            raise ValueError("No valid edges found -- check txId mapping")

        edge_array = np.array(valid_edges, dtype=np.int64).T
        edge_index = torch.tensor(edge_array, dtype=torch.long)

    logger.info(f"Edge index: {edge_index.shape}")

    time_steps = df["timeStep"].values
    labeled = (y >= 0)

    train_mask = labeled & torch.tensor(time_steps <= 34)
    val_mask = labeled & torch.tensor((time_steps >= 35) & (time_steps <= 39))
    test_mask = labeled & torch.tensor(time_steps >= 40)

    logger.info(
        f"Masks: train={int(train_mask.sum())}, "
        f"val={int(val_mask.sum())}, test={int(test_mask.sum())}"
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        time_step=torch.tensor(time_steps, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    assert (data.y[data.train_mask] == -1).sum() == 0,        "CRITICAL BUG: Unknown nodes in train_mask"
    assert data.train_mask.sum() + data.val_mask.sum() + data.test_mask.sum() ==        (data.y >= 0).sum(),        "Mask counts don't match labeled node count"
    assert data.x.shape[0] == 203769, f"Node count mismatch: {data.x.shape[0]}"

    logger.info("All invariant checks passed")
    return data
