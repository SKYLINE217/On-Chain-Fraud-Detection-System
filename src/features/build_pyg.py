# src/features/build_pyg.py
"""
Converts features_combined.parquet → PyG Data object.
Handoff artifact for Person B.

Compliance Disclaimer: This system is a research and portfolio
demonstration only. Not a certified AML/CFT compliance tool.
"""
import pandas as pd
import torch
from torch_geometric.data import Data
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

FEATURE_COLS = [f"f{i}" for i in range(1, 167)] + [
    "tx_freq", "amount_mean", "amount_skew", "address_age",
    "clustering_coeff", "burst_score", "pageRank", "communityId"
]

LABEL_MAP = {"1": 1, "2": 0, "unknown": -1}   # illicit=1, licit=0, unknown=-1


def load_pyg_data(
    parquet_path: str = "data/processed/features_combined.parquet",
    edgelist_path: str = "data/raw/elliptic_txs_edgelist.csv",
) -> Data:
    df = pd.read_parquet(parquet_path)
    edges = pd.read_csv(edgelist_path, header=None, names=["src", "dst"])

    # Build txId → integer index map
    txid_to_idx = {txid: i for i, txid in enumerate(df["txId"])}

    # Node features (float32)
    x = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32)

    # Edge index (map txIds to integer indices)
    src_idx = edges["src"].map(txid_to_idx).dropna().astype(int)
    dst_idx = edges["dst"].map(txid_to_idx).dropna().astype(int)
    edge_index = torch.tensor([src_idx.values, dst_idx.values], dtype=torch.long)

    # Labels
    y = torch.tensor(df["class"].map(LABEL_MAP).values, dtype=torch.long)

    # Temporal masks (non-negotiable split)
    ts = torch.tensor(df["timeStep"].values)
    labeled = y >= 0

    train_mask = (ts <= 34) & labeled
    val_mask   = (ts >= 35) & (ts <= 39) & labeled
    test_mask  = (ts >= 40) & labeled

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    # Critical invariant assertions (from ml_models.md)
    assert (data.y[data.train_mask] == -1).sum() == 0, \
        "CRITICAL: Unknown nodes leaked into train_mask"
    assert data.edge_index.shape == (2, 234355), \
        f"Edge shape: {data.edge_index.shape}"
    assert data.x.shape[0] == 203769, \
        f"Node count: {data.x.shape[0]}"

    logger.info(
        f"PyG Data built: nodes={data.x.shape[0]}, edges={data.edge_index.shape[1]}, "
        f"features={data.x.shape[1]}, "
        f"train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}"
    )

    return data


if __name__ == "__main__":
    data = load_pyg_data()
    torch.save(data, "data/processed/pyg_data.pt")
    print("Saved data/processed/pyg_data.pt")
