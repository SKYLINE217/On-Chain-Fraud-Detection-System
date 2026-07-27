"""
train_pipeline.py — Standalone end-to-end training pipeline.

Downloads Elliptic dataset via kagglehub, engineers features locally
(without Neo4j GDS), trains baseline + GNN models, saves checkpoints.

Usage:
    python scripts/train_pipeline.py

Compliance Disclaimer: This system is a research and portfolio
demonstration only. Not a certified AML/CFT compliance tool.
"""

import os
import sys
import json
import logging
import shutil
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import skew

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
CHECKPOINT_DIR = "checkpoints"
FEATURE_COLS_BASE = [f"f{i}" for i in range(1, 167)]
OUTPUT_PARQUET = os.path.join(PROCESSED_DIR, "features_combined.parquet")


# =========================================================================
# STEP 1: Download dataset
# =========================================================================
def download_dataset():
    """Download Elliptic dataset using kagglehub."""
    logger.info("=" * 60)
    logger.info("STEP 1: Downloading Elliptic dataset via kagglehub")
    logger.info("=" * 60)

    import kagglehub
    path = kagglehub.dataset_download("ellipticco/elliptic-data-set")
    logger.info(f"Dataset downloaded to: {path}")

    # Copy files to data/raw/
    os.makedirs(RAW_DIR, exist_ok=True)

    # Find the CSV files (may be in subdirectories)
    csv_files = {
        "elliptic_txs_features.csv": None,
        "elliptic_txs_classes.csv": None,
        "elliptic_txs_edgelist.csv": None,
    }

    for root, dirs, files in os.walk(path):
        for f in files:
            if f in csv_files:
                csv_files[f] = os.path.join(root, f)

    for fname, src_path in csv_files.items():
        if src_path is None:
            raise FileNotFoundError(f"Could not find {fname} in downloaded dataset")
        dst_path = os.path.join(RAW_DIR, fname)
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
            logger.info(f"  Copied {fname}")
        else:
            logger.info(f"  {fname} already exists, skipping")

    return path


# =========================================================================
# STEP 2: Engineer features (locally, no Neo4j)
# =========================================================================
def engineer_features():
    """Build features_combined.parquet without Neo4j GDS."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: Engineering features (local, no Neo4j)")
    logger.info("=" * 60)

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Load base data
    logger.info("Loading CSVs...")
    features = pd.read_csv(
        os.path.join(RAW_DIR, "elliptic_txs_features.csv"),
        header=None,
        names=["txId", "timeStep"] + FEATURE_COLS_BASE,
    )
    features["txId"] = features["txId"].astype(int)

    classes = pd.read_csv(os.path.join(RAW_DIR, "elliptic_txs_classes.csv"))
    edges = pd.read_csv(os.path.join(RAW_DIR, "elliptic_txs_edgelist.csv"))
    edges.columns = ["src", "dst"]
    edges["src"] = edges["src"].astype(int)
    edges["dst"] = edges["dst"].astype(int)

    df = features.merge(classes, on="txId", how="left")
    df["class"] = df["class"].fillna("unknown").astype(str)

    logger.info(f"Nodes: {len(df)}, Edges: {len(edges)}")

    # --- Engineered features ---
    logger.info("Computing tx_freq...")
    out_deg = edges.groupby("src").size().rename("out_deg")
    in_deg = edges.groupby("dst").size().rename("in_deg")
    deg = pd.DataFrame({"txId": df["txId"]})
    deg = deg.join(out_deg, on="txId").join(in_deg, on="txId").fillna(0)
    df["tx_freq"] = (deg["out_deg"] + deg["in_deg"]).values

    logger.info("Computing amount_mean, amount_skew...")
    df["amount_mean"] = df.groupby("timeStep")["f93"].transform("mean")
    df["amount_skew"] = df.groupby("timeStep")["f93"].transform(
        lambda x: skew(x.fillna(0))
    )

    logger.info("Computing address_age...")
    df["address_age"] = df.groupby("txId")["timeStep"].transform("min")

    logger.info("Computing burst_score...")
    out_deg_df = edges.groupby("src").size().rename("out_deg").reset_index()
    out_deg_df.columns = ["txId", "out_deg"]
    df_with_deg = df[["txId", "timeStep"]].merge(out_deg_df, on="txId", how="left").fillna(0)
    step_mean = df_with_deg.groupby("timeStep")["out_deg"].transform("mean")
    step_std = df_with_deg.groupby("timeStep")["out_deg"].transform("std").replace(0, 1)
    df["burst_score"] = ((df_with_deg["out_deg"] - step_mean) / step_std).values

    # --- Graph features via networkx ---
    logger.info("Computing graph features (PageRank, Louvain, clustering)...")
    import networkx as nx

    G = nx.DiGraph()
    G.add_nodes_from(df["txId"].values)
    G.add_edges_from(zip(edges["src"].values, edges["dst"].values))
    logger.info(f"NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # PageRank
    logger.info("  Computing PageRank...")
    pr = nx.pagerank(G, alpha=0.85, max_iter=20, tol=1e-4)
    df["pageRank"] = df["txId"].map(pr).fillna(0).values

    # Louvain community detection (requires undirected graph)
    logger.info("  Computing Louvain communities...")
    try:
        import community as community_louvain
        G_undir = G.to_undirected()
        partition = community_louvain.best_partition(G_undir, random_state=42)
        df["communityId"] = df["txId"].map(partition).fillna(-1).astype(int).values
    except ImportError:
        logger.warning("python-louvain not installed, using label propagation instead")
        G_undir = G.to_undirected()
        communities = nx.community.label_propagation_communities(G_undir)
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i
        df["communityId"] = df["txId"].map(partition).fillna(-1).astype(int).values

    # Clustering coefficient (undirected)
    logger.info("  Computing clustering coefficients...")
    G_undir = G.to_undirected()
    cc = nx.clustering(G_undir)
    df["clustering_coeff"] = df["txId"].map(cc).fillna(0).values

    # Fill NaN amounts
    for col in ["amount_mean", "amount_skew"]:
        df[col] = df.groupby("timeStep")[col].transform(
            lambda x: x.fillna(x.median())
        )

    # Final column order (must match model_config.json)
    final_cols = (
        ["txId", "timeStep", "class"]
        + FEATURE_COLS_BASE
        + ["tx_freq", "amount_mean", "amount_skew", "address_age",
           "clustering_coeff", "burst_score", "pageRank", "communityId"]
    )
    df = df[final_cols]

    # Validate
    assert df.shape == (203769, 177), f"Shape mismatch: {df.shape}"
    nan_counts = df.drop(columns=["class"]).select_dtypes("number").isna().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"NaN found, filling with 0:\n{nan_counts[nan_counts > 0]}")
        df = df.fillna(0)

    df.to_parquet(OUTPUT_PARQUET, index=False)
    logger.info(f"Saved: {OUTPUT_PARQUET}  shape={df.shape}")
    return df


# =========================================================================
# STEP 3: Train baseline models
# =========================================================================
def train_baselines():
    """Train LR, RF, XGBoost baselines."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: Training baseline models (LR, RF, XGBoost)")
    logger.info("=" * 60)

    from src.models.baselines import run_all_baselines
    import joblib

    results, xgb_model, pr_curve_data = run_all_baselines(
        parquet_path=OUTPUT_PARQUET,
        use_wandb=False,
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    xgb_path = os.path.join(CHECKPOINT_DIR, "xgb_baseline.pkl")
    joblib.dump(xgb_model, xgb_path)
    logger.info(f"XGBoost checkpoint saved: {xgb_path}")

    # Print results table
    df_results = pd.DataFrame(results)
    print("\n" + df_results.to_string(index=False))

    return results


# =========================================================================
# STEP 4: Train GNN models
# =========================================================================
def train_gnns():
    """Train GraphSAGE and GAT models."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: Training GNN models (GraphSAGE + GAT)")
    logger.info("=" * 60)

    import torch
    from src.features.build_pyg import load_pyg_data
    from src.models.graphsage import GraphSAGE
    from src.models.gat import GAT
    from src.models.train import train

    data = load_pyg_data(
        parquet_path=OUTPUT_PARQUET,
        edgelist_path=os.path.join(RAW_DIR, "elliptic_txs_edgelist.csv"),
    )
    logger.info(f"PyG Data: {data}")

    config = {
        "lr": 0.001,
        "weight_decay": 5e-4,
        "epochs": 100,
        "patience": 20,
    }

    # --- GraphSAGE ---
    logger.info("")
    logger.info("-" * 40)
    logger.info("Training GraphSAGE (3L, h=128, mean)")
    logger.info("-" * 40)

    sage_model = GraphSAGE(
        in_channels=data.x.shape[1],
        hidden_channels=128,
        out_channels=2,
        num_layers=3,
        dropout=0.3,
        aggr="mean",
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    sage_metrics = train(
        model=sage_model,
        data=data,
        config=config,
        checkpoint_path=os.path.join(CHECKPOINT_DIR, "best_model.pt"),
        use_wandb=False,
    )
    logger.info(f"GraphSAGE test metrics: {sage_metrics}")

    # --- GAT ---
    logger.info("")
    logger.info("-" * 40)
    logger.info("Training GAT (2L, h=128, 4 heads)")
    logger.info("-" * 40)

    gat_model = GAT(
        in_channels=data.x.shape[1],
        hidden_channels=128,
        out_channels=2,
        heads=4,
        dropout=0.3,
    )

    gat_metrics = train(
        model=gat_model,
        data=data,
        config=config,
        checkpoint_path=os.path.join(CHECKPOINT_DIR, "gat_first_run.pt"),
        use_wandb=False,
    )
    logger.info(f"GAT test metrics: {gat_metrics}")

    # --- Update model_config.json ---
    config_path = os.path.join(CHECKPOINT_DIR, "model_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            model_config = json.load(f)
    else:
        model_config = {}

    model_config["training_info"] = {
        "best_val_pr_auc": sage_metrics.get("pr_auc", 0.0),
        "best_epoch": 0,
        "test_pr_auc": sage_metrics.get("pr_auc", 0.0),
        "test_f1": sage_metrics.get("f1", 0.0),
        "test_precision": sage_metrics.get("precision", 0.0),
        "test_recall": sage_metrics.get("recall", 0.0),
        "wandb_run_id": "local-training",
        "train_time_steps": "1-34",
        "val_time_steps": "35-39",
        "test_time_steps": "40-49",
    }

    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    logger.info(f"Updated {config_path}")

    # Verify embeddings
    sage_model.eval()
    with torch.no_grad():
        emb = sage_model.get_embedding(data.x, data.edge_index)
    assert emb.shape == (203769, 128), f"Embedding shape mismatch: {emb.shape}"
    assert not torch.isnan(emb).any(), "NaN in embeddings"
    logger.info(f"get_embedding() verified: {emb.shape}")

    return sage_metrics, gat_metrics


# =========================================================================
# MAIN
# =========================================================================
def main():
    start = time.time()

    logger.info("=" * 60)
    logger.info("ON-CHAIN FRAUD DETECTION — FULL TRAINING PIPELINE")
    logger.info("=" * 60)

    # Step 1: Download
    download_dataset()

    # Step 2: Engineer features
    engineer_features()

    # Step 3: Train baselines
    baseline_results = train_baselines()

    # Step 4: Train GNNs
    sage_metrics, gat_metrics = train_gnns()

    # Summary
    elapsed = time.time() - start
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    logger.info("")
    logger.info("Artifacts saved:")
    logger.info(f"  - {OUTPUT_PARQUET}")
    logger.info(f"  - {CHECKPOINT_DIR}/best_model.pt")
    logger.info(f"  - {CHECKPOINT_DIR}/gat_first_run.pt")
    logger.info(f"  - {CHECKPOINT_DIR}/xgb_baseline.pkl")
    logger.info(f"  - {CHECKPOINT_DIR}/model_config.json")

    # Print final comparison
    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    all_results = baseline_results + [
        {"model": "GraphSAGE (3L)", **sage_metrics},
        {"model": "GAT (2L, 4h)", **gat_metrics},
    ]
    df = pd.DataFrame(all_results)
    cols_to_show = ["model", "pr_auc", "f1", "precision", "recall"]
    available = [c for c in cols_to_show if c in df.columns]
    print("\n" + df[available].to_string(index=False))


if __name__ == "__main__":
    main()
