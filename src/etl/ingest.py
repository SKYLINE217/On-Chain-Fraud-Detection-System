# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

"""
Data ingestion module — loads and validates the Elliptic dataset CSVs.

Expected files:
    elliptic_txs_features.csv  — 203,769 rows × 167 cols (txId + 166 features)
    elliptic_txs_classes.csv   — txId, class ∈ {1=illicit, 2=licit, unknown}
    elliptic_txs_edgelist.csv  — directed edges txId1 → txId2
"""

import os
import pandas as pd


EXPECTED_NODE_COUNT = 203769
EXPECTED_EDGE_COUNT = 234355
EXPECTED_FEATURE_COLS = 167  # txId + timeStep-like index + 165 features = 167


def load_raw_data(data_dir: str) -> tuple:
    """
    Loads the three raw Elliptic CSV files.

    Args:
        data_dir: Path to the directory containing the CSVs.

    Returns:
        Tuple of (df_features, df_classes, df_edges).

    Raises:
        FileNotFoundError: If any required file is missing.
        ValueError: If row/column counts do not match expected values.
    """
    features_path = os.path.join(data_dir, "elliptic_txs_features.csv")
    classes_path = os.path.join(data_dir, "elliptic_txs_classes.csv")
    edgelist_path = os.path.join(data_dir, "elliptic_txs_edgelist.csv")

    for path, name in [
        (features_path, "features"),
        (classes_path, "classes"),
        (edgelist_path, "edgelist"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Elliptic {name} file not found at {path}. "
                f"Run scripts/download_elliptic.sh first."
            )

    # Load CSVs
    df_features = pd.read_csv(features_path, header=None)
    df_classes = pd.read_csv(classes_path)
    df_edges = pd.read_csv(edgelist_path)

    # Validate counts
    _validate_counts(df_features, df_edges)

    return df_features, df_classes, df_edges


def _validate_counts(df_features: pd.DataFrame, df_edges: pd.DataFrame):
    """
    Post-load validation — row counts must match Elliptic dataset specification.
    Raises ValueError on mismatch.
    """
    node_count = len(df_features)
    edge_count = len(df_edges)

    if node_count != EXPECTED_NODE_COUNT:
        raise ValueError(
            f"Node count mismatch: got {node_count}, expected {EXPECTED_NODE_COUNT}. "
            f"Investigate data integrity before proceeding."
        )

    if edge_count != EXPECTED_EDGE_COUNT:
        raise ValueError(
            f"Edge count mismatch: got {edge_count}, expected {EXPECTED_EDGE_COUNT}. "
            f"Investigate data integrity before proceeding."
        )

    print(f"[✓] Validation passed: {node_count} nodes, {edge_count} edges.")


def get_class_distribution(df_classes: pd.DataFrame) -> dict:
    """
    Returns class label distribution.
    Expected: ~2% illicit (class 1), ~21% licit (class 2), ~77% unknown.
    """
    dist = df_classes["class"].value_counts(normalize=True).to_dict()
    print("[i] Class distribution:")
    for cls, pct in sorted(dist.items()):
        label = {"1": "illicit", "2": "licit", "unknown": "unknown"}.get(str(cls), str(cls))
        print(f"    {label}: {pct:.1%}")
    return dist
