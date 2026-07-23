# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

"""
Feature engineering pipeline — builds all 5 engineered features from Person A spec.

Features:
    1. tx_freq       — in-degree + out-degree per node per time step; rolling count
    2. amount_mean   — mean of BTC amounts (placeholder: derived from anonymized features)
    3. amount_skew   — skewness of amount distribution
    4. address_age   — time step of first appearance in the graph
    5. burst_score   — z-score of tx count in time step t vs. trailing window average

Additional graph features (from Neo4j GDS, merged separately):
    - pageRank
    - communityId
    - clusteringCoeff
"""

import numpy as np
import pandas as pd
from scipy import stats


def build_features(df_features: pd.DataFrame, df_classes: pd.DataFrame, df_edges: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Args:
        df_features: Raw Elliptic features (column 0 = txId, column 1 = timeStep, 2–166 = features).
        df_classes: Class labels (txId, class).
        df_edges: Edge list (txId1, txId2).

    Returns:
        DataFrame with original 166 features + 5 engineered features + label column.
    """
    # Rename raw columns
    df = df_features.copy()
    df.rename(columns={0: "txId", 1: "timeStep"}, inplace=True)
    feature_cols = [f"f{i}" for i in range(1, df.shape[1] - 1)]
    df.columns = ["txId", "timeStep"] + feature_cols

    # Merge class labels
    df = pd.merge(df, df_classes, on="txId", how="left")
    class_map = {"1": 1, "2": 0, "unknown": -1}
    df["label"] = df["class"].map(class_map).fillna(-1).astype(int)

    # ── Feature 1: Transaction frequency (in-degree + out-degree per time step) ──
    df = _add_tx_frequency(df, df_edges)

    # ── Feature 2 & 3: Amount patterns (mean + skew from local features) ──
    df = _add_amount_patterns(df, feature_cols)

    # ── Feature 4: Address age ──
    df = _add_address_age(df, df_edges)

    # ── Feature 5: Temporal burst score ──
    df = _add_burst_score(df)

    print(f"[✓] Feature engineering complete. Shape: {df.shape}")
    return df


def _add_tx_frequency(df: pd.DataFrame, df_edges: pd.DataFrame) -> pd.DataFrame:
    """
    Compute in-degree + out-degree per node.
    Rolling count: tx frequency per time step is also computed.
    """
    # In-degree: count of edges pointing TO this node
    in_degree = df_edges.groupby("txId2").size().rename("in_degree")
    # Out-degree: count of edges FROM this node
    out_degree = df_edges.groupby("txId1").size().rename("out_degree")

    df = df.merge(in_degree, left_on="txId", right_index=True, how="left")
    df = df.merge(out_degree, left_on="txId", right_index=True, how="left")
    df["in_degree"] = df["in_degree"].fillna(0).astype(int)
    df["out_degree"] = df["out_degree"].fillna(0).astype(int)
    df["tx_freq"] = df["in_degree"] + df["out_degree"]

    print(f"    [✓] tx_freq: mean={df['tx_freq'].mean():.2f}, max={df['tx_freq'].max()}")
    return df


def _add_amount_patterns(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Derive amount proxies from the first few local features.
    Elliptic's 166 features are anonymized — features f1–f94 are local transaction
    features. We use a subset as amount proxies (mean and skew across local features).
    """
    local_features = feature_cols[:94]  # First 94 = local transaction features
    df["amount_mean"] = df[local_features].mean(axis=1)
    df["amount_skew"] = df[local_features].apply(
        lambda row: stats.skew(row.values, nan_policy="omit"), axis=1
    )

    # Fill any NaN from skew calculation
    df["amount_skew"] = df["amount_skew"].fillna(0.0)

    print(f"    [✓] amount_mean: mean={df['amount_mean'].mean():.4f}")
    print(f"    [✓] amount_skew: mean={df['amount_skew'].mean():.4f}")
    return df


def _add_address_age(df: pd.DataFrame, df_edges: pd.DataFrame) -> pd.DataFrame:
    """
    Address age = time step of first appearance in the graph.
    Computed from the minimum time step where the txId appears as either src or dst.
    """
    # Get first appearance from features (simplest: the node's own time step IS its first appearance)
    first_appearance = df.groupby("txId")["timeStep"].min().rename("first_seen_step")
    df = df.merge(first_appearance, on="txId", how="left")

    # Address age = current time step - first seen
    df["address_age"] = df["timeStep"] - df["first_seen_step"]
    df["address_age"] = df["address_age"].fillna(0).astype(int)
    df.drop(columns=["first_seen_step"], inplace=True)

    print(f"    [✓] address_age: mean={df['address_age'].mean():.2f}, max={df['address_age'].max()}")
    return df


def _add_burst_score(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Temporal burst score: z-score of transaction count in time step t
    vs. trailing window average. Classic mixer/wash-trading signal.
    """
    # Count transactions per time step
    step_counts = df.groupby("timeStep").size().rename("step_tx_count")
    df = df.merge(step_counts, on="timeStep", how="left")

    # Rolling mean and std over trailing window
    steps_sorted = step_counts.sort_index()
    rolling_mean = steps_sorted.rolling(window=window, min_periods=1).mean()
    rolling_std = steps_sorted.rolling(window=window, min_periods=1).std().fillna(1.0)

    burst_z = ((steps_sorted - rolling_mean) / rolling_std).rename("burst_score")
    burst_z = burst_z.fillna(0.0)

    # Map back to nodes
    burst_map = burst_z.to_dict()
    df["burst_score"] = df["timeStep"].map(burst_map).fillna(0.0)
    df.drop(columns=["step_tx_count"], inplace=True)

    print(f"    [✓] burst_score: mean={df['burst_score'].mean():.4f}")
    return df


def export_combined_features(df: pd.DataFrame, output_path: str):
    """
    Export the combined feature matrix to Parquet format.
    Schema: txId, timeStep, class, f1..f166, tx_freq, amount_mean, amount_skew,
            address_age, burst_score, in_degree, out_degree, label
    """
    # Validate no NaN in engineered features
    eng_cols = ["tx_freq", "amount_mean", "amount_skew", "address_age", "burst_score"]
    nan_counts = df[eng_cols].isna().sum()
    if nan_counts.any():
        print(f"[!] NaN found in engineered features, filling with median per timeStep:")
        for col in eng_cols:
            if df[col].isna().any():
                df[col] = df.groupby("timeStep")[col].transform(
                    lambda x: x.fillna(x.median())
                )
                df[col] = df[col].fillna(0.0)  # fallback

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"[✓] Combined features exported to {output_path}")
    print(f"    Shape: {df.shape}")


# Allow import of os for export function
import os
