# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

"""
Neo4j ETL Loader — loads Elliptic CSV data into Neo4j graph database.

Node schema:
    (:Transaction {txId, timeStep, class, f1..f166})

Edge schema:
    (:Transaction)-[:FLOWS_TO]->(:Transaction)

Uses UNWIND + MERGE to avoid duplicates on re-runs.
"""

import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
# BUG-02 Fix: No default password — fail loudly if not set
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
if not NEO4J_PASSWORD:
    import warnings
    warnings.warn(
        "NEO4J_PASSWORD is not set. Copy .env.example to .env and configure it.",
        RuntimeWarning,
        stacklevel=2,
    )
    NEO4J_PASSWORD = ""  # Empty — will fail on connect


class Neo4jLoader:
    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def verify_connectivity(self):
        """Verify Neo4j is reachable."""
        with self.driver.session() as session:
            result = session.run("RETURN 1 AS ping")
            assert result.single()["ping"] == 1
            print("[✓] Neo4j connectivity verified.")

    def load_transactions(self, features_path: str, classes_path: str, batch_size: int = 5000):
        """
        Batch-load transaction nodes from Elliptic CSVs.
        Merges on txId to avoid duplicates on re-runs.
        """
        df_features = pd.read_csv(features_path, header=None)
        df_features.rename(columns={0: "txId", 1: "timeStep"}, inplace=True)

        df_classes = pd.read_csv(classes_path)
        df_merged = pd.merge(df_features, df_classes, left_on="txId", right_on="txId", how="left")
        df_merged["class"] = df_merged["class"].fillna("unknown")

        feature_cols = [c for c in df_merged.columns if c not in ["txId", "timeStep", "class"]]

        # BUG-25 Fix: Replace slow iterrows() with vectorized to_dict("records")
        # Original iterrows() on 200K rows took 30-120 seconds
        df_merged["txId"] = df_merged["txId"].astype(str)
        df_merged["timeStep"] = df_merged["timeStep"].astype(int)
        df_merged = df_merged.rename(columns={"class": "txClass"})

        # Build records with renamed feature columns f1..f166
        records = []
        for i, col in enumerate(feature_cols, start=1):
            df_merged[f"f{i}"] = df_merged[col].astype(float)
        keep_cols = ["txId", "timeStep", "txClass"] + [f"f{i}" for i in range(1, len(feature_cols) + 1)]
        records = df_merged[keep_cols].to_dict("records")

        print(f"[→] Loading {len(records)} transaction nodes in batches of {batch_size}...")
        with self.driver.session() as session:
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                session.run(
                    """
                    UNWIND $batch AS row
                    MERGE (t:Transaction {txId: row.txId})
                    SET t.timeStep = row.timeStep,
                        t.class    = row.txClass
                    """,
                    batch=batch,
                )
                print(f"    Loaded nodes {start + 1}–{min(start + batch_size, len(records))}")

        print("[✓] Transaction nodes loaded.")

    def load_edges(self, edgelist_path: str, batch_size: int = 10000):
        """Batch-load FLOWS_TO edges from the edgelist CSV."""
        df_edges = pd.read_csv(edgelist_path)
        # BUG-26 Fix: Replace slow iterrows() on 234K rows with vectorized to_dict
        df_edges = df_edges.astype(str)
        edges = df_edges.rename(columns={"txId1": "src", "txId2": "dst"})[["src", "dst"]].to_dict("records")

        print(f"[→] Loading {len(edges)} edges in batches of {batch_size}...")
        with self.driver.session() as session:
            for start in range(0, len(edges), batch_size):
                batch = edges[start : start + batch_size]
                session.run(
                    """
                    UNWIND $batch AS row
                    MATCH (a:Transaction {txId: row.src})
                    MATCH (b:Transaction {txId: row.dst})
                    MERGE (a)-[:FLOWS_TO]->(b)
                    """,
                    batch=batch,
                )
                print(f"    Loaded edges {start + 1}–{min(start + batch_size, len(edges))}")

        print("[✓] Edges loaded.")

    def validate_counts(self, expected_nodes: int = 203769, expected_edges: int = 234355):
        """Post-load validation — counts must match Elliptic dataset specification."""
        with self.driver.session() as session:
            node_count = session.run("MATCH (n:Transaction) RETURN count(n) AS c").single()["c"]
            edge_count = session.run("MATCH ()-[r:FLOWS_TO]->() RETURN count(r) AS c").single()["c"]

        print(f"    Nodes: {node_count} (expected {expected_nodes})")
        print(f"    Edges: {edge_count} (expected {expected_edges})")
        assert node_count == expected_nodes, f"Node count mismatch: {node_count} != {expected_nodes}"
        assert edge_count == expected_edges, f"Edge count mismatch: {edge_count} != {expected_edges}"
        print("[✓] Counts validated.")


if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
    loader = Neo4jLoader()
    try:
        loader.verify_connectivity()
        loader.load_transactions(
            features_path=os.path.join(DATA_DIR, "elliptic_txs_features.csv"),
            classes_path=os.path.join(DATA_DIR, "elliptic_txs_classes.csv"),
        )
        loader.load_edges(
            edgelist_path=os.path.join(DATA_DIR, "elliptic_txs_edgelist.csv"),
        )
        loader.validate_counts()
    finally:
        loader.close()
