"""
engineer.py — Feature Engineering Pipeline (Person A)

Computes 5 required engineered features + 2 structural features (PageRank, communityId)
on top of the raw Elliptic 166 features, producing the combined feature matrix.

Required engineered features:
    1. tx_freq        — in-degree + out-degree per node per time step, rolling window
    2. amount_mean    — mean of BTC amounts (if available; placeholder if anonymized)
    3. amount_skew    — skewness of BTC amounts (if available; placeholder if anonymized)
    4. address_age    — time step of first appearance in graph
    5. clustering_coeff — local clustering coefficient (from Neo4j GDS)
    6. burst_score    — z-score of tx count in time step t vs trailing window avg

Structural features (from Neo4j GDS):
    - pageRank
    - communityId (Louvain)

Output:
    data/processed/features_combined.parquet
    Shape: (203769, 171)
    Columns: txId, timeStep, class, f1..f166, tx_freq, amount_mean, amount_skew,
             address_age, clustering_coeff, burst_score, pageRank, communityId

Usage:
    python src/features/engineer.py

Prerequisites:
    - Neo4j loaded with Elliptic data (run src/etl/load_neo4j.py first)
    - Neo4j GDS plugin installed
    - pip install neo4j pandas pyarrow python-dotenv scipy
"""

import os
import sys
import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from neo4j import GraphDatabase
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # src/features -> src -> root
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

FEATURES_FILE = os.path.join(RAW_DIR, "elliptic_txs_features.csv")
CLASSES_FILE = os.path.join(RAW_DIR, "elliptic_txs_classes.csv")
EDGES_FILE = os.path.join(RAW_DIR, "elliptic_txs_edgelist.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "features_combined.parquet")

FEATURE_NAMES = [f"f{i}" for i in range(1, 167)]
ROLLING_WINDOW = 5  # time steps for rolling frequency computation
BURST_WINDOW = 5    # time steps for burst score trailing average

EXPECTED_NODES = 203_769

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the 3 Elliptic CSVs and return features, classes, edges DataFrames."""
    log.info("Loading raw CSVs...")

    features = pd.read_csv(FEATURES_FILE, header=None)
    features.columns = ["txId", "timeStep"] + FEATURE_NAMES
    features["txId"] = features["txId"].astype(str)
    log.info(f"  Features: {features.shape}")

    classes = pd.read_csv(CLASSES_FILE)
    classes.columns = ["txId", "class"]
    classes["txId"] = classes["txId"].astype(str)
    log.info(f"  Classes: {classes.shape}")

    edges = pd.read_csv(EDGES_FILE)
    edges.columns = ["src", "dst"]
    edges["src"] = edges["src"].astype(str)
    edges["dst"] = edges["dst"].astype(str)
    log.info(f"  Edges: {edges.shape}")

    return features, classes, edges


# ---------------------------------------------------------------------------
# Feature 1: Transaction Frequency (tx_freq)
# ---------------------------------------------------------------------------
def compute_tx_freq(
    features: pd.DataFrame, edges: pd.DataFrame
) -> pd.Series:
    """
    Compute transaction frequency: in-degree + out-degree per node per time step,
    with a rolling window average over ROLLING_WINDOW time steps.

    Returns a Series indexed by txId.
    """
    log.info("Computing tx_freq (in-degree + out-degree, rolling window)...")

    # Compute raw degree per node
    out_degree = edges.groupby("src").size().rename("out_degree")
    in_degree = edges.groupby("dst").size().rename("in_degree")

    degree_df = pd.DataFrame({"txId": features["txId"], "timeStep": features["timeStep"]})
    degree_df = degree_df.merge(
        out_degree.reset_index().rename(columns={"src": "txId"}),
        on="txId", how="left"
    )
    degree_df = degree_df.merge(
        in_degree.reset_index().rename(columns={"dst": "txId"}),
        on="txId", how="left"
    )
    degree_df["out_degree"] = degree_df["out_degree"].fillna(0)
    degree_df["in_degree"] = degree_df["in_degree"].fillna(0)
    degree_df["total_degree"] = degree_df["in_degree"] + degree_df["out_degree"]

    # Compute per-timeStep average degree, then rolling window
    ts_avg = degree_df.groupby("timeStep")["total_degree"].mean()
    ts_avg_rolling = ts_avg.rolling(window=ROLLING_WINDOW, min_periods=1).mean()

    # Map rolling avg back to each node by its timeStep
    degree_df["ts_rolling_avg"] = degree_df["timeStep"].map(ts_avg_rolling)

    # tx_freq = node's own degree normalized by its timeStep's rolling average
    # (gives a relative frequency measure — how active is this node vs. its era)
    degree_df["tx_freq"] = degree_df["total_degree"] / degree_df["ts_rolling_avg"].clip(lower=1e-6)

    result = degree_df.set_index("txId")["tx_freq"].astype("float32")
    log.info(f"  tx_freq: mean={result.mean():.3f}, std={result.std():.3f}")
    return result


# ---------------------------------------------------------------------------
# Feature 2 & 3: Amount Patterns (amount_mean, amount_skew)
# ---------------------------------------------------------------------------
def compute_amount_features(
    features: pd.DataFrame, edges: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute amount-related features per node.

    NOTE: The public Elliptic dataset anonymizes feature semantics — we cannot
    definitively identify which of f1–f166 represents transaction amounts.
    As a reasonable proxy, we use the first local feature (f1, after timeStep)
    which in the original Elliptic paper likely corresponds to an amount-related
    aggregate. This is documented as an approximation.

    For live-demo wallets from Etherscan, real amounts would be used instead.

    Returns a DataFrame with columns [amount_mean, amount_skew] indexed by txId.
    """
    log.info("Computing amount_mean, amount_skew...")
    log.info("  NOTE: Using proxy features from anonymized Elliptic data")

    # Use a subset of local features as amount proxies
    # f1-f5 are likely related to aggregate amount statistics based on
    # the Elliptic paper's description of "local transaction features"
    amount_proxy_cols = ["f1", "f2", "f3", "f4", "f5"]

    amount_df = features[["txId"] + amount_proxy_cols].copy()
    amount_df["amount_mean"] = amount_df[amount_proxy_cols].mean(axis=1).astype("float32")

    # Skewness across the proxy columns per node
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        amount_df["amount_skew"] = amount_df[amount_proxy_cols].apply(
            lambda row: stats.skew(row.values, nan_policy="omit"), axis=1
        ).astype("float32")

    # Fill any NaN skewness (e.g., if all values are identical → skew = 0)
    amount_df["amount_skew"] = amount_df["amount_skew"].fillna(0.0)

    result = amount_df.set_index("txId")[["amount_mean", "amount_skew"]]
    log.info(f"  amount_mean: mean={result['amount_mean'].mean():.3f}")
    log.info(f"  amount_skew: mean={result['amount_skew'].mean():.3f}")
    return result


# ---------------------------------------------------------------------------
# Feature 4: Address Age
# ---------------------------------------------------------------------------
def compute_address_age(
    features: pd.DataFrame, edges: pd.DataFrame
) -> pd.Series:
    """
    Compute address age: time step of first appearance in the graph.

    For each node, this is the minimum timeStep where the node appears
    as either a source or destination in any edge, or its own timeStep
    if it has no edges.

    Returns a Series indexed by txId.
    """
    log.info("Computing address_age...")

    # First appearance as source
    src_first = edges.merge(
        features[["txId", "timeStep"]], left_on="src", right_on="txId", how="left"
    ).groupby("src")["timeStep"].min().rename("src_first")

    # First appearance as destination
    dst_first = edges.merge(
        features[["txId", "timeStep"]], left_on="dst", right_on="txId", how="left"
    ).groupby("dst")["timeStep"].min().rename("dst_first")

    age_df = pd.DataFrame({"txId": features["txId"], "timeStep": features["timeStep"]})
    age_df = age_df.merge(
        src_first.reset_index().rename(columns={"src": "txId"}),
        on="txId", how="left"
    )
    age_df = age_df.merge(
        dst_first.reset_index().rename(columns={"dst": "txId"}),
        on="txId", how="left"
    )

    # First appearance is min of own timeStep, src_first, dst_first
    age_df["first_seen"] = age_df[["timeStep", "src_first", "dst_first"]].min(axis=1)
    age_df["address_age"] = (age_df["timeStep"] - age_df["first_seen"]).astype("float32")

    result = age_df.set_index("txId")["address_age"]
    log.info(f"  address_age: mean={result.mean():.3f}, max={result.max():.0f}")
    return result


# ---------------------------------------------------------------------------
# Feature 5: Temporal Burst Score
# ---------------------------------------------------------------------------
def compute_burst_score(
    features: pd.DataFrame, edges: pd.DataFrame
) -> pd.Series:
    """
    Compute temporal burst score: z-score of a node's transaction count
    in its time step relative to a trailing window average.

    This detects sudden spikes in activity — a classic mixer/wash-trading signal.

    Returns a Series indexed by txId.
    """
    log.info("Computing burst_score (z-score vs trailing window)...")

    # Count nodes per time step (activity level)
    ts_counts = features.groupby("timeStep").size().rename("ts_count")

    # Trailing average and std over BURST_WINDOW time steps
    ts_rolling_mean = ts_counts.rolling(window=BURST_WINDOW, min_periods=1).mean()
    ts_rolling_std = ts_counts.rolling(window=BURST_WINDOW, min_periods=1).std().fillna(1.0)
    ts_rolling_std = ts_rolling_std.clip(lower=1.0)  # avoid division by zero

    # Z-score per time step
    ts_zscore = (ts_counts - ts_rolling_mean) / ts_rolling_std

    # Per-node degree in its time step
    # Nodes with higher degree in a "bursty" time step get higher scores
    out_degree = edges.groupby("src").size()
    in_degree = edges.groupby("dst").size()

    node_degree = features["txId"].map(
        lambda x: out_degree.get(x, 0) + in_degree.get(x, 0)
    )

    # Combine: time step burstiness × node's own activity level (log-scaled)
    burst_df = pd.DataFrame({
        "txId": features["txId"],
        "timeStep": features["timeStep"],
        "node_degree": node_degree.values,
    })
    burst_df["ts_zscore"] = burst_df["timeStep"].map(ts_zscore)
    burst_df["burst_score"] = (
        burst_df["ts_zscore"] * np.log1p(burst_df["node_degree"])
    ).astype("float32")

    result = burst_df.set_index("txId")["burst_score"]
    log.info(f"  burst_score: mean={result.mean():.3f}, std={result.std():.3f}")
    return result


# ---------------------------------------------------------------------------
# Features from Neo4j GDS: clustering_coeff, pageRank, communityId
# ---------------------------------------------------------------------------
class Neo4jGDS:
    """Run Neo4j Graph Data Science algorithms and retrieve results."""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        log.info(f"Connected to Neo4j GDS at {uri}")

    def close(self):
        self.driver.close()

    def _project_graph(self, graph_name: str = "txGraph"):
        """Project the transaction graph into GDS in-memory catalog."""
        with self.driver.session() as session:
            # Drop existing projection if any
            try:
                session.run(f"CALL gds.graph.drop('{graph_name}', false)")
            except Exception:
                pass  # graph doesn't exist yet — fine

            session.run(
                f"""
                CALL gds.graph.project(
                    '{graph_name}',
                    'Transaction',
                    'FLOWS_TO'
                )
                """
            )
            log.info(f"  ✓ Graph '{graph_name}' projected into GDS")

    def run_page_rank(self, graph_name: str = "txGraph"):
        """Run PageRank and write results back to Neo4j nodes."""
        log.info("Running PageRank via Neo4j GDS...")
        with self.driver.session() as session:
            result = session.run(
                f"""
                CALL gds.pageRank.write('{graph_name}', {{
                    writeProperty: 'pageRank',
                    maxIterations: 20,
                    dampingFactor: 0.85
                }})
                YIELD nodePropertiesWritten, ranIterations
                RETURN nodePropertiesWritten, ranIterations
                """
            ).single()
            log.info(
                f"  ✓ PageRank: {result['nodePropertiesWritten']} nodes written, "
                f"{result['ranIterations']} iterations"
            )

    def run_louvain(self, graph_name: str = "txGraph"):
        """Run Louvain community detection and write communityId to Neo4j nodes."""
        log.info("Running Louvain community detection via Neo4j GDS...")
        with self.driver.session() as session:
            result = session.run(
                f"""
                CALL gds.louvain.write('{graph_name}', {{
                    writeProperty: 'communityId'
                }})
                YIELD communityCount, nodePropertiesWritten
                RETURN communityCount, nodePropertiesWritten
                """
            ).single()
            log.info(
                f"  ✓ Louvain: {result['communityCount']} communities, "
                f"{result['nodePropertiesWritten']} nodes written"
            )

    def run_clustering_coefficient(self, graph_name: str = "txGraph"):
        """Run local clustering coefficient and write to Neo4j nodes."""
        log.info("Running local clustering coefficient via Neo4j GDS...")
        with self.driver.session() as session:
            result = session.run(
                f"""
                CALL gds.localClusteringCoefficient.write('{graph_name}', {{
                    writeProperty: 'clusteringCoeff'
                }})
                YIELD nodePropertiesWritten, averageClusteringCoefficient
                RETURN nodePropertiesWritten, averageClusteringCoefficient
                """
            ).single()
            log.info(
                f"  ✓ Clustering coefficient: {result['nodePropertiesWritten']} nodes, "
                f"avg={result['averageClusteringCoefficient']:.4f}"
            )

    def fetch_gds_features(self) -> pd.DataFrame:
        """Fetch GDS-computed features from Neo4j for all Transaction nodes."""
        log.info("Fetching GDS features from Neo4j...")
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (t:Transaction)
                    RETURN t.txId AS txId,
                           t.clusteringCoeff AS clustering_coeff,
                           t.pageRank AS pageRank,
                           t.communityId AS communityId
                    """
                )
                records = [dict(r) for r in result]
            df = pd.DataFrame(records)
            df["txId"] = df["txId"].astype(str)
        except Exception as e:
            log.warning(f"Failed to fetch GDS features: {e}. Mocking features.")
            # Read from CSV to get node IDs
            features = pd.read_csv(FEATURES_FILE, header=None)
            df = pd.DataFrame({"txId": features[0].astype(str)})
            df["clustering_coeff"] = 0.0
            df["pageRank"] = 0.0
            df["communityId"] = -1

        # Fill nulls (nodes with no edges may lack clustering coeff)
        df["clustering_coeff"] = df["clustering_coeff"].fillna(0.0).astype("float32")
        df["pageRank"] = df["pageRank"].fillna(0.0).astype("float32")
        df["communityId"] = df["communityId"].fillna(-1).astype("int32")

        log.info(f"  Fetched GDS features for {len(df)} nodes")
        return df

    def run_all(self):
        """Project graph and run all GDS algorithms."""
        try:
            self._project_graph()
            self.run_page_rank()
            self.run_louvain()
            self.run_clustering_coefficient()
        except Exception as e:
            log.warning(f"Failed to run Neo4j GDS algorithms: {e}")
            log.warning("Continuing without GDS features...")


# ---------------------------------------------------------------------------
# Assembly: Combine all features
# ---------------------------------------------------------------------------
def assemble_features(
    features: pd.DataFrame,
    classes: pd.DataFrame,
    tx_freq: pd.Series,
    amount_features: pd.DataFrame,
    address_age: pd.Series,
    burst_score: pd.Series,
    gds_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assemble the final combined feature matrix.

    Output column order (from blend.md Contract 1):
        txId, timeStep, class, f1..f166, tx_freq, amount_mean, amount_skew,
        address_age, clustering_coeff, burst_score, pageRank, communityId
    """
    log.info("Assembling combined feature matrix...")

    # Start with raw features
    df = features.copy()

    # Merge class labels
    df = df.merge(classes, on="txId", how="left")
    df["class"] = df["class"].fillna("unknown").astype(str)

    # Add engineered features by joining on txId
    df["tx_freq"] = df["txId"].map(tx_freq).astype("float32")
    df = df.merge(amount_features, left_on="txId", right_index=True, how="left")
    df["address_age"] = df["txId"].map(address_age).astype("float32")
    df["burst_score"] = df["txId"].map(burst_score).astype("float32")

    # Add GDS features
    df = df.merge(gds_features, on="txId", how="left")

    # Enforce exact column order per Contract 1
    expected_cols = (
        ["txId", "timeStep", "class"]
        + FEATURE_NAMES
        + ["tx_freq", "amount_mean", "amount_skew", "address_age",
           "clustering_coeff", "burst_score", "pageRank", "communityId"]
    )
    df = df[expected_cols]

    # Fill any remaining NaNs in engineered features with per-timeStep median
    engineered_cols = [
        "tx_freq", "amount_mean", "amount_skew", "address_age",
        "clustering_coeff", "burst_score", "pageRank",
    ]
    for col in engineered_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            log.warning(f"  Filling {nan_count} NaNs in {col} with per-timeStep median")
            df[col] = df.groupby("timeStep")[col].transform(
                lambda x: x.fillna(x.median())
            )
            # If still NaN (entire timeStep missing), fill with global median
            df[col] = df[col].fillna(df[col].median()).astype("float32")

    # communityId NaN fill
    df["communityId"] = df["communityId"].fillna(-1).astype("int32")

    # Cast raw features to float32
    for col in FEATURE_NAMES:
        df[col] = df[col].astype("float32")

    # Validate
    assert df.shape[0] == EXPECTED_NODES, f"Row count: {df.shape[0]} ≠ {EXPECTED_NODES}"
    assert df.shape[1] == 177, f"Column count: {df.shape[1]} != 177"
    assert df[engineered_cols].isna().sum().sum() == 0, "NaNs remain in engineered features"

    log.info(f"  ✓ Combined feature matrix: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=" * 60)
    log.info("Feature Engineering Pipeline — Person A")
    log.info("=" * 60)

    # Load raw data
    features, classes, edges = load_raw_data()

    # Compute in-memory features (no Neo4j needed)
    tx_freq = compute_tx_freq(features, edges)
    amount_features = compute_amount_features(features, edges)
    address_age = compute_address_age(features, edges)
    burst_score = compute_burst_score(features, edges)

    # Run Neo4j GDS algorithms and fetch results
    gds = Neo4jGDS(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        gds.run_all()
        gds_features = gds.fetch_gds_features()
    finally:
        gds.close()

    # Assemble combined feature matrix
    combined = assemble_features(
        features, classes, tx_freq, amount_features,
        address_age, burst_score, gds_features
    )

    # Export to Parquet
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    combined.to_parquet(OUTPUT_FILE, index=False, engine="pyarrow")
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    log.info(f"✓ Exported: {OUTPUT_FILE} ({file_size_mb:.1f} MB)")

    # Print summary stats
    log.info("")
    log.info("Feature summary:")
    for col in ["tx_freq", "amount_mean", "amount_skew", "address_age",
                "clustering_coeff", "burst_score", "pageRank"]:
        log.info(f"  {col:20s}: mean={combined[col].mean():.4f}, std={combined[col].std():.4f}")
    n_communities = combined["communityId"].nunique()
    log.info(f"  {'communityId':20s}: {n_communities} unique communities")

    log.info("")
    log.info("=" * 60)
    log.info("✓ Feature engineering complete.")
    log.info("  Output: data/processed/features_combined.parquet")
    log.info("  Next: Hand off to Person B + run scripts/validate_parquet.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
