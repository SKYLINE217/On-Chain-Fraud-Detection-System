# src/features/engineer.py
"""
Compute 5 engineered features + run GDS algorithms on Neo4j.
Reads CSVs directly (not Neo4j) for vectorized speed; writes GDS results back to Neo4j.

Compliance Disclaimer: This system is a research and portfolio
demonstration only. Not a certified AML/CFT compliance tool.
"""
import pandas as pd
import numpy as np
from scipy.stats import skew
from neo4j import GraphDatabase
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

FEATURES_PATH  = "data/raw/elliptic_txs_features.csv"
CLASSES_PATH   = "data/raw/elliptic_txs_classes.csv"
EDGELIST_PATH  = "data/raw/elliptic_txs_edgelist.csv"
OUTPUT_PATH    = "data/processed/features_combined.parquet"
FEATURE_COLS   = [f"f{i}" for i in range(1, 167)]


def load_base(features_path, classes_path, edgelist_path):
    features = pd.read_csv(features_path, header=None,
                           names=["txId", "timeStep"] + FEATURE_COLS)
    classes  = pd.read_csv(classes_path)
    edges    = pd.read_csv(edgelist_path, header=None, names=["src", "dst"])
    df = features.merge(classes, on="txId", how="left")
    df["class"] = df["class"].fillna("unknown").astype(str)
    return df, edges


def compute_tx_freq(df: pd.DataFrame, edges: pd.DataFrame) -> pd.Series:
    """In+out degree per node per timestep (rolling 3-step window)."""
    out_deg = edges.groupby("src").size().rename("out_deg")
    in_deg  = edges.groupby("dst").size().rename("in_deg")
    deg = pd.DataFrame({"txId": df["txId"]})
    deg = deg.join(out_deg, on="txId").join(in_deg, on="txId").fillna(0)
    return (deg["out_deg"] + deg["in_deg"]).rename("tx_freq")


def compute_amount_features(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Proxy BTC amount from Elliptic raw features.
    Features f93-f94 approximate input/output amounts on local tx.
    If Etherscan CSV is available, merge it in.
    """
    # Use f93 (local total BTC input proxy) as amount surrogate
    # Mean + skew computed per communityId group (approximated per-node here)
    # Real amount enrichment would come from Etherscan merge

    # Rolling window stats per timeStep
    amount_mean = (
        df.groupby("timeStep")["f93"]
        .transform("mean")
        .rename("amount_mean")
    )
    # Skew per timeStep
    amount_skew = (
        df.groupby("timeStep")["f93"]
        .transform(lambda x: skew(x.fillna(0)))
        .rename("amount_skew")
    )
    return amount_mean, amount_skew


def compute_address_age(df: pd.DataFrame) -> pd.Series:
    """Timestep of first appearance per txId."""
    first_seen = df.groupby("txId")["timeStep"].transform("min")
    return first_seen.rename("address_age")


def compute_burst_score(df: pd.DataFrame, edges: pd.DataFrame) -> np.ndarray:
    """
    Z-score of tx count vs trailing 3-step window average.
    High burst_score = sudden spike in activity.
    """
    out_deg = edges.groupby("src").size().rename("out_deg").reset_index()
    out_deg.columns = ["txId", "out_deg"]
    df_with_deg = df[["txId", "timeStep"]].merge(out_deg, on="txId", how="left").fillna(0)

    step_mean = df_with_deg.groupby("timeStep")["out_deg"].transform("mean")
    step_std  = df_with_deg.groupby("timeStep")["out_deg"].transform("std").replace(0, 1)
    burst = ((df_with_deg["out_deg"] - step_mean) / step_std).rename("burst_score")
    return burst.values  # return as array aligned to df index


def run_gds_algorithms(driver) -> pd.DataFrame:
    """
    Run GDS PageRank + Louvain + LocalClusteringCoefficient.
    Writes properties directly to Neo4j nodes.
    CRITICAL: Run ONCE. Never re-run GDS separately from parquet export.
    """
    with driver.session() as session:
        # Create in-memory graph projection
        session.run("""
            CALL gds.graph.project(
              'fraud-graph',
              'Transaction',
              'FLOWS_TO'
            )
        """)
        logger.info("Graph projected.")

        # PageRank
        session.run("""
            CALL gds.pageRank.write('fraud-graph', {
              writeProperty: 'pageRank',
              maxIterations: 20,
              dampingFactor: 0.85
            })
        """)
        logger.info("PageRank written.")

        # Louvain community detection
        session.run("""
            CALL gds.louvain.write('fraud-graph', {
              writeProperty: 'communityId'
            })
        """)
        logger.info("Louvain communityId written.")

        # Local clustering coefficient
        session.run("""
            CALL gds.localClusteringCoefficient.write('fraud-graph', {
              writeProperty: 'clusteringCoeff'
            })
        """)
        logger.info("ClusteringCoeff written.")

        # Drop projection to free memory
        session.run("CALL gds.graph.drop('fraud-graph')")

    logger.info("GDS complete. Reading back properties...")
    # Read all GDS-computed properties back from Neo4j
    with driver.session() as session:
        result = session.run("""
            MATCH (t:Transaction)
            RETURN t.txId AS txId, t.pageRank AS pageRank,
                   t.communityId AS communityId, t.clusteringCoeff AS clusteringCoeff
        """)
        records = [r.data() for r in result]

    return pd.DataFrame(records)


def build_features_parquet():
    """Full pipeline: load → engineer → GDS → export parquet."""
    os.makedirs("data/processed", exist_ok=True)

    logger.info("Loading base data...")
    df, edges = load_base(FEATURES_PATH, CLASSES_PATH, EDGELIST_PATH)

    logger.info("Computing engineered features...")
    df["tx_freq"]     = compute_tx_freq(df, edges)
    df["amount_mean"], df["amount_skew"] = compute_amount_features(df)
    df["address_age"] = compute_address_age(df)
    df["burst_score"] = compute_burst_score(df, edges)

    # Fill NaN amounts with per-timeStep median
    for col in ["amount_mean", "amount_skew"]:
        df[col] = df.groupby("timeStep")[col].transform(
            lambda x: x.fillna(x.median())
        )

    # Run GDS algorithms and merge
    driver = GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.environ.get("NEO4J_USER", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "changeme_in_prod"),
        )
    )
    try:
        gds_df = run_gds_algorithms(driver)
    finally:
        driver.close()

    df = df.merge(gds_df, on="txId", how="left")

    # Final column order (must match model_config.json feature_columns)
    final_cols = (
        ["txId", "timeStep", "class"]
        + FEATURE_COLS
        + ["tx_freq", "amount_mean", "amount_skew", "address_age",
           "clusteringCoeff", "burst_score", "pageRank", "communityId"]
    )
    df = df[final_cols]

    # Rename clusteringCoeff → clustering_coeff for consistency with ml_models.md
    df = df.rename(columns={"clusteringCoeff": "clustering_coeff"})

    # Validate shape and NaNs
    assert df.shape == (203769, 171), f"Shape mismatch: {df.shape}"
    nan_count = df.drop(columns=["class"]).select_dtypes("number").isna().sum().sum()
    assert nan_count == 0, f"NaN found: {nan_count}"

    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Saved: {OUTPUT_PATH}  shape={df.shape}")
    return df


if __name__ == "__main__":
    build_features_parquet()
