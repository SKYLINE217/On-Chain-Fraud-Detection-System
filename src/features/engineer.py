
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
    features["txId"] = features["txId"].astype(int)
    classes  = pd.read_csv(classes_path)
    edges    = pd.read_csv(edgelist_path)
    edges.columns = ["src", "dst"]
    edges["src"] = edges["src"].astype(int)
    edges["dst"] = edges["dst"].astype(int)
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

    amount_mean = (
        df.groupby("timeStep")["f93"]
        .transform("mean")
        .rename("amount_mean")
    )

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
    return burst.values  

def run_gds_algorithms(driver) -> pd.DataFrame:
    """
    Run GDS PageRank + Louvain + LocalClusteringCoefficient.
    Writes properties directly to Neo4j nodes.
    CRITICAL: Run ONCE. Never re-run GDS separately from parquet export.
    """
    with driver.session() as session:

        session.run("CALL gds.graph.drop('fraud-graph', false) YIELD graphName")

        session.run("""
            CALL gds.graph.project(
              'fraud-graph',
              'Transaction',
              'FLOWS_TO'
            )
        """)
        logger.info("Graph projected.")

        session.run("""
            CALL gds.pageRank.write('fraud-graph', {
              writeProperty: 'pageRank',
              maxIterations: 20,
              dampingFactor: 0.85,
              concurrency: 1
            })
        """)
        logger.info("PageRank written.")

        session.run("""
            CALL gds.louvain.write('fraud-graph', {
              writeProperty: 'communityId',
              concurrency: 1
            })
        """)
        logger.info("Louvain communityId written.")

        session.run("CALL gds.graph.drop('fraud-graph', false) YIELD graphName")

        session.run("CALL gds.graph.drop('fraud-graph-undir', false) YIELD graphName")

        session.run("""
            CALL gds.graph.project(
              'fraud-graph-undir',
              'Transaction',
              { FLOWS_TO: { orientation: 'UNDIRECTED' } }
            )
        """)
        session.run("""
            CALL gds.localClusteringCoefficient.write('fraud-graph-undir', {
              writeProperty: 'clusteringCoeff',
              concurrency: 1
            })
        """)
        logger.info("ClusteringCoeff written.")

        session.run("CALL gds.graph.drop('fraud-graph-undir', false) YIELD graphName")

    logger.info("GDS complete. Reading back properties...")

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

    for col in ["amount_mean", "amount_skew"]:
        df[col] = df.groupby("timeStep")[col].transform(
            lambda x: x.fillna(x.median())
        )

    neo4j_pwd = os.environ.get("NEO4J_PASSWORD")
    if not neo4j_pwd:
        raise ValueError("NEO4J_PASSWORD environment variable is not set")

    driver = GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.environ.get("NEO4J_USER", "neo4j"),
            neo4j_pwd,
        )
    )
    try:
        gds_df = run_gds_algorithms(driver)
    finally:
        driver.close()

    df = df.merge(gds_df, on="txId", how="left")

    final_cols = (
        ["txId", "timeStep", "class"]
        + FEATURE_COLS
        + ["tx_freq", "amount_mean", "amount_skew", "address_age",
           "clusteringCoeff", "burst_score", "pageRank", "communityId"]
    )
    df = df[final_cols]

    df = df.rename(columns={"clusteringCoeff": "clustering_coeff"})

    assert df.shape == (203769, 177), f"Shape mismatch: {df.shape}"
    nan_counts = df.drop(columns=["class"]).select_dtypes("number").isna().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"NaN found in columns:\n{nan_counts[nan_counts > 0]}")
        df = df.fillna(0)

    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Saved: {OUTPUT_PATH}  shape={df.shape}")
    return df

if __name__ == "__main__":
    build_features_parquet()
