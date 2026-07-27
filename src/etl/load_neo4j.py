import pandas as pd
from neo4j import GraphDatabase
import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

NEO4J_URI  = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASSWORD")
if not NEO4J_PASS:
    raise ValueError("NEO4J_PASSWORD environment variable is not set")

FEATURES_PATH  = "data/raw/elliptic_txs_features.csv"
CLASSES_PATH   = "data/raw/elliptic_txs_classes.csv"
EDGELIST_PATH  = "data/raw/elliptic_txs_edgelist.csv"

FEATURE_COLS = [f"f{i}" for i in range(1, 167)]
BATCH_SIZE = 1000

def load_nodes(driver):
    """Load Transaction nodes with all 166 features + class + timeStep."""
    logger.info("Loading features CSV...")
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"{FEATURES_PATH} not found. Run download_elliptic script first.")

    features_df = pd.read_csv(
        FEATURES_PATH, header=None,
        names=["txId", "timeStep"] + FEATURE_COLS
    )
    features_df["txId"] = features_df["txId"].astype(int)

    logger.info("Loading classes CSV...")
    classes_df = pd.read_csv(CLASSES_PATH)
    classes_df["class"] = classes_df["class"].replace({"unknown": "unknown", "1": "1", "2": "2"})

    df = features_df.merge(classes_df, on="txId", how="left")
    df["class"] = df["class"].fillna("unknown")

    logger.info(f"Loading {len(df)} nodes into Neo4j (batch_size={BATCH_SIZE})...")

    node_props = ["txId", "timeStep", "class"] + FEATURE_COLS
    records = df[node_props].to_dict("records")

    with driver.session() as session:
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i : i + BATCH_SIZE]
            session.run(
                """
                UNWIND $batch AS row
                MERGE (t:Transaction {txId: row.txId})
                SET t += row
                """,
                batch=batch
            )
            if i % 10000 == 0:
                logger.info(f"  nodes: {i}/{len(records)}")

    with driver.session() as session:
        count = session.run("MATCH (n:Transaction) RETURN count(n) AS cnt").single()["cnt"]
        assert count == 203769, f"Node count mismatch: {count}"
        logger.info(f"Node count validated: {count}")

def load_edges(driver):
    """Load FLOWS_TO edges."""
    logger.info("Loading edgelist CSV...")
    edges_df = pd.read_csv(EDGELIST_PATH)
    edges_df.columns = ["src", "dst"]
    edges_df["src"] = edges_df["src"].astype(int)
    edges_df["dst"] = edges_df["dst"].astype(int)
    records = edges_df.to_dict("records")

    logger.info(f"Loading {len(records)} edges into Neo4j...")
    with driver.session() as session:
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i : i + BATCH_SIZE]
            session.run(
                """
                UNWIND $batch AS row
                MATCH (src:Transaction {txId: row.src})
                MATCH (dst:Transaction {txId: row.dst})
                MERGE (src)-[:FLOWS_TO]->(dst)
                """,
                batch=batch
            )
            if i % 10000 == 0:
                logger.info(f"  edges: {i}/{len(records)}")

    with driver.session() as session:
        count = session.run("MATCH ()-[r:FLOWS_TO]->() RETURN count(r) AS cnt").single()["cnt"]
        assert count == 234355, f"Edge count mismatch: {count}"
        logger.info(f"Edge count validated: {count}")

def create_indexes(driver):
    """Create indexes for fast lookups (idempotent)."""
    indexes = [
        "CREATE INDEX txId_idx IF NOT EXISTS FOR (t:Transaction) ON (t.txId)",
        "CREATE INDEX timeStep_idx IF NOT EXISTS FOR (t:Transaction) ON (t.timeStep)",
        "CREATE INDEX communityId_idx IF NOT EXISTS FOR (t:Transaction) ON (t.communityId)",
    ]
    with driver.session() as session:
        for idx in indexes:
            session.run(idx)
    logger.info("Indexes created (or already exist).")

if __name__ == "__main__":
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    try:
        create_indexes(driver)
        load_nodes(driver)
        load_edges(driver)
        logger.info("ETL complete.")
    finally:
        driver.close()
