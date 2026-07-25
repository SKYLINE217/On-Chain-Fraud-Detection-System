"""
load_predictions.py — Load GNN predictions into Neo4j.
"""

import os
import logging
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PREDICTIONS_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "predictions.csv")

def load_predictions():
    if not os.path.exists(PREDICTIONS_FILE):
        log.error(f"Predictions not found at {PREDICTIONS_FILE}")
        return

    log.info(f"Loading predictions from {PREDICTIONS_FILE}...")
    df = pd.read_csv(PREDICTIONS_FILE)
    
    # Fill any NaNs
    df["risk_score"] = df["risk_score"].fillna(0.0)
    df["confidence"] = df["confidence"].fillna(0.0)
    df["predicted_label"] = df["predicted_label"].fillna("unknown")

    records = df.to_dict("records")
    
    log.info(f"Connecting to Neo4j at {NEO4J_URI}...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    query = """
    UNWIND $batch AS row
    MATCH (t:Transaction {txId: row.txId})
    SET t.risk_score = toFloat(row.risk_score),
        t.confidence = toFloat(row.confidence),
        t.predicted_label = row.predicted_label
    """
    
    batch_size = 10000
    total_updated = 0
    
    try:
        with driver.session() as session:
            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]
                session.run(query, batch=batch)
                total_updated += len(batch)
                log.info(f"  Updated {total_updated}/{len(records)} nodes...")
                
        log.info("✓ All predictions loaded successfully.")
    except Exception as e:
        log.error(f"Failed to update Neo4j: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    load_predictions()
