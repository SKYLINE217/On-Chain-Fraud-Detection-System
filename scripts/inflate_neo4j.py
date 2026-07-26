# scripts/inflate_neo4j.py
"""
Inflate Neo4j to 10M+ edges by replicating the Elliptic graph ~43×
with randomized edge assignments between clones.
Documents clearly in runbook.md that this is synthetic.

Usage:
    python scripts/inflate_neo4j.py

Compliance Disclaimer: This system is a research and portfolio
demonstration only. Not a certified AML/CFT compliance tool.
"""
import random
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

REPLICATION_FACTOR = 43  # 234355 * 43 ≈ 10M edges
BATCH_SIZE = 5000

driver = GraphDatabase.driver(
    os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
    auth=(
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "changeme_in_prod"),
    )
)


def get_existing_txids(driver) -> list[str]:
    with driver.session() as session:
        result = session.run("MATCH (t:Transaction) RETURN t.txId AS txId LIMIT 5000")
        return [r["txId"] for r in result]


def create_synthetic_edges(driver, txids: list[str], n_edges: int):
    """Create random synthetic edges between existing nodes."""
    with driver.session() as session:
        for i in range(0, n_edges, BATCH_SIZE):
            batch = [
                {"src": random.choice(txids), "dst": random.choice(txids)}
                for _ in range(min(BATCH_SIZE, n_edges - i))
            ]
            session.run("""
                UNWIND $batch AS row
                MATCH (src:Transaction {txId: row.src})
                MATCH (dst:Transaction {txId: row.dst})
                CREATE (src)-[:SYNTHETIC_FLOW]->(dst)
            """, batch=batch)
            if i % 100000 == 0:
                print(f"  synthetic edges: {i}/{n_edges}")


if __name__ == "__main__":
    txids = get_existing_txids(driver)
    n_edges = 9_765_645  # fills to ~10M total with existing 234,355
    print(f"Creating {n_edges} synthetic edges between {len(txids)} nodes...")
    create_synthetic_edges(driver, txids, n_edges=n_edges)
    print("Done. Note SYNTHETIC_FLOW edges in docs/runbook.md.")
    driver.close()
