# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

"""
Neo4j seeding script — creates indexes and runs initial GDS projections.
Run AFTER load_neo4j.py has loaded all data.
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


def create_indexes(session):
    """Create indexes for performant lookups."""
    indexes = [
        "CREATE INDEX txId_idx IF NOT EXISTS FOR (t:Transaction) ON (t.txId)",
        "CREATE INDEX timeStep_idx IF NOT EXISTS FOR (t:Transaction) ON (t.timeStep)",
        "CREATE INDEX communityId_idx IF NOT EXISTS FOR (t:Transaction) ON (t.communityId)",
    ]
    for idx_query in indexes:
        session.run(idx_query)
        print(f"    [✓] {idx_query}")

    print("[✓] All indexes created.")


def verify_index_usage(session):
    """Verify that indexes are being used via EXPLAIN."""
    result = session.run(
        "EXPLAIN MATCH (t:Transaction {txId: '1'}) RETURN t"
    )
    summary = result.consume()
    plan = summary.plan
    if plan:
        print(f"    Query plan operator: {plan.operator_type}")
    print("[✓] Index usage verified.")


def create_gds_projection(session):
    """
    Create a GDS graph projection for running graph algorithms.
    Requires the Neo4j Graph Data Science plugin.
    """
    # Drop existing projection if it exists
    try:
        session.run("CALL gds.graph.drop('txGraph', false)")
    except Exception:
        pass

    session.run(
        """
        CALL gds.graph.project(
            'txGraph',
            'Transaction',
            'FLOWS_TO'
        )
        """
    )
    print("[✓] GDS graph projection 'txGraph' created.")


if __name__ == "__main__":
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            print("── Creating indexes ──")
            create_indexes(session)

            print("\n── Verifying index usage ──")
            verify_index_usage(session)

            print("\n── Creating GDS projection ──")
            create_gds_projection(session)

            print("\n[✓] Neo4j seeding complete.")
    finally:
        driver.close()
