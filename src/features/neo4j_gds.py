# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

"""
Neo4j GDS feature extraction — runs graph algorithms and exports results.

Algorithms:
    - PageRank           → node influence/importance
    - Louvain            → community detection (communityId)
    - Local Clustering   → clustering coefficient

Results are stored as Neo4j node properties and can be exported to merge
with the feature DataFrame before PyG construction.
"""

import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


class GDSFeatureExtractor:
    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def ensure_projection(self, graph_name: str = "txGraph"):
        """Create the GDS graph projection if it doesn't exist."""
        with self.driver.session() as session:
            # Check if projection exists
            result = session.run("CALL gds.graph.exists($name) YIELD exists", name=graph_name)
            exists = result.single()["exists"]

            if not exists:
                session.run(
                    """
                    CALL gds.graph.project(
                        $name,
                        'Transaction',
                        'FLOWS_TO'
                    )
                    """,
                    name=graph_name,
                )
                print(f"[✓] GDS projection '{graph_name}' created.")
            else:
                print(f"[i] GDS projection '{graph_name}' already exists.")

    def run_pagerank(self, graph_name: str = "txGraph"):
        """Run PageRank and write results as node property."""
        with self.driver.session() as session:
            result = session.run(
                "CALL gds.pageRank.write($name, { writeProperty: 'pageRank' })",
                name=graph_name,
            )
            stats = result.single()
            print(f"[✓] PageRank complete. Nodes processed: {stats.get('nodePropertiesWritten', 'N/A')}")

    def run_louvain(self, graph_name: str = "txGraph"):
        """Run Louvain community detection and write communityId."""
        with self.driver.session() as session:
            result = session.run(
                "CALL gds.louvain.write($name, { writeProperty: 'communityId' })",
                name=graph_name,
            )
            stats = result.single()
            print(f"[✓] Louvain complete. Communities: {stats.get('communityCount', 'N/A')}")

    def run_clustering_coefficient(self, graph_name: str = "txGraph"):
        """Run local clustering coefficient and write as node property."""
        with self.driver.session() as session:
            result = session.run(
                "CALL gds.localClusteringCoefficient.write($name, { writeProperty: 'clusteringCoeff' })",
                name=graph_name,
            )
            stats = result.single()
            print(f"[✓] Clustering coefficient complete. Avg: {stats.get('averageClusteringCoefficient', 'N/A'):.4f}")

    def run_all(self, graph_name: str = "txGraph"):
        """Run all GDS algorithms in sequence."""
        print("── Running GDS algorithms ──")
        self.ensure_projection(graph_name)
        self.run_pagerank(graph_name)
        self.run_louvain(graph_name)
        self.run_clustering_coefficient(graph_name)
        print("[✓] All GDS algorithms complete.")

    def export_gds_features(self) -> pd.DataFrame:
        """
        Export GDS-computed features from Neo4j as a DataFrame.
        Returns: DataFrame with columns [txId, pageRank, communityId, clusteringCoeff].
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (t:Transaction)
                RETURN t.txId AS txId,
                       t.pageRank AS pageRank,
                       t.communityId AS communityId,
                       t.clusteringCoeff AS clusteringCoeff
                """
            )
            records = [dict(r) for r in result]

        df = pd.DataFrame(records)
        print(f"[✓] Exported GDS features for {len(df)} nodes.")
        return df


if __name__ == "__main__":
    extractor = GDSFeatureExtractor()
    try:
        extractor.run_all()
        df_gds = extractor.export_gds_features()
        print(df_gds.head())
    finally:
        extractor.close()
