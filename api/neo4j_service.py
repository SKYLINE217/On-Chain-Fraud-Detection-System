# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

"""
Neo4j service layer — async-compatible query helpers for all API endpoints.
Centralizes driver management, connection pooling, and query patterns.
"""

import os
import json
import logging
from contextlib import contextmanager
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


class Neo4jService:
    """
    Singleton-style Neo4j service for the API layer.
    Manages the driver lifecycle and provides query methods
    for wallet lookups, subgraph traversals, and cluster queries.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._driver = None
        return cls._instance

    @property
    def driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            logger.info("Neo4j driver initialized: %s", NEO4J_URI)
        return self._driver

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None

    @contextmanager
    def session(self):
        s = self.driver.session()
        try:
            yield s
        finally:
            s.close()

    # ── Wallet Lookup ─────────────────────────────────────────────────────

    def get_wallet(self, address: str) -> dict | None:
        """
        Fetch a single transaction node by txId.
        Returns risk_score, predicted_label, confidence, community, pageRank.
        Uses txId_idx index for sub-5s lookup.
        """
        with self.session() as s:
            result = s.run(
                """
                MATCH (t:Transaction {txId: $address})
                RETURN t.txId          AS txId,
                       t.timeStep      AS timeStep,
                       t.class         AS txClass,
                       t.risk_score    AS risk_score,
                       t.predicted_label AS predicted_label,
                       t.confidence    AS confidence,
                       t.communityId   AS communityId,
                       t.pageRank      AS pageRank,
                       t.clusteringCoeff AS clusteringCoeff
                """,
                address=address,
            )
            record = result.single()
            if record is None:
                return None
            return dict(record)

    # ── K-Hop Subgraph ────────────────────────────────────────────────────

    def get_subgraph(self, address: str, hops: int = 2, max_nodes: int = 200) -> dict:
        """
        Fetch k-hop ego-graph from Neo4j, capped at max_nodes.
        HARD CAPS (per spec): max 2 hops, max 200 nodes — never unbounded.

        Returns:
            {
                "center": address,
                "hops": hops,
                "nodes": [{txId, risk_score, communityId, ...}, ...],
                "edges": [{source, target}, ...],
                "node_count": int,
                "edge_count": int,
                "capped": bool
            }
        """
        # Enforce hard caps
        hops = min(hops, 2)
        max_nodes = min(max_nodes, 200)

        with self.session() as s:
            # Fetch ego-graph nodes within k hops
            node_result = s.run(
                """
                MATCH (center:Transaction {txId: $address})
                CALL apoc.path.subgraphNodes(center, {
                    maxLevel: $hops,
                    limit: $maxNodes
                })
                YIELD node
                RETURN node.txId          AS txId,
                       node.timeStep      AS timeStep,
                       node.class         AS txClass,
                       node.risk_score    AS risk_score,
                       node.predicted_label AS predicted_label,
                       node.confidence    AS confidence,
                       node.communityId   AS communityId,
                       node.pageRank      AS pageRank
                """,
                address=address,
                hops=hops,
                maxNodes=max_nodes,
            )
            nodes = [dict(r) for r in node_result]

            if not nodes:
                return {
                    "center": address,
                    "hops": hops,
                    "nodes": [],
                    "edges": [],
                    "node_count": 0,
                    "edge_count": 0,
                    "capped": False,
                }

            # Collect txIds in the subgraph for edge filtering
            node_ids = [n["txId"] for n in nodes]

            # Fetch edges within the subgraph only
            edge_result = s.run(
                """
                MATCH (a:Transaction)-[r:FLOWS_TO]->(b:Transaction)
                WHERE a.txId IN $nodeIds AND b.txId IN $nodeIds
                RETURN a.txId AS source, b.txId AS target
                """,
                nodeIds=node_ids,
            )
            edges = [dict(r) for r in edge_result]

        capped = len(nodes) >= max_nodes
        return {
            "center": address,
            "hops": hops,
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "capped": capped,
        }

    # ── Subgraph fallback (no APOC) ───────────────────────────────────────

    def get_subgraph_no_apoc(self, address: str, hops: int = 2, max_nodes: int = 200) -> dict:
        """
        Fallback subgraph query using variable-length paths instead of APOC.
        Use this if the APOC plugin is not installed.
        """
        hops = min(hops, 2)
        max_nodes = min(max_nodes, 200)

        with self.session() as s:
            node_result = s.run(
                """
                MATCH (center:Transaction {txId: $address})
                MATCH path = (center)-[:FLOWS_TO*0..""" + str(hops) + """]->(neighbor)
                WITH DISTINCT neighbor
                LIMIT $maxNodes
                RETURN neighbor.txId          AS txId,
                       neighbor.timeStep      AS timeStep,
                       neighbor.class         AS txClass,
                       neighbor.risk_score    AS risk_score,
                       neighbor.predicted_label AS predicted_label,
                       neighbor.confidence    AS confidence,
                       neighbor.communityId   AS communityId,
                       neighbor.pageRank      AS pageRank
                """,
                address=address,
                maxNodes=max_nodes,
            )
            nodes = [dict(r) for r in node_result]

            if not nodes:
                return {
                    "center": address, "hops": hops,
                    "nodes": [], "edges": [],
                    "node_count": 0, "edge_count": 0, "capped": False,
                }

            node_ids = [n["txId"] for n in nodes]
            edge_result = s.run(
                """
                MATCH (a:Transaction)-[r:FLOWS_TO]->(b:Transaction)
                WHERE a.txId IN $nodeIds AND b.txId IN $nodeIds
                RETURN a.txId AS source, b.txId AS target
                """,
                nodeIds=node_ids,
            )
            edges = [dict(r) for r in edge_result]

        return {
            "center": address, "hops": hops,
            "nodes": nodes, "edges": edges,
            "node_count": len(nodes), "edge_count": len(edges),
            "capped": len(nodes) >= max_nodes,
        }

    # ── Cluster Query ─────────────────────────────────────────────────────

    def get_cluster(self, cluster_id: int, limit: int = 100) -> dict:
        """
        Fetch all transactions in a community cluster.
        Returns cluster summary + member nodes.
        """
        with self.session() as s:
            result = s.run(
                """
                MATCH (t:Transaction {communityId: $clusterId})
                RETURN t.txId          AS txId,
                       t.timeStep      AS timeStep,
                       t.class         AS txClass,
                       t.risk_score    AS risk_score,
                       t.predicted_label AS predicted_label,
                       t.confidence    AS confidence,
                       t.pageRank      AS pageRank
                ORDER BY t.risk_score DESC
                LIMIT $limit
                """,
                clusterId=cluster_id,
                limit=limit,
            )
            members = [dict(r) for r in result]

            # Aggregate stats
            stats_result = s.run(
                """
                MATCH (t:Transaction {communityId: $clusterId})
                RETURN count(t) AS size,
                       avg(t.risk_score) AS avg_risk,
                       max(t.risk_score) AS max_risk,
                       min(t.risk_score) AS min_risk
                """,
                clusterId=cluster_id,
            )
            stats = dict(stats_result.single())

        return {
            "cluster_id": cluster_id,
            "size": stats.get("size", 0),
            "avg_risk_score": stats.get("avg_risk"),
            "max_risk_score": stats.get("max_risk"),
            "min_risk_score": stats.get("min_risk"),
            "members": members,
            "members_returned": len(members),
        }

    # ── Top Risky Clusters ────────────────────────────────────────────────

    def get_top_clusters(self, limit: int = 100) -> list:
        """
        Returns top clusters ranked by average risk score.
        Used by the Cluster Explorer dashboard tab.
        """
        with self.session() as s:
            result = s.run(
                """
                MATCH (t:Transaction)
                WHERE t.communityId IS NOT NULL
                RETURN t.communityId   AS cluster_id,
                       count(t)        AS size,
                       avg(t.risk_score) AS avg_risk
                ORDER BY avg_risk DESC
                LIMIT $limit
                """,
                limit=limit,
            )
            return [dict(r) for r in result]

    # ── Transaction Path ──────────────────────────────────────────────────

    def get_shortest_path(self, src_address: str, dst_address: str, max_depth: int = 10) -> dict:
        """
        Find shortest path between two transactions.
        Capped at max_depth hops; returns 'no path found' if exceeded.
        """
        max_depth = min(max_depth, 10)

        with self.session() as s:
            result = s.run(
                """
                MATCH (a:Transaction {txId: $src}), (b:Transaction {txId: $dst})
                MATCH p = shortestPath((a)-[:FLOWS_TO*..10]-(b))
                RETURN [n IN nodes(p) | n.txId] AS path_nodes,
                       length(p) AS path_length
                """,
                src=src_address,
                dst=dst_address,
            )
            record = result.single()

        if record is None:
            return {
                "source": src_address,
                "target": dst_address,
                "path_found": False,
                "path_nodes": [],
                "path_length": 0,
            }

        return {
            "source": src_address,
            "target": dst_address,
            "path_found": True,
            "path_nodes": record["path_nodes"],
            "path_length": record["path_length"],
        }


# Module-level singleton
neo4j_service = Neo4jService()
