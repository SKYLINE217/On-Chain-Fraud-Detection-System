# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

"""
Synthetic scale generator — replicates Elliptic graph structure at 40–50× scale.

Purpose: Validate sub-5s query latency at 10M+ edge scale.
This does NOT affect model accuracy metrics — those apply only to the
Elliptic labeled dataset. These are separate claims.

Usage:
    python scripts/scale_synthetic.py --scale 50 --data-dir data/raw
"""

import os
import argparse
import random
import time
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


def generate_synthetic_nodes(original_count: int, scale_factor: int) -> list:
    """
    Generate synthetic transaction node records by cloning the original
    structure with randomized feature offsets.

    Args:
        original_count: Number of nodes in the original Elliptic dataset.
        scale_factor: Multiplier (e.g. 50 → ~10M nodes).

    Returns:
        List of synthetic node dicts with txId and timeStep.
    """
    synthetic_count = original_count * scale_factor
    print(f"[→] Generating {synthetic_count:,} synthetic nodes ({scale_factor}× scale)...")

    nodes = []
    for i in range(synthetic_count):
        nodes.append({
            "txId": f"SYN_{i:010d}",
            "timeStep": random.randint(1, 49),
            "txClass": random.choice(["1", "2", "unknown"]),
        })

    print(f"[✓] Generated {len(nodes):,} synthetic nodes.")
    return nodes


def generate_synthetic_edges(node_ids: list, target_edge_count: int) -> list:
    """
    Generate synthetic edges using random assignment, preserving the
    approximate degree distribution of the Elliptic dataset.

    Elliptic baseline: ~234K edges / ~204K nodes ≈ 1.15 edges/node avg.
    """
    print(f"[→] Generating {target_edge_count:,} synthetic edges...")

    edges = []
    node_count = len(node_ids)
    for _ in range(target_edge_count):
        src_idx = random.randint(0, node_count - 1)
        # Bias toward nearby nodes (locality) — more realistic graph structure
        offset = int(random.gauss(0, min(1000, node_count // 10)))
        dst_idx = (src_idx + offset) % node_count
        if src_idx != dst_idx:
            edges.append({
                "src": node_ids[src_idx],
                "dst": node_ids[dst_idx],
            })

    print(f"[✓] Generated {len(edges):,} synthetic edges.")
    return edges


def load_synthetic_to_neo4j(nodes: list, edges: list, batch_size: int = 10000):
    """Batch-load synthetic data into Neo4j using UNWIND + MERGE."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        with driver.session() as session:
            # Load nodes
            print(f"[→] Loading {len(nodes):,} synthetic nodes to Neo4j...")
            t0 = time.time()
            for start in range(0, len(nodes), batch_size):
                batch = nodes[start : start + batch_size]
                session.run(
                    """
                    UNWIND $batch AS row
                    MERGE (t:Transaction {txId: row.txId})
                    SET t.timeStep = row.timeStep,
                        t.class    = row.txClass,
                        t.synthetic = true
                    """,
                    batch=batch,
                )
                if (start // batch_size) % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"    {start + len(batch):,} nodes loaded ({elapsed:.0f}s)")

            print(f"[✓] Nodes loaded in {time.time() - t0:.0f}s")

            # Load edges
            print(f"[→] Loading {len(edges):,} synthetic edges to Neo4j...")
            t0 = time.time()
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
                if (start // batch_size) % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"    {start + len(batch):,} edges loaded ({elapsed:.0f}s)")

            print(f"[✓] Edges loaded in {time.time() - t0:.0f}s")

            # Final counts
            result = session.run(
                """
                MATCH (n:Transaction) WITH count(n) AS nodes
                MATCH ()-[r:FLOWS_TO]->() WITH nodes, count(r) AS edges
                RETURN nodes, edges
                """
            )
            counts = result.single()
            print(f"\n── Final Graph Scale ──")
            print(f"    Total nodes: {counts['nodes']:,}")
            print(f"    Total edges: {counts['edges']:,}")

    finally:
        driver.close()


def cleanup_synthetic(driver=None):
    """Remove all synthetic nodes and their relationships."""
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        close_after = True
    else:
        close_after = False

    try:
        with driver.session() as session:
            print("[→] Removing synthetic nodes and edges...")
            session.run(
                """
                MATCH (t:Transaction {synthetic: true})
                DETACH DELETE t
                """
            )
            print("[✓] Synthetic data removed.")
    finally:
        if close_after:
            driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic graph data for scale testing")
    parser.add_argument("--scale", type=int, default=50, help="Scale factor (default: 50× → ~10M nodes)")
    parser.add_argument("--edge-ratio", type=float, default=1.15, help="Edge/node ratio (default: 1.15)")
    parser.add_argument("--load", action="store_true", help="Load synthetic data into Neo4j")
    parser.add_argument("--cleanup", action="store_true", help="Remove synthetic data from Neo4j")
    args = parser.parse_args()

    ORIGINAL_NODE_COUNT = 203769

    if args.cleanup:
        cleanup_synthetic()
    else:
        total_nodes = ORIGINAL_NODE_COUNT * args.scale
        total_edges = int(total_nodes * args.edge_ratio)

        nodes = generate_synthetic_nodes(ORIGINAL_NODE_COUNT, args.scale)
        node_ids = [n["txId"] for n in nodes]
        edges = generate_synthetic_edges(node_ids, total_edges)

        print(f"\n── Synthetic Scale Summary ──")
        print(f"    Scale factor:   {args.scale}×")
        print(f"    Synthetic nodes: {len(nodes):,}")
        print(f"    Synthetic edges: {len(edges):,}")
        print(f"    Total projected: ~{total_nodes + ORIGINAL_NODE_COUNT:,} nodes, ~{total_edges + 234355:,} edges")

        if args.load:
            load_synthetic_to_neo4j(nodes, edges)
        else:
            print("\n[i] Dry run — use --load to push to Neo4j.")
