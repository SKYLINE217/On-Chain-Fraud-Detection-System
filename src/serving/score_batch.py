# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

"""
Batch scoring job — runs GNN inference on all nodes and writes
risk_score, predicted_label, confidence, and embedding back to Neo4j.

Schedule: nightly cadence for demo (cron or Airflow DAG).

Usage:
    python -m src.serving.score_batch --checkpoint checkpoints/model.pt

Neo4j write-back:
    MATCH (t:Transaction {txId: $txId})
    SET t.risk_score     = $risk_score,
        t.predicted_label = $label,
        t.confidence      = $confidence,
        t.embedding       = $embedding
"""

import os
import time
import argparse
import logging
import torch
import numpy as np
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


def load_model(checkpoint_path: str, model_class: str = "graphsage"):
    """
    Load a trained GNN model from checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        model_class: 'graphsage' or 'gat'.

    Returns:
        Loaded model in eval mode.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if model_class == "graphsage":
        from src.models.graphsage import GraphSAGE
        model = GraphSAGE(
            in_channels=checkpoint.get("in_channels", 166),
            hidden_channels=checkpoint.get("hidden_channels", 128),
            out_channels=checkpoint.get("out_channels", 2),
        )
    elif model_class == "gat":
        from src.models.gat import GAT
        model = GAT(
            in_channels=checkpoint.get("in_channels", 166),
            hidden_channels=checkpoint.get("hidden_channels", 128),
            out_channels=checkpoint.get("out_channels", 2),
        )
    else:
        raise ValueError(f"Unknown model class: {model_class}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Model loaded from %s (%s)", checkpoint_path, model_class)
    return model


def run_inference(model, data) -> dict:
    """
    Run inference on all nodes in the PyG Data object.

    Returns:
        Dict mapping txId index → {risk_score, label, confidence, embedding}
    """
    logger.info("Running inference on %d nodes...", data.num_nodes)
    t0 = time.time()

    with torch.no_grad():
        # Get logits from the last layer before softmax
        out = model(data.x, data.edge_index)
        probs = torch.exp(out)  # convert log_softmax to probabilities

        # Risk score = probability of illicit class (index 1)
        risk_scores = probs[:, 1].numpy()

        # Predicted label: argmax
        predictions = out.argmax(dim=-1).numpy()

        # Confidence: max probability
        confidence = probs.max(dim=-1).values.numpy()

        # Get embeddings from the hidden layer (hook or re-forward)
        # For now, use the output logits as a compact embedding
        embeddings = out.numpy()

    elapsed = time.time() - t0
    logger.info("Inference complete in %.1fs", elapsed)

    results = {}
    for i in range(data.num_nodes):
        results[i] = {
            "risk_score": float(risk_scores[i]),
            "predicted_label": "illicit" if predictions[i] == 1 else "licit",
            "confidence": float(confidence[i]),
            "embedding": embeddings[i].tolist(),
        }

    return results


def write_scores_to_neo4j(
    results: dict,
    txid_list: list,
    batch_size: int = 5000,
):
    """
    Batch write-back of inference results to Neo4j node properties.

    Args:
        results: Dict from run_inference (index → scores).
        txid_list: Ordered list of txIds matching the node indices.
        batch_size: Number of nodes per UNWIND batch.
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        records = []
        for idx, scores in results.items():
            if idx < len(txid_list):
                records.append({
                    "txId": str(txid_list[idx]),
                    "risk_score": scores["risk_score"],
                    "predicted_label": scores["predicted_label"],
                    "confidence": scores["confidence"],
                    "embedding": scores["embedding"],
                })

        logger.info("Writing %d scored nodes to Neo4j...", len(records))
        t0 = time.time()

        with driver.session() as session:
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                session.run(
                    """
                    UNWIND $batch AS row
                    MATCH (t:Transaction {txId: row.txId})
                    SET t.risk_score       = row.risk_score,
                        t.predicted_label  = row.predicted_label,
                        t.confidence       = row.confidence,
                        t.embedding        = row.embedding
                    """,
                    batch=batch,
                )
                if (start // batch_size) % 20 == 0:
                    elapsed = time.time() - t0
                    logger.info(
                        "    Written %d/%d (%.0fs)",
                        min(start + batch_size, len(records)),
                        len(records),
                        elapsed,
                    )

        logger.info("[✓] All scores written to Neo4j in %.0fs", time.time() - t0)

    finally:
        driver.close()


def verify_writeback(sample_size: int = 5):
    """Spot-check that scores were written correctly."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (t:Transaction)
                WHERE t.risk_score IS NOT NULL
                RETURN t.txId AS txId,
                       t.risk_score AS risk_score,
                       t.predicted_label AS label,
                       t.confidence AS confidence
                LIMIT $n
                """,
                n=sample_size,
            )
            logger.info("── Writeback verification (sample) ──")
            for record in result:
                logger.info(
                    "    %s → risk=%.4f, label=%s, conf=%.4f",
                    record["txId"],
                    record["risk_score"] or 0,
                    record["label"],
                    record["confidence"] or 0,
                )
    finally:
        driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch scoring job")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt")
    parser.add_argument("--model", type=str, default="graphsage", choices=["graphsage", "gat"])
    parser.add_argument("--graph", type=str, default="data/processed/pyg_graph.pt", help="Path to PyG Data object")
    parser.add_argument("--verify", action="store_true", help="Run writeback verification after scoring")
    args = parser.parse_args()

    from src.features.build_pyg import load_pyg_graph

    model = load_model(args.checkpoint, model_class=args.model)
    data = load_pyg_graph(args.graph)

    results = run_inference(model, data)

    # Extract txId order (need the original DataFrame for mapping)
    # For now, use integer indices as txIds — replace with actual mapping in production
    import pandas as pd
    txid_list = list(range(data.num_nodes))  # placeholder — wire to actual txIds

    write_scores_to_neo4j(results, txid_list)

    if args.verify:
        verify_writeback()
