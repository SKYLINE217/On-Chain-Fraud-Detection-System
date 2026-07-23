# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

"""
Batch scoring job — runs GNN inference on all nodes and writes
risk_score, predicted_label, confidence, and embedding back to Neo4j.

Schedule: nightly cadence for demo (cron or Airflow DAG).

Usage:
    python -m src.serving.score_batch --checkpoint checkpoints/best_model.pt

Neo4j write-back (Contract 3 — blend.md):
    UNWIND $batch AS row
    MATCH (t:Transaction {txId: row.txId})
    SET t.risk_score      = row.risk_score,
        t.predicted_label = row.predicted_label,
        t.confidence      = row.confidence,
        t.embedding       = row.embedding

Post-scoring: flush Redis to avoid stale cached scores.
"""

import os
import json
import time
import argparse
import logging
from pathlib import Path

import torch
import numpy as np
import pandas as pd
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
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))


def load_model(checkpoint_path: str, config_path: str = None):
    """
    Load a trained GNN model from checkpoint + config.

    Uses model_config.json (Contract 3) to instantiate the model
    without hunting for hyperparameters.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        config_path: Path to model_config.json. If None, looks in same dir.

    Returns:
        Tuple of (model in eval mode, model_config dict).
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Load model config (Contract 3 — blend.md)
    if config_path is None:
        config_path = Path(checkpoint_path).parent / "model_config.json"

    if Path(config_path).exists():
        with open(config_path) as f:
            model_config = json.load(f)
        logger.info("Model config loaded from %s", config_path)
    else:
        # Fall back to checkpoint's embedded config
        model_config = checkpoint.get("config", {})
        logger.warning("model_config.json not found — using embedded config")

    model_type = model_config.get("model_type", "GraphSAGE").lower()

    if model_type == "graphsage":
        from src.models.graphsage import GraphSAGE
        model = GraphSAGE(
            in_channels=model_config.get("in_channels", 166),
            hidden_channels=model_config.get("hidden_channels", 128),
            out_channels=model_config.get("out_channels", 2),
            num_layers=model_config.get("num_layers", 3),
            dropout=model_config.get("dropout", 0.3),
        )
    elif model_type == "gat":
        from src.models.gat import GAT
        model = GAT(
            in_channels=model_config.get("in_channels", 166),
            hidden_channels=model_config.get("hidden_channels", 128),
            out_channels=model_config.get("out_channels", 2),
            heads=model_config.get("heads", 4),
            dropout=model_config.get("dropout", 0.3),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Model loaded from %s (%s)", checkpoint_path, model_type)
    return model, model_config


def load_txid_mapping(parquet_path: str = "data/processed/features_combined.parquet") -> list:
    """
    Load ordered txId list from the parquet file.
    This ensures the node index → txId mapping is correct.

    Returns:
        List of txIds in the same order as the feature matrix.
    """
    df = pd.read_parquet(parquet_path, columns=["txId"])
    txid_list = [str(tx_id) for tx_id in df["txId"].values]
    logger.info("Loaded %d txIds from %s", len(txid_list), parquet_path)
    return txid_list


def run_inference(model, data, model_config: dict = None) -> dict:
    """
    Run inference on all nodes in the PyG Data object.

    Uses model.get_embeddings() for proper hidden-layer embeddings
    (not raw logits) as specified by Contract 3.

    Returns:
        Dict mapping node index → {risk_score, predicted_label, confidence, embedding}
    """
    logger.info("Running inference on %d nodes...", data.num_nodes)
    t0 = time.time()

    hidden_dim = (model_config or {}).get("hidden_channels", 128)

    with torch.no_grad():
        # Get logits for predictions
        out = model(data.x, data.edge_index)
        probs = torch.softmax(out, dim=-1)

        # Risk score = probability of illicit class (index 1)
        risk_scores = probs[:, 1].numpy()

        # Predicted label: argmax
        predictions = out.argmax(dim=-1).numpy()

        # Confidence: max probability
        confidence = probs.max(dim=-1).values.numpy()

        # Get proper embeddings from hidden layer
        # (not logits — Contract 3 specifies hidden_dim length)
        try:
            embeddings = model.get_embeddings(data.x, data.edge_index).numpy()
        except Exception as e:
            logger.warning("get_embeddings() failed (%s) — using logits", e)
            embeddings = out.numpy()

    elapsed = time.time() - t0
    logger.info("Inference complete in %.1fs", elapsed)

    # Label encoding (Contract 3)
    label_map = {0: "licit", 1: "illicit"}

    results = {}
    for i in range(data.num_nodes):
        # For unknown nodes (y == -1), still assign predicted label
        pred_label = label_map.get(int(predictions[i]), "unknown")
        if hasattr(data, 'y') and data.y[i].item() == -1:
            pred_label = pred_label  # Still predict, but the ground truth is unknown

        results[i] = {
            "risk_score": float(np.clip(risk_scores[i], 0.0, 1.0)),
            "predicted_label": pred_label,
            "confidence": float(np.clip(confidence[i], 0.0, 1.0)),
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

    Uses the exact Cypher pattern from Contract 3 (blend.md):
        UNWIND $batch AS row
        MATCH (t:Transaction {txId: row.txId})
        SET t.risk_score      = row.risk_score,
            t.predicted_label = row.predicted_label,
            t.confidence      = row.confidence,
            t.embedding       = row.embedding

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
                    "txId": txid_list[idx],
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
                batch_num = start // batch_size
                if batch_num % 20 == 0:
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


def flush_redis():
    """
    Flush Redis cache after batch scoring to prevent stale cached scores.
    Per blend.md §6: "Flush Redis after batch job."
    """
    try:
        import redis
        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, socket_connect_timeout=2)
        client.flushdb()
        logger.info("[✓] Redis cache flushed after batch scoring")
    except Exception as e:
        logger.warning("Could not flush Redis: %s", e)


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
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pt",
        help="Path to model checkpoint .pt",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to model_config.json (default: same dir as checkpoint)",
    )
    parser.add_argument(
        "--graph", type=str, default="data/processed/pyg_data.pt",
        help="Path to PyG Data object",
    )
    parser.add_argument(
        "--parquet", type=str, default="data/processed/features_combined.parquet",
        help="Path to features_combined.parquet (for txId mapping)",
    )
    parser.add_argument("--verify", action="store_true", help="Run writeback verification")
    parser.add_argument("--flush-redis", action="store_true", help="Flush Redis after scoring")
    parser.add_argument("--batch-size", type=int, default=5000, help="Neo4j write batch size")
    args = parser.parse_args()

    from src.features.build_pyg import load_pyg_graph

    # Load model with config (Contract 3)
    model, model_config = load_model(args.checkpoint, args.config)

    # Load PyG graph
    data = load_pyg_graph(args.graph)

    # Load txId mapping from parquet (NOT placeholder indices)
    txid_list = load_txid_mapping(args.parquet)

    # Run inference
    results = run_inference(model, data, model_config)

    # Write to Neo4j
    write_scores_to_neo4j(results, txid_list, batch_size=args.batch_size)

    # Post-scoring cleanup
    if args.flush_redis:
        flush_redis()

    if args.verify:
        verify_writeback()
