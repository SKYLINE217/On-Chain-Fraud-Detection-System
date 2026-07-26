# src/serving/score_batch.py
"""
Nightly batch job: loads GNN checkpoint, runs inference on all 203,769 nodes,
writes risk_score / predicted_label / confidence / embedding back to Neo4j.
Flushes Redis after write.

See person_a_stages.md §4.1 for full reference.

Usage:
    python -m src.serving.score_batch --flush-redis

Compliance Disclaimer: This system is a research and portfolio
demonstration only. Not a certified AML/CFT compliance tool.
"""
import argparse
import json
import logging
import os
import time

import torch
import torch.nn.functional as F
from neo4j import GraphDatabase
from redis import Redis
from dotenv import load_dotenv

from src.features.build_pyg import load_pyg_data
from src.models.graphsage import GraphSAGE
from src.models.gat import GAT

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_CHECKPOINT = os.environ.get("MODEL_CHECKPOINT", "checkpoints/best_model.pt")
MODEL_CONFIG     = os.environ.get("MODEL_CONFIG", "checkpoints/model_config.json")
BATCH_SIZE = 1000  # Neo4j write batch size


def load_model_from_config(checkpoint_path: str, config_path: str) -> tuple[torch.nn.Module, dict]:
    """Load model using model_config.json schema (see ml_models.md §9.3)."""
    with open(config_path) as f:
        config = json.load(f)

    model_type = config["model_type"]
    if model_type.lower() == "graphsage":
        model = GraphSAGE(
            in_channels=config["in_channels"],
            hidden_channels=config["hidden_channels"],
            out_channels=config["out_channels"],
            num_layers=config["num_layers"],
            dropout=config.get("dropout", 0.0),
            aggr=config.get("aggr", "mean"),
        )
    elif model_type.lower() == "gat":
        model = GAT(
            in_channels=config["in_channels"],
            hidden_channels=config["hidden_channels"],
            out_channels=config["out_channels"],
            dropout=config.get("dropout", 0.0),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


def write_scores_to_neo4j(driver, records: list[dict]):
    """Batch UNWIND MATCH + SET for scored node properties."""
    with driver.session() as session:
        session.run(
            """
            UNWIND $records AS row
            MATCH (t:Transaction {txId: row.txId})
            SET t.risk_score      = row.risk_score,
                t.predicted_label = row.predicted_label,
                t.confidence      = row.confidence,
                t.embedding       = row.embedding
            """,
            records=records
        )


def run_batch_scoring(flush_redis: bool = True):
    start = time.time()
    logger.info("Loading model + config...")
    model, config = load_model_from_config(MODEL_CHECKPOINT, MODEL_CONFIG)

    logger.info("Loading PyG data...")
    data = load_pyg_data()

    logger.info("Running inference on all 203,769 nodes...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device))
        probs  = F.softmax(logits, dim=1).cpu()
        embeddings = model.get_embedding(
            data.x.to(device), data.edge_index.to(device)
        ).cpu()

    risk_scores     = probs[:, 1].numpy()
    confidence      = probs.max(dim=1).values.numpy()
    predicted_class = probs.argmax(dim=1).numpy()
    label_map_inv   = {1: "illicit", 0: "licit"}

    # Build records — Read txIds from parquet (preserved ordering)
    import pandas as pd
    df = pd.read_parquet("data/processed/features_combined.parquet", columns=["txId"])
    txids = df["txId"].tolist()

    records = [
        {
            "txId":            txids[i],
            "risk_score":      float(risk_scores[i]),
            "predicted_label": label_map_inv.get(int(predicted_class[i]), "unknown"),
            "confidence":      float(confidence[i]),
            "embedding":       embeddings[i].tolist(),
        }
        for i in range(len(txids))
    ]

    logger.info("Writing scores to Neo4j...")
    driver = GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.environ.get("NEO4J_USER", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "changeme_in_prod"),
        )
    )
    try:
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i : i + BATCH_SIZE]
            write_scores_to_neo4j(driver, batch)
            if i % 10000 == 0:
                logger.info(f"  written: {i}/{len(records)}")
    finally:
        driver.close()

    if flush_redis:
        logger.info("Flushing Redis cache...")
        redis = Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))
        redis.flushdb()
        redis.close()
        logger.info("Redis flushed.")

    elapsed = time.time() - start
    logger.info(f"Batch scoring complete. Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flush-redis", action="store_true", default=True)
    args = parser.parse_args()
    run_batch_scoring(flush_redis=args.flush_redis)
