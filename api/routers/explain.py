# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

"""
Explain router — POST /explain/{address}

Provides per-node explainability via GNNExplainer + SHAP/TreeExplainer.

Fixes applied:
  BUG-04: torch.load now uses weights_only=True for model checkpoints
  BUG-07: _init_explain_context() is now async with asyncio.Lock to prevent race conditions
  BUG-24: Feature names read from parquet schema only (no full file read)

Contract 5 (blend.md) compliant response schema:
{
    "address": "string",
    "shap_top_features": [
        {"feature_name": "string", "feature_value": 0.0, "shap_value": 0.0}
    ],
    "subgraph_explanation": {
        "important_nodes": [{"node_id": "string", "importance_score": 0.0}],
        "important_edges": [{"src": "string", "dst": "string", "importance_score": 0.0}]
    },
    "rationale": "Flagged due to: ...",
    "explanation_model": "GNNExplainer + SHAP/TreeExplainer",
    "latency_warning": "This endpoint may take 5-15 seconds"
}

Latency budget: 5–15 seconds (GNNExplainer is slow by design).
No caching — explanations are instance-specific.
"""

import asyncio
import os
import time
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/explain", tags=["Explainability"])
logger = logging.getLogger(__name__)


# ── Response Models (Contract 5 compliant) ───────────────────────────────

class SHAPFeature(BaseModel):
    feature_name: str
    feature_value: float
    shap_value: float


class ImportantNode(BaseModel):
    node_id: str
    importance_score: float


class ImportantEdge(BaseModel):
    src: str
    dst: str
    importance_score: float


class SubgraphExplanation(BaseModel):
    important_nodes: list[ImportantNode]
    important_edges: list[ImportantEdge]


class ExplainResponse(BaseModel):
    address: str
    shap_top_features: list[SHAPFeature]
    subgraph_explanation: SubgraphExplanation
    rationale: str
    explanation_model: str
    latency_warning: str


# ── Lazy-loaded explain context ──────────────────────────────────────────

_explain_context = {
    "model": None,
    "data": None,
    "feature_names": None,
    "txid_to_idx": None,
    "gnn_explainer": None,
    "shap_explainer": None,
    "initialized": False,
}

# BUG-07: asyncio Lock prevents concurrent initialization on first request
_explain_lock = asyncio.Lock()


def _get_feature_names_from_parquet():
    """
    Load feature column names from the parquet schema only.
    BUG-24 Fix: Uses pyarrow schema read (metadata only, ~100x faster than full file read).
    """
    try:
        parquet_path = Path("data/processed/features_combined.parquet")
        if parquet_path.exists():
            try:
                # BUG-24: Read only the file footer (schema metadata) — no data loaded
                import pyarrow.parquet as pq
                schema = pq.read_schema(parquet_path)
                all_cols = schema.names
            except ImportError:
                # Fallback: read just txId column to get schema, then read columns attr
                import pandas as pd
                df = pd.read_parquet(parquet_path, columns=["txId"])
                import pyarrow.parquet as pq
                # Still try pyarrow but accept pandas fallback
                schema = pq.read_schema(parquet_path)
                all_cols = schema.names

            meta_cols = {"txId", "timeStep", "class"}
            feature_cols = [c for c in all_cols if c not in meta_cols]
            return feature_cols
    except Exception as e:
        logger.warning("Could not read parquet for feature names: %s", e)
    # Fallback
    return [f"f{i}" for i in range(1, 167)] + [
        "tx_freq", "amount_mean", "amount_skew", "address_age",
        "clustering_coeff", "burst_score", "pageRank", "communityId",
    ]


def _get_txid_mapping():
    """Build txId → node index mapping from parquet."""
    try:
        import pandas as pd
        parquet_path = Path("data/processed/features_combined.parquet")
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path, columns=["txId"])
            return {str(tx_id): idx for idx, tx_id in enumerate(df["txId"].values)}
    except Exception as e:
        logger.warning("Could not build txId mapping: %s", e)
    return {}


async def _init_explain_context():
    """
    Initialize model, data, and explainers on first request.
    BUG-07 Fix: Protected by asyncio.Lock to prevent race conditions under concurrent load.
    """
    # BUG-07: Lock prevents multiple concurrent initializations
    async with _explain_lock:
        if _explain_context["initialized"]:
            return

        logger.info("Initializing explain context (first request)...")

        # Load feature names and txId mapping
        _explain_context["feature_names"] = _get_feature_names_from_parquet()
        _explain_context["txid_to_idx"] = _get_txid_mapping()

        # Try to load model and PyG data
        try:
            import torch
            import torch.serialization
            from src.explain.gnn_explainer import GNNExplainerWrapper
            from src.explain.shap_explainer import SHAPExplainer

            # Load PyG data
            # BUG-04 Fix: Use safe_globals for PyG Data objects
            pyg_path = Path("data/processed/pyg_data.pt")
            if pyg_path.exists():
                try:
                    from torch_geometric.data import Data as PyGData
                    with torch.serialization.safe_globals([PyGData]):
                        data = torch.load(pyg_path, map_location="cpu", weights_only=True)
                except Exception:
                    # Older PyTorch without safe_globals support — use weights_only=False
                    # with a warning
                    logger.warning(
                        "safe_globals not available for PyG Data — falling back to "
                        "weights_only=False. Ensure pyg_data.pt is from a trusted source."
                    )
                    data = torch.load(pyg_path, map_location="cpu", weights_only=False)
                _explain_context["data"] = data
                logger.info("PyG data loaded: %d nodes", data.num_nodes)
            else:
                logger.warning("PyG data not found at %s", pyg_path)

            # Load best model checkpoint
            checkpoint_path = Path("checkpoints/best_model.pt")
            config_path = Path("checkpoints/model_config.json")

            if checkpoint_path.exists() and config_path.exists():
                with open(config_path) as f:
                    model_config = json.load(f)

                model_type = model_config.get("model_type", "GraphSAGE")
                if model_type == "GraphSAGE":
                    from src.models.graphsage import GraphSAGE
                    model = GraphSAGE(
                        in_channels=model_config.get("in_channels", 166),
                        hidden_channels=model_config.get("hidden_channels", 128),
                        out_channels=model_config.get("out_channels", 2),
                        num_layers=model_config.get("num_layers", 3),
                        dropout=model_config.get("dropout", 0.3),
                    )
                else:
                    from src.models.gat import GAT
                    model = GAT(
                        in_channels=model_config.get("in_channels", 166),
                        hidden_channels=model_config.get("hidden_channels", 128),
                        out_channels=model_config.get("out_channels", 2),
                        heads=model_config.get("heads", 4),
                        dropout=model_config.get("dropout", 0.3),
                    )

                # BUG-04 Fix: Use weights_only=True for model state dict
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                _explain_context["model"] = model
                logger.info("Model loaded: %s", model_type)

                # Initialize GNNExplainer
                if _explain_context["data"] is not None:
                    _explain_context["gnn_explainer"] = GNNExplainerWrapper(
                        model=model,
                        data=_explain_context["data"],
                        feature_names=_explain_context["feature_names"],
                        explainer_epochs=200,
                    )
            else:
                logger.warning(
                    "Model checkpoint not found at %s / %s — "
                    "explain will use synthetic explanations",
                    checkpoint_path, config_path,
                )

            # Initialize SHAP explainer
            _explain_context["shap_explainer"] = SHAPExplainer(
                feature_names=_explain_context["feature_names"]
            )

        except ImportError as e:
            logger.warning("ML dependencies not available: %s", e)
        except Exception as e:
            logger.error("Failed to initialize explain context: %s", e)

        _explain_context["initialized"] = True
        logger.info("Explain context initialized")


def _generate_synthetic_explanation(address: str) -> dict:
    """
    Generate a synthetic but structurally valid explanation when
    model/data are not available. Used for development and testing.
    """
    from src.explain.rationale import generate_rationale

    feature_names = _explain_context.get("feature_names") or [f"f{i}" for i in range(1, 167)]

    # Deterministic-ish synthetic values based on address hash
    seed = hash(address) % 10000
    rng = np.random.RandomState(seed)

    risk_score = rng.beta(2, 5)  # Skew toward low risk
    n_features = min(10, len(feature_names))

    shap_top_features = [
        {
            "feature_name": feature_names[i % len(feature_names)],
            "feature_value": float(rng.randn()),
            "shap_value": float(rng.randn() * 0.1),
        }
        for i in rng.choice(len(feature_names), size=n_features, replace=False)
    ]

    # Sort by absolute SHAP value
    shap_top_features.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    important_nodes = [
        {"node_id": str(rng.randint(0, 200000)), "importance_score": float(rng.random())}
        for _ in range(5)
    ]

    important_edges = [
        {
            "src": str(rng.randint(0, 200000)),
            "dst": str(rng.randint(0, 200000)),
            "importance_score": float(rng.random()),
        }
        for _ in range(5)
    ]

    predicted_label = "illicit" if risk_score > 0.5 else "licit"

    rationale = generate_rationale(
        shap_top_features=shap_top_features,
        important_edges=important_edges,
        important_nodes=important_nodes,
        risk_score=risk_score,
        predicted_label=predicted_label,
    )

    return {
        "address": address,
        "shap_top_features": shap_top_features,
        "subgraph_explanation": {
            "important_nodes": important_nodes,
            "important_edges": important_edges,
        },
        "rationale": rationale,
        "explanation_model": "GNNExplainer + SHAP/TreeExplainer (synthetic fallback)",
        "latency_warning": "This endpoint may take 5-15 seconds",
    }


# ── Endpoint ─────────────────────────────────────────────────────────────

@router.post(
    "/{address}",
    response_model=ExplainResponse,
    summary="Explain wallet risk prediction",
    description=(
        "Generate an explanation for a wallet's risk prediction using "
        "GNNExplainer + SHAP. Returns top features, important subgraph "
        "elements, and a human-readable rationale. "
        "Expected latency: 5–15 seconds (GNNExplainer is slow by design)."
    ),
)
async def explain_wallet(address: str):
    """
    POST /explain/{address}
    Owner: Person B | Latency budget: 5–15s
    """
    t0 = time.perf_counter()

    # BUG-07: Async-safe lazy initialization
    await _init_explain_context()

    model = _explain_context["model"]
    data = _explain_context["data"]
    txid_to_idx = _explain_context["txid_to_idx"]
    gnn_explainer = _explain_context["gnn_explainer"]
    shap_explainer = _explain_context["shap_explainer"]
    feature_names = _explain_context["feature_names"]

    # If model/data not available, return synthetic explanation
    if model is None or data is None:
        logger.warning(
            "Model or data not loaded — returning synthetic explanation for %s",
            address,
        )
        result = _generate_synthetic_explanation(address)
        return result

    # Resolve txId to node index
    node_idx = txid_to_idx.get(str(address))
    if node_idx is None:
        raise HTTPException(
            status_code=404,
            detail=f"Address {address} not found in the transaction graph.",
        )

    # ── GNNExplainer ─────────────────────────────────────────────────
    gnn_result = {}
    if gnn_explainer is not None:
        try:
            gnn_result = gnn_explainer.explain_node(
                node_idx=node_idx,
                top_k_features=10,
                top_k_edges=10,
                top_k_nodes=10,
            )
        except Exception as e:
            logger.error("GNNExplainer failed for %s: %s", address, e)

    # ── SHAP ─────────────────────────────────────────────────────────
    import torch
    features = data.x[node_idx].detach().cpu().numpy()
    shap_top_features = shap_explainer.explain_node(features, top_k=10)

    # ── Risk Score & Label ───────────────────────────────────────────
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.softmax(out[node_idx], dim=0)
        risk_score = float(probs[1].item())
        predicted_label = "illicit" if risk_score > 0.5 else "licit"

    # ── Subgraph Explanation ─────────────────────────────────────────
    important_nodes = gnn_result.get("important_nodes", [])
    important_edges = gnn_result.get("important_edges", [])

    # ── Rationale ────────────────────────────────────────────────────
    from src.explain.rationale import generate_rationale

    rationale = generate_rationale(
        shap_top_features=shap_top_features,
        important_edges=important_edges,
        important_nodes=important_nodes,
        risk_score=risk_score,
        predicted_label=predicted_label,
    )

    latency = (time.perf_counter() - t0) * 1000
    logger.info(
        "Explanation for %s: risk=%.4f, label=%s (%.1fms)",
        address, risk_score, predicted_label, latency,
    )

    return {
        "address": address,
        "shap_top_features": shap_top_features,
        "subgraph_explanation": {
            "important_nodes": important_nodes,
            "important_edges": important_edges,
        },
        "rationale": rationale,
        "explanation_model": "GNNExplainer + SHAP/TreeExplainer",
        "latency_warning": "This endpoint may take 5-15 seconds",
    }
