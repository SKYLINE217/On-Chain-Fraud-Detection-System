
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Path
from typing import Annotated

try:
    from neo4j import AsyncDriver
except ImportError:
    AsyncDriver = type("AsyncDriver", (), {})

import torch
import time
import os
import joblib

try:
    from api.deps import get_neo4j_driver
    from api.middleware.auth import verify_api_key
    from api.models.responses import ExplainResponse
except ImportError:

    async def get_neo4j_driver(): yield None
    async def verify_api_key(): pass

    from pydantic import BaseModel
    class ExplainResponse(BaseModel):
        pass

from src.explain.gnn_explainer import build_gnn_explainer, explain_node
from src.explain.shap_explainer import build_kernel_explainer, compute_shap_for_node
from src.explain.rationale import generate_rationale
from src.features.build_pyg import load_pyg_data, FEATURE_COLS
from src.serving.score_batch import load_model_from_config

router = APIRouter(dependencies=[Depends(verify_api_key)])

_model = None
_data = None
_gnn_explainer = None
_shap_tree_explainer = None
_txid_to_idx: dict = {}

import asyncio
_load_lock = asyncio.Lock()
_model_version = "GNNExplainer (PyG) + SHAP TreeExplainer (XGBoost)"

async def _lazy_load():
    global _model, _data, _gnn_explainer, _shap_tree_explainer, _txid_to_idx, _model_version
    if _model is not None:
        return
    async with _load_lock:
        if _model is not None:
            return
        import pandas as pd

        checkpoint_path = os.environ.get("MODEL_CHECKPOINT", "checkpoints/best_model.pt")
        config_path = os.environ.get("MODEL_CONFIG", "checkpoints/model_config.json")

        _model, config = load_model_from_config(checkpoint_path, config_path)
        _model_version = f"{_model_version} (Model: {config.get('model_type', 'unknown')})"

        if os.environ.get("ENV") == "production":
            _data = load_pyg_data()
        else:
            try:
                _data = load_pyg_data()
            except FileNotFoundError:
                _data = load_pyg_data(parquet_path="mocks/person_a/mock_features_combined.parquet", mock_edges=True)

        try:
            df = pd.read_parquet("data/processed/features_combined.parquet", columns=["txId"])
        except FileNotFoundError:
            df = pd.read_parquet("mocks/person_a/mock_features_combined.parquet", columns=["txId"])

        _txid_to_idx = {str(txid): i for i, txid in enumerate(df["txId"])}
        _gnn_explainer = build_gnn_explainer(_model)

        try:
            xgb = joblib.load("checkpoints/xgb_baseline.pkl")
            import shap
            _shap_tree_explainer = shap.TreeExplainer(xgb)
        except FileNotFoundError:
            _shap_tree_explainer = None

AddressParam = Annotated[
    str,
    Path(pattern=r'^[a-zA-Z0-9_\-]{1,100}$', description="Transaction address")
]

@router.post("/{address}", response_model=ExplainResponse)
async def explain_address(address: AddressParam, driver: AsyncDriver = Depends(get_neo4j_driver)):
    """
    On-demand GNNExplainer + SHAP for a single address.
    Expected latency: 5–15s. Always includes latency_warning in response.
    NOT cached — instance-specific.
    """
    await _lazy_load()

    if address not in _txid_to_idx:
        raise HTTPException(status_code=404, detail=f"Address not found: {address}")

    node_idx = _txid_to_idx[address]
    t_start = time.time()

    gnn_result = explain_node(_gnn_explainer, _data, node_idx)

    import numpy as np
    x_node = _data.x[node_idx].numpy().reshape(1, -1)

    if _shap_tree_explainer is not None:

        x_node_xgb = x_node[:, :-1]
        shap_features = compute_shap_for_node(
            _shap_tree_explainer,
            x_node_xgb,
            feature_names=FEATURE_COLS[:-1],
            top_k=10,
        )
    else:
        shap_features = []

    neighbors = []
    if driver is not None:
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (t:Transaction {txId: $address})-[:FLOWS_TO]-(neighbor)
                RETURN neighbor.txId AS neighbor_id,
                       neighbor.predicted_label AS label
                LIMIT 10
                """,
                address=address
            )
            neighbors = [r.data() async for r in result]

    edge_idx = _data.edge_index
    incident_mask = (edge_idx[0] == node_idx) | (edge_idx[1] == node_idx)
    incident_edge_indices = incident_mask.nonzero(as_tuple=True)[0]
    top_k_edge_map = {
        i.item(): gnn_result["top_edge_values"][j]
        for j, i in enumerate(gnn_result["top_edge_indices"][:3])
        if i.item() in incident_edge_indices.tolist()
    }

    edge_mask_top = []
    for j, n in enumerate(neighbors[:3]):
        importance = 0.0
        neighbor_idx = _txid_to_idx.get(n["neighbor_id"])
        if neighbor_idx is not None:

            mask = ((edge_idx[0] == node_idx) & (edge_idx[1] == neighbor_idx)) |                   ((edge_idx[1] == node_idx) & (edge_idx[0] == neighbor_idx))
            matched = mask.nonzero(as_tuple=True)[0]
            if len(matched) > 0:

                try:

                    idx_in_top = (gnn_result["top_edge_indices"] == matched[0]).nonzero(as_tuple=True)[0]
                    if len(idx_in_top) > 0:
                        importance = gnn_result["top_edge_values"][idx_in_top[0].item()].item()
                except Exception:
                    pass
        edge_mask_top.append((n["neighbor_id"], importance, n.get("label", "unknown")))

    import torch.nn.functional as F
    with torch.no_grad():
        logits = _model(_data.x, _data.edge_index)
        probs  = F.softmax(logits[node_idx], dim=0)
    predicted_label = "illicit" if probs[1] > 0.5 else "licit"

    rationale = generate_rationale(shap_features, edge_mask_top, predicted_label)

    elapsed = time.time() - t_start

    return {
        "address": address,
        "shap_top_features": shap_features,
        "subgraph_explanation": {
            "important_nodes": [
                {"node_id": address, "importance_score": 1.0}  
            ],
            "important_edges": [
                {
                    "src": address,
                    "dst": n["neighbor_id"],
                    "importance_score": gnn_result["top_edge_values"][i]
                    if i < len(gnn_result["top_edge_values"]) else 0.0
                }
                for i, n in enumerate(neighbors[:3])
            ],
        },
        "rationale": rationale,
        "explanation_model": _model_version,
        "latency_warning": (
            f"Explanation generated in {elapsed:.1f}s. "
            "GNNExplainer is computationally intensive; 5–15s is expected."
        ),
    }
