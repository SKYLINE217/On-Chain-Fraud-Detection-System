# api/routers/explain.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
# Using Any instead of neo4j.AsyncDriver for typing if neo4j is missing.
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
    # Mocks for unit testing if Person A/C haven't built these yet
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

# Module-level: load model + data + explainers once on startup
_model = None
_data = None
_gnn_explainer = None
_shap_tree_explainer = None
_txid_to_idx: dict = {}


def _lazy_load():
    global _model, _data, _gnn_explainer, _shap_tree_explainer, _txid_to_idx
    if _model is None:
        import pandas as pd
        
        checkpoint_path = os.environ.get("MODEL_CHECKPOINT", "checkpoints/best_model.pt")
        config_path = os.environ.get("MODEL_CONFIG", "checkpoints/model_config.json")
        
        _model, _ = load_model_from_config(checkpoint_path, config_path)
        
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

        # XGBoost SHAP (TreeExplainer — fast + exact)
        try:
            xgb = joblib.load("checkpoints/xgb_baseline.pkl")
            import shap
            _shap_tree_explainer = shap.TreeExplainer(xgb)
        except FileNotFoundError:
            _shap_tree_explainer = None


@router.post("/{address}", response_model=ExplainResponse)
async def explain_address(address: str, driver: AsyncDriver = Depends(get_neo4j_driver)):
    """
    On-demand GNNExplainer + SHAP for a single address.
    Expected latency: 5–15s. Always includes latency_warning in response.
    NOT cached — instance-specific.
    """
    _lazy_load()

    if address not in _txid_to_idx:
        raise HTTPException(status_code=404, detail=f"Address not found: {address}")

    node_idx = _txid_to_idx[address]
    t_start = time.time()

    # 1. GNNExplainer
    gnn_result = explain_node(_gnn_explainer, _data, node_idx)

    # 2. SHAP (XGBoost TreeExplainer — fast)
    import numpy as np
    x_node = _data.x[node_idx].numpy().reshape(1, -1)
    
    if _shap_tree_explainer is not None:
        shap_features = compute_shap_for_node(
            _shap_tree_explainer,
            x_node,
            feature_names=FEATURE_COLS,
            top_k=10,
        )
    else:
        shap_features = []

    # 3. Get neighbor info for rationale
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
            neighbors = [r.data() for r in await result.data()] # Handle mock appropriately, but async driver works like this

    # 4. Map important edges to neighbor info
    edge_idx = _data.edge_index
    incident_mask = (edge_idx[0] == node_idx) | (edge_idx[1] == node_idx)
    incident_edge_indices = incident_mask.nonzero(as_tuple=True)[0]
    top_k_edge_map = {
        i.item(): gnn_result["top_edge_values"][j]
        for j, i in enumerate(gnn_result["top_edge_indices"][:3])
        if i.item() in incident_edge_indices.tolist()
    }

    # Build edge_mask_top for rationale
    edge_mask_top = []
    for n in neighbors[:3]:
        importance = top_k_edge_map.get(0, 0.0)  # simplified; use real idx in prod
        edge_mask_top.append((n["neighbor_id"], importance, n.get("label", "unknown")))

    # 5. Get predicted label for rationale
    import torch.nn.functional as F
    with torch.no_grad():
        logits = _model(_data.x, _data.edge_index)
        probs  = F.softmax(logits[node_idx], dim=0)
    predicted_label = "illicit" if probs[1] > 0.5 else "licit"

    # 6. Generate rationale
    rationale = generate_rationale(shap_features, edge_mask_top, predicted_label)

    elapsed = time.time() - t_start

    return {
        "address": address,
        "shap_top_features": shap_features,
        "subgraph_explanation": {
            "important_nodes": [
                {"node_id": address, "importance_score": 1.0}  # target node
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
        "explanation_model": "GNNExplainer (PyG) + SHAP TreeExplainer (XGBoost)",
        "latency_warning": (
            f"Explanation generated in {elapsed:.1f}s. "
            "GNNExplainer is computationally intensive; 5–15s is expected."
        ),
    }
