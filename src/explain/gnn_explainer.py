import torch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.data import Data

def build_gnn_explainer(model: torch.nn.Module) -> Explainer:
    """
    Build GNNExplainer instance.

    Expected runtime: 1-5 seconds per node.
    Do NOT precompute or cache — explanations are instance-specific.
    """
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="log_probs",
        ),
    )

def explain_node(
    explainer: Explainer,
    data: Data,
    node_idx: int,
) -> dict:
    """
    Explain a single node.

    Args:
        explainer: Built GNNExplainer instance
        data: Full PyG Data object
        node_idx: Integer index of the target node

    Returns:
        dict with node_mask, edge_mask, and top-k indices
    """
    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index,
        index=node_idx,
    )

    node_mask = explanation.node_mask    
    edge_mask = explanation.edge_mask    

    feature_importance = node_mask[node_idx].abs()
    top_k_features = torch.topk(feature_importance, k=min(10, len(feature_importance)))

    top_k_edges = torch.topk(edge_mask, k=min(10, len(edge_mask)))

    return {
        "node_mask": node_mask.detach().cpu(),
        "edge_mask": edge_mask.detach().cpu(),
        "top_feature_indices": top_k_features.indices.tolist(),
        "top_feature_values": top_k_features.values.tolist(),
        "top_edge_indices": top_k_edges.indices.tolist(),
        "top_edge_values": top_k_edges.values.tolist(),
    }
