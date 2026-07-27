import shap
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

def build_tree_explainer(xgb_model):
    """Fast, exact SHAP for XGBoost baseline."""
    return shap.TreeExplainer(xgb_model)

def build_kernel_explainer(model: torch.nn.Module, data: Data, n_background: int = 100):
    """
    Approximate SHAP for GNN (treats GNN as black box over node features).

    LIMITATION: KernelExplainer does NOT account for message-passing structure.
    It treats each node independently — neighbor influence is not captured.
    Always document this limitation.

    Background sample: random subset of n_background labeled nodes.
    """

    labeled_mask = (data.y >= 0)
    labeled_indices = labeled_mask.nonzero(as_tuple=True)[0]
    bg_indices = labeled_indices[
        torch.randperm(len(labeled_indices))[:n_background]
    ]
    background = data.x[bg_indices].numpy()

    def gnn_predict(x_np: np.ndarray) -> np.ndarray:
        """Wrapper: numpy in → numpy probabilities out."""
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x_np)
            out = model(x_tensor, data.edge_index)
        return F.softmax(out, dim=1).numpy()

    explainer = shap.KernelExplainer(gnn_predict, background)
    return explainer

def compute_shap_for_node(
    explainer,  
    x_node: np.ndarray,  
    feature_names: list[str],
    top_k: int = 10,
) -> list[dict]:
    """
    Compute SHAP values for a single node.

    Returns:
        list of {feature_name, feature_value, shap_value} sorted by |shap_value|
    """
    shap_values = explainer.shap_values(x_node)

    if isinstance(shap_values, list):
        sv = shap_values[1][0]  
    else:
        sv = shap_values[0]     

    sorted_idx = np.argsort(np.abs(sv))[::-1][:top_k]

    return [
        {
            "feature_name": feature_names[i],
            "feature_value": float(x_node[0, i]),
            "shap_value": float(sv[i]),
        }
        for i in sorted_idx
    ]
