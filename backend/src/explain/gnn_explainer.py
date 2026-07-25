# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

"""
GNNExplainer wrapper — per-node subgraph + feature importance.

Uses PyG's Explainer API (torch_geometric.explain) to produce:
  - node_mask:  per-feature importance scores for the explained node
  - edge_mask:  importance scores for edges in the local subgraph

Runtime: 1–5 seconds per node (GNNExplainer runs a small gradient-descent
loop per node). This is expected behavior — document in API reference that
/explain has 5–15s latency budget.

Contract 5 (blend.md) compliance:
  - Returns important_nodes with importance_score
  - Returns important_edges (src, dst) with importance_score
  - Feature names use exact column names from features_combined.parquet
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

logger = logging.getLogger(__name__)


class GNNExplainerWrapper:
    """
    Wraps PyG's GNNExplainer for per-node explanations.

    Parameters
    ----------
    model : torch.nn.Module
        Trained GNN model (GraphSAGE or GAT).
    data : torch_geometric.data.Data
        PyG Data object with x, edge_index.
    feature_names : list[str]
        Feature column names (from features_combined.parquet).
    explainer_epochs : int
        Number of optimization epochs for GNNExplainer (default: 200).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        data,
        feature_names: List[str],
        explainer_epochs: int = 200,
    ):
        self.model = model
        self.data = data
        self.feature_names = feature_names
        self.explainer_epochs = explainer_epochs
        self._explainer = None

    def _get_explainer(self):
        """Lazy-initialize the Explainer object."""
        if self._explainer is None:
            try:
                from torch_geometric.explain import Explainer, GNNExplainer

                self._explainer = Explainer(
                    model=self.model,
                    algorithm=GNNExplainer(epochs=self.explainer_epochs),
                    explanation_type="model",
                    node_mask_type="attributes",
                    edge_mask_type="object",
                    model_config=dict(
                        mode="multiclass_classification",
                        task_level="node",
                        return_type="raw",
                    ),
                )
                logger.info(
                    "GNNExplainer initialized (epochs=%d)", self.explainer_epochs
                )
            except ImportError:
                logger.warning(
                    "torch_geometric.explain not available — using fallback"
                )
                self._explainer = "fallback"
        return self._explainer

    def explain_node(
        self,
        node_idx: int,
        top_k_features: int = 10,
        top_k_edges: int = 10,
        top_k_nodes: int = 10,
    ) -> Dict:
        """
        Generate an explanation for a single node.

        Parameters
        ----------
        node_idx : int
            Index of the node to explain.
        top_k_features : int
            Number of top features to return.
        top_k_edges : int
            Number of top important edges to return.
        top_k_nodes : int
            Number of top important neighbor nodes to return.

        Returns
        -------
        dict with keys:
            - node_mask: raw feature importance tensor
            - edge_mask: raw edge importance tensor
            - top_features: list of (feature_name, importance_score)
            - important_nodes: list of {node_id, importance_score}
            - important_edges: list of {src, dst, importance_score}
        """
        explainer = self._get_explainer()

        if explainer == "fallback":
            return self._fallback_explain(node_idx, top_k_features, top_k_edges, top_k_nodes)

        try:
            explanation = explainer(
                x=self.data.x,
                edge_index=self.data.edge_index,
                index=node_idx,
            )

            # Extract feature importances from node_mask
            node_mask = explanation.node_mask
            if node_mask is not None:
                # node_mask shape: (num_features,) or (num_nodes, num_features)
                if node_mask.dim() == 2:
                    feature_importance = node_mask[node_idx].detach().cpu().numpy()
                else:
                    feature_importance = node_mask.detach().cpu().numpy()
            else:
                feature_importance = np.zeros(len(self.feature_names))

            # Extract edge importances
            edge_mask = explanation.edge_mask
            if edge_mask is not None:
                edge_importance = edge_mask.detach().cpu().numpy()
            else:
                edge_importance = np.zeros(self.data.edge_index.shape[1])

            # Top-K features
            top_feat_indices = np.argsort(np.abs(feature_importance))[::-1][:top_k_features]
            top_features = [
                {
                    "feature_name": self.feature_names[i] if i < len(self.feature_names) else f"f{i}",
                    "feature_value": float(self.data.x[node_idx, i].item()),
                    "importance_score": float(feature_importance[i]),
                }
                for i in top_feat_indices
            ]

            # Top-K edges
            top_edge_indices = np.argsort(edge_importance)[::-1][:top_k_edges]
            important_edges = []
            for idx in top_edge_indices:
                if edge_importance[idx] > 0:
                    src = int(self.data.edge_index[0, idx].item())
                    dst = int(self.data.edge_index[1, idx].item())
                    important_edges.append({
                        "src": str(src),
                        "dst": str(dst),
                        "importance_score": float(edge_importance[idx]),
                    })

            # Top-K important neighbor nodes (aggregate edge importance per node)
            node_importance = {}
            for idx in range(len(edge_importance)):
                src = int(self.data.edge_index[0, idx].item())
                dst = int(self.data.edge_index[1, idx].item())
                # Aggregate importance for neighboring nodes
                for nid in [src, dst]:
                    if nid != node_idx:
                        node_importance[nid] = node_importance.get(nid, 0) + float(edge_importance[idx])

            sorted_nodes = sorted(node_importance.items(), key=lambda x: x[1], reverse=True)
            important_nodes = [
                {"node_id": str(nid), "importance_score": score}
                for nid, score in sorted_nodes[:top_k_nodes]
                if score > 0
            ]

            return {
                "node_mask": node_mask,
                "edge_mask": edge_mask,
                "top_features": top_features,
                "important_nodes": important_nodes,
                "important_edges": important_edges,
            }

        except Exception as e:
            logger.error("GNNExplainer failed for node %d: %s", node_idx, e)
            return self._fallback_explain(node_idx, top_k_features, top_k_edges, top_k_nodes)

    def _fallback_explain(
        self,
        node_idx: int,
        top_k_features: int = 10,
        top_k_edges: int = 10,
        top_k_nodes: int = 10,
    ) -> Dict:
        """
        Gradient-based fallback when GNNExplainer is not available.
        Uses input gradient magnitudes as a proxy for feature importance.
        """
        logger.info("Using gradient-based fallback for node %d", node_idx)

        self.model.eval()
        x = self.data.x.clone().detach().requires_grad_(True)

        out = self.model(x, self.data.edge_index)
        pred_class = out[node_idx].argmax().item()

        # Backward pass on predicted class score
        out[node_idx, pred_class].backward()

        # Feature importance = gradient magnitude at the target node
        grad = x.grad[node_idx].detach().cpu().numpy()
        feature_importance = np.abs(grad)

        # Top features
        top_feat_indices = np.argsort(feature_importance)[::-1][:top_k_features]
        top_features = [
            {
                "feature_name": self.feature_names[i] if i < len(self.feature_names) else f"f{i}",
                "feature_value": float(self.data.x[node_idx, i].item()),
                "importance_score": float(feature_importance[i]),
            }
            for i in top_feat_indices
        ]

        # Find neighboring edges and nodes
        edge_index = self.data.edge_index
        mask = (edge_index[0] == node_idx) | (edge_index[1] == node_idx)
        neighbor_edge_indices = mask.nonzero(as_tuple=True)[0].tolist()

        important_edges = []
        node_importance = {}
        for idx in neighbor_edge_indices[:top_k_edges]:
            src = int(edge_index[0, idx].item())
            dst = int(edge_index[1, idx].item())
            # Use uniform importance as fallback
            imp = 1.0 / max(len(neighbor_edge_indices), 1)
            important_edges.append({
                "src": str(src),
                "dst": str(dst),
                "importance_score": imp,
            })
            for nid in [src, dst]:
                if nid != node_idx:
                    node_importance[nid] = node_importance.get(nid, 0) + imp

        sorted_nodes = sorted(node_importance.items(), key=lambda x: x[1], reverse=True)
        important_nodes = [
            {"node_id": str(nid), "importance_score": score}
            for nid, score in sorted_nodes[:top_k_nodes]
        ]

        return {
            "node_mask": None,
            "edge_mask": None,
            "top_features": top_features,
            "important_nodes": important_nodes,
            "important_edges": important_edges,
        }
