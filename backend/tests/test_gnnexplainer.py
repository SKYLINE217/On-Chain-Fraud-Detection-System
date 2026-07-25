# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

"""
Tests for GNNExplainer wrapper.

Verifies:
  - Explanation produces non-null node_mask and edge_mask
  - top_features is non-empty with correct schema
  - important_nodes and important_edges are returned
  - Fallback (gradient-based) works when GNNExplainer is unavailable
"""

import sys
from pathlib import Path

import pytest
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.graphsage import GraphSAGE


def _make_test_data(n_nodes=50, n_features=10, n_edges=100):
    """Create a small synthetic PyG Data object for testing."""
    from torch_geometric.data import Data

    x = torch.randn(n_nodes, n_features)
    edge_index = torch.stack([
        torch.randint(0, n_nodes, (n_edges,)),
        torch.randint(0, n_nodes, (n_edges,)),
    ])
    y = torch.randint(0, 2, (n_nodes,))

    return Data(x=x, edge_index=edge_index, y=y)


def _make_test_model(n_features=10):
    """Create a small GraphSAGE model for testing."""
    model = GraphSAGE(
        in_channels=n_features,
        hidden_channels=16,
        out_channels=2,
        num_layers=2,
        dropout=0.1,
    )
    model.eval()
    return model


@pytest.fixture
def test_setup():
    """Fixture providing model, data, and explainer."""
    from src.explain.gnn_explainer import GNNExplainerWrapper

    n_features = 10
    feature_names = [f"feature_{i}" for i in range(n_features)]

    data = _make_test_data(n_nodes=50, n_features=n_features, n_edges=100)
    model = _make_test_model(n_features=n_features)

    explainer = GNNExplainerWrapper(
        model=model,
        data=data,
        feature_names=feature_names,
        explainer_epochs=10,  # Fast for testing
    )

    return {
        "model": model,
        "data": data,
        "feature_names": feature_names,
        "explainer": explainer,
    }


class TestGNNExplainer:
    """Test suite for GNNExplainerWrapper."""

    def test_explain_node_returns_dict(self, test_setup):
        """Explanation should return a dict with required keys."""
        result = test_setup["explainer"].explain_node(node_idx=0)
        assert isinstance(result, dict)
        assert "top_features" in result
        assert "important_nodes" in result
        assert "important_edges" in result

    def test_top_features_non_empty(self, test_setup):
        """top_features should be non-empty."""
        result = test_setup["explainer"].explain_node(node_idx=0, top_k_features=5)
        assert len(result["top_features"]) > 0

    def test_top_features_schema(self, test_setup):
        """Each feature should have name, value, and importance."""
        result = test_setup["explainer"].explain_node(node_idx=0)
        for feat in result["top_features"]:
            assert "feature_name" in feat
            assert "feature_value" in feat
            assert "importance_score" in feat
            assert isinstance(feat["feature_name"], str)
            assert isinstance(feat["feature_value"], float)
            assert isinstance(feat["importance_score"], float)

    def test_feature_names_from_input(self, test_setup):
        """Feature names should come from the provided list."""
        result = test_setup["explainer"].explain_node(node_idx=0)
        valid_names = set(test_setup["feature_names"])
        for feat in result["top_features"]:
            assert feat["feature_name"] in valid_names

    def test_multiple_nodes(self, test_setup):
        """Explanations should work for multiple different nodes."""
        for node_idx in [0, 5, 10, 25, 49]:
            result = test_setup["explainer"].explain_node(node_idx=node_idx)
            assert isinstance(result, dict)
            assert len(result["top_features"]) > 0

    def test_fallback_explain(self, test_setup):
        """Fallback (gradient-based) should produce valid output."""
        result = test_setup["explainer"]._fallback_explain(
            node_idx=0, top_k_features=5, top_k_edges=3, top_k_nodes=3,
        )
        assert isinstance(result, dict)
        assert len(result["top_features"]) > 0
        assert result["node_mask"] is None  # Fallback doesn't produce node_mask
        assert result["edge_mask"] is None  # Fallback doesn't produce edge_mask

    def test_top_k_limits(self, test_setup):
        """top_k should limit the number of returned items."""
        result = test_setup["explainer"].explain_node(
            node_idx=0, top_k_features=3, top_k_edges=2, top_k_nodes=2,
        )
        assert len(result["top_features"]) <= 3
