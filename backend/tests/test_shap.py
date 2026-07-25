# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

"""
Tests for SHAP explainer wrapper.

Verifies:
  - SHAP values shape matches feature count
  - Output format matches Contract 5 schema
  - Feature names are preserved in output
  - Synthetic fallback works when SHAP is unavailable
"""

import sys
from pathlib import Path

import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.explain.shap_explainer import SHAPExplainer


@pytest.fixture
def feature_names():
    """Sample feature names matching expected format."""
    raw = [f"f{i}" for i in range(1, 167)]
    engineered = [
        "tx_freq", "amount_mean", "amount_skew", "address_age",
        "clustering_coeff", "burst_score", "pageRank", "communityId",
    ]
    return raw + engineered


@pytest.fixture
def sample_features(feature_names):
    """Sample feature vector."""
    return np.random.randn(len(feature_names)).astype(np.float32)


@pytest.fixture
def shap_explainer(feature_names):
    """SHAP explainer instance."""
    return SHAPExplainer(feature_names=feature_names)


class TestSHAPExplainer:
    """Test suite for SHAP feature attribution."""

    def test_synthetic_shap_returns_list(self, shap_explainer, sample_features):
        """Synthetic SHAP should return a list of feature dicts."""
        result = shap_explainer._synthetic_shap(sample_features, top_k=10)
        assert isinstance(result, list)
        assert len(result) == 10

    def test_synthetic_shap_schema(self, shap_explainer, sample_features):
        """Each entry should match Contract 5 schema."""
        result = shap_explainer._synthetic_shap(sample_features, top_k=5)
        for entry in result:
            assert "feature_name" in entry
            assert "feature_value" in entry
            assert "shap_value" in entry
            assert isinstance(entry["feature_name"], str)
            assert isinstance(entry["feature_value"], float)
            assert isinstance(entry["shap_value"], float)

    def test_explain_node_fallback(self, shap_explainer, sample_features):
        """explain_node should fall back to synthetic when no backend available."""
        result = shap_explainer.explain_node(sample_features, top_k=10)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_feature_names_preserved(self, shap_explainer, sample_features, feature_names):
        """Feature names in output should come from the configured list."""
        result = shap_explainer.explain_node(sample_features, top_k=10)
        valid_names = set(feature_names)
        for entry in result:
            assert entry["feature_name"] in valid_names, \
                f"Unexpected feature name: {entry['feature_name']}"

    def test_top_k_limits(self, shap_explainer, sample_features):
        """top_k should limit the number of returned features."""
        for k in [3, 5, 10, 20]:
            result = shap_explainer.explain_node(sample_features, top_k=k)
            assert len(result) <= k

    def test_sorted_by_abs_value(self, shap_explainer, sample_features):
        """Results should be sorted by absolute SHAP value (descending)."""
        result = shap_explainer._synthetic_shap(sample_features, top_k=10)
        abs_values = [abs(entry["shap_value"]) for entry in result]
        assert abs_values == sorted(abs_values, reverse=True)

    def test_format_top_features(self, shap_explainer, sample_features):
        """_format_top_features should produce correct output."""
        n = len(sample_features)
        shap_values = np.random.randn(n).astype(np.float32)
        result = shap_explainer._format_top_features(sample_features, shap_values, top_k=5)
        assert len(result) == 5
        for entry in result:
            assert "feature_name" in entry
            assert "feature_value" in entry
            assert "shap_value" in entry

    def test_different_feature_vectors(self, shap_explainer):
        """Different feature vectors should produce different explanations."""
        features_a = np.random.randn(174).astype(np.float32)
        features_b = np.random.randn(174).astype(np.float32) * 10

        result_a = shap_explainer.explain_node(features_a, top_k=5)
        result_b = shap_explainer.explain_node(features_b, top_k=5)

        # At least some values should differ
        values_a = [e["shap_value"] for e in result_a]
        values_b = [e["shap_value"] for e in result_b]
        assert values_a != values_b, "Different inputs should produce different SHAP values"
