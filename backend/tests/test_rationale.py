# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

"""
Tests for the rationale generator.

Verifies:
  - Rationale string is non-empty
  - Rationale is parseable (starts with expected prefix)
  - Different risk levels produce different rationale text
  - Handles edge cases (empty features, empty edges)
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.explain.rationale import generate_rationale, generate_risk_summary


@pytest.fixture
def sample_shap_features():
    """Sample SHAP feature attributions."""
    return [
        {"feature_name": "burst_score", "feature_value": 3.2, "shap_value": 0.15},
        {"feature_name": "tx_freq", "feature_value": 42.0, "shap_value": 0.08},
        {"feature_name": "clustering_coeff", "feature_value": 0.01, "shap_value": -0.05},
        {"feature_name": "f42", "feature_value": 1.5, "shap_value": 0.03},
        {"feature_name": "pageRank", "feature_value": 0.0001, "shap_value": -0.02},
    ]


@pytest.fixture
def sample_important_edges():
    """Sample important edges from GNNExplainer."""
    return [
        {"src": "1001", "dst": "1002", "importance_score": 0.85},
        {"src": "1002", "dst": "1003", "importance_score": 0.62},
        {"src": "1001", "dst": "1004", "importance_score": 0.41},
    ]


@pytest.fixture
def sample_important_nodes():
    """Sample important nodes from GNNExplainer."""
    return [
        {"node_id": "1002", "importance_score": 0.9},
        {"node_id": "1003", "importance_score": 0.7},
        {"node_id": "1004", "importance_score": 0.3},
    ]


class TestRationale:
    """Test suite for rationale generation."""

    def test_rationale_non_empty(self, sample_shap_features, sample_important_edges, sample_important_nodes):
        """Rationale should be a non-empty string."""
        rationale = generate_rationale(
            shap_top_features=sample_shap_features,
            important_edges=sample_important_edges,
            important_nodes=sample_important_nodes,
            risk_score=0.85,
            predicted_label="illicit",
        )
        assert isinstance(rationale, str)
        assert len(rationale) > 0

    def test_rationale_starts_with_flagged(self, sample_shap_features, sample_important_edges, sample_important_nodes):
        """Rationale should start with the 'Flagged as' prefix."""
        rationale = generate_rationale(
            shap_top_features=sample_shap_features,
            important_edges=sample_important_edges,
            important_nodes=sample_important_nodes,
            risk_score=0.85,
            predicted_label="illicit",
        )
        assert rationale.startswith("Flagged as")

    def test_high_risk_rationale(self, sample_shap_features, sample_important_edges, sample_important_nodes):
        """High risk scores should produce 'HIGH' in the rationale."""
        rationale = generate_rationale(
            shap_top_features=sample_shap_features,
            important_edges=sample_important_edges,
            important_nodes=sample_important_nodes,
            risk_score=0.9,
            predicted_label="illicit",
        )
        assert "HIGH" in rationale

    def test_low_risk_rationale(self, sample_shap_features, sample_important_edges, sample_important_nodes):
        """Low risk scores should produce 'LOW' or 'MINIMAL' in the rationale."""
        rationale = generate_rationale(
            shap_top_features=sample_shap_features,
            important_edges=sample_important_edges,
            important_nodes=sample_important_nodes,
            risk_score=0.15,
            predicted_label="licit",
        )
        assert "MINIMAL" in rationale or "LOW" in rationale

    def test_empty_features(self, sample_important_edges, sample_important_nodes):
        """Should handle empty SHAP features gracefully."""
        rationale = generate_rationale(
            shap_top_features=[],
            important_edges=sample_important_edges,
            important_nodes=sample_important_nodes,
            risk_score=0.5,
            predicted_label="licit",
        )
        assert isinstance(rationale, str)
        assert len(rationale) > 0

    def test_empty_edges(self, sample_shap_features, sample_important_nodes):
        """Should handle empty edges gracefully."""
        rationale = generate_rationale(
            shap_top_features=sample_shap_features,
            important_edges=[],
            important_nodes=sample_important_nodes,
            risk_score=0.5,
            predicted_label="licit",
        )
        assert isinstance(rationale, str)
        assert len(rationale) > 0

    def test_empty_everything(self):
        """Should handle completely empty inputs gracefully."""
        rationale = generate_rationale(
            shap_top_features=[],
            important_edges=[],
            important_nodes=[],
            risk_score=0.5,
            predicted_label="unknown",
        )
        assert isinstance(rationale, str)
        assert len(rationale) > 0

    def test_feature_name_humanization(self, sample_important_edges, sample_important_nodes):
        """Engineered features should be humanized in the rationale."""
        features = [
            {"feature_name": "burst_score", "feature_value": 5.0, "shap_value": 0.3},
        ]
        rationale = generate_rationale(
            shap_top_features=features,
            important_edges=sample_important_edges,
            important_nodes=sample_important_nodes,
            risk_score=0.8,
            predicted_label="illicit",
        )
        assert "Temporal burst score" in rationale


class TestRiskSummary:
    """Test suite for generate_risk_summary."""

    def test_summary_non_empty(self, sample_shap_features):
        """Risk summary should be non-empty."""
        summary = generate_risk_summary(
            risk_score=0.85,
            predicted_label="illicit",
            shap_top_features=sample_shap_features,
        )
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_high_risk_label(self, sample_shap_features):
        """High risk should be labeled accordingly."""
        summary = generate_risk_summary(
            risk_score=0.9,
            predicted_label="illicit",
            shap_top_features=sample_shap_features,
        )
        assert "High risk" in summary

    def test_low_risk_label(self, sample_shap_features):
        """Low risk should be labeled accordingly."""
        summary = generate_risk_summary(
            risk_score=0.1,
            predicted_label="licit",
            shap_top_features=sample_shap_features,
        )
        assert "Minimal risk" in summary

    def test_empty_features_summary(self):
        """Should handle empty features."""
        summary = generate_risk_summary(
            risk_score=0.5,
            predicted_label="unknown",
            shap_top_features=[],
        )
        assert isinstance(summary, str)
        assert len(summary) > 0
