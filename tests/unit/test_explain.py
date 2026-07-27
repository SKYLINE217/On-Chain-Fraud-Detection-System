
import pytest
import torch
import numpy as np
import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.explain.rationale import generate_rationale, FEATURE_READABLE_NAMES
from src.explain.shap_explainer import compute_shap_for_node

class TestRationale:
    def test_illicit_prefix(self):
        shap_features = [
            {"feature_name": "burst_score", "feature_value": 4.2, "shap_value": 0.34},
            {"feature_name": "tx_freq", "feature_value": 312.0, "shap_value": 0.22},
        ]
        rationale = generate_rationale(shap_features, [], "illicit")
        assert rationale.startswith("Flagged due to")
        assert "burst_score" not in rationale or "temporal burst" in rationale

    def test_licit_prefix(self):
        shap_features = [
            {"feature_name": "clustering_coeff", "feature_value": 0.8, "shap_value": -0.15},
        ]
        rationale = generate_rationale(shap_features, [], "licit")
        assert rationale.startswith("Classified as licit due to")

    def test_empty_shap_fallback(self):
        rationale = generate_rationale([], [], "unknown")
        assert "Insufficient" in rationale

    def test_anonymized_features_no_semantic_claim(self):
        shap_features = [
            {"feature_name": "f42", "feature_value": 1.23, "shap_value": 0.5},
        ]
        rationale = generate_rationale(shap_features, [], "illicit")

        assert "f42" in rationale
        assert "Anonymized" in rationale

class TestShapOutput:
    def test_shap_sorted_by_abs(self):

        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = [
            None,  
            np.array([[0.1, -0.5, 0.3, 0.05]])  
        ]
        feature_names = ["f1", "f2", "f3", "f4"]
        x_node = np.zeros((1, 4))

        result = compute_shap_for_node(mock_explainer, x_node, feature_names, top_k=3)

        assert len(result) == 3
        assert result[0]["feature_name"] == "f2"
        assert result[1]["feature_name"] == "f3"
        assert result[2]["feature_name"] == "f1"
