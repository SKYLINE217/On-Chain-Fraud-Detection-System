# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

"""
Rationale generator — produces human-readable explanations.

Combines SHAP feature attributions + GNNExplainer edge importances
into a structured natural-language rationale string.

Contract 5 (blend.md):
    "rationale": "Flagged due to: ..."

The rationale highlights:
  1. Top contributing features with their SHAP direction
  2. Notable neighbor connections (especially to illicit-labeled nodes)
  3. Risk level assessment
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def generate_rationale(
    shap_top_features: List[Dict],
    important_edges: List[Dict],
    important_nodes: List[Dict],
    risk_score: float,
    predicted_label: str,
    node_metadata: Optional[Dict] = None,
) -> str:
    """
    Generate a human-readable rationale string.

    Parameters
    ----------
    shap_top_features : list of dict
        Top features from SHAP: [{feature_name, feature_value, shap_value}, ...]
    important_edges : list of dict
        From GNNExplainer: [{src, dst, importance_score}, ...]
    important_nodes : list of dict
        From GNNExplainer: [{node_id, importance_score}, ...]
    risk_score : float
        Model's predicted risk score [0, 1].
    predicted_label : str
        "illicit", "licit", or "unknown".
    node_metadata : dict, optional
        Additional metadata for neighbor nodes (labels, scores, etc.).

    Returns
    -------
    str
        Human-readable rationale string.
    """
    reasons = []

    # ── Risk Level Assessment ────────────────────────────────────────
    if risk_score >= 0.8:
        risk_level = "HIGH"
    elif risk_score >= 0.5:
        risk_level = "MODERATE"
    elif risk_score >= 0.2:
        risk_level = "LOW"
    else:
        risk_level = "MINIMAL"

    # ── SHAP Feature Contributions ───────────────────────────────────
    positive_features = []
    negative_features = []

    for feat in shap_top_features[:5]:  # Top 5 most impactful
        name = feat.get("feature_name", "unknown")
        value = feat.get("feature_value", 0.0)
        shap_val = feat.get("shap_value", 0.0)

        # Humanize feature names
        display_name = _humanize_feature_name(name)

        if shap_val > 0.01:
            positive_features.append(
                f"{display_name} = {value:.3f} ({shap_val:+.3f} risk impact)"
            )
        elif shap_val < -0.01:
            negative_features.append(
                f"{display_name} = {value:.3f} ({shap_val:+.3f} risk reduction)"
            )

    if positive_features:
        reasons.append(
            "Risk-increasing features: " + "; ".join(positive_features)
        )

    if negative_features:
        reasons.append(
            "Risk-decreasing features: " + "; ".join(negative_features)
        )

    # ── Network Context (GNNExplainer) ───────────────────────────────
    if important_nodes:
        n_important = len(important_nodes)
        node_metadata = node_metadata or {}

        flagged_neighbors = []
        for node in important_nodes[:3]:
            nid = node.get("node_id", "?")
            imp = node.get("importance_score", 0)
            meta = node_metadata.get(nid, {})
            label = meta.get("predicted_label", "unknown")

            if label == "illicit":
                flagged_neighbors.append(
                    f"node {nid} (illicit, importance: {imp:.3f})"
                )
            elif imp > 0.1:
                flagged_neighbors.append(
                    f"node {nid} ({label}, importance: {imp:.3f})"
                )

        if flagged_neighbors:
            reasons.append(
                f"Connected to {len(flagged_neighbors)} notable neighbor(s): "
                + "; ".join(flagged_neighbors)
            )
        elif n_important > 0:
            reasons.append(
                f"Network analysis identified {n_important} influential "
                f"neighbor node(s) in the transaction subgraph"
            )

    if important_edges:
        reasons.append(
            f"GNNExplainer identified {len(important_edges)} important "
            f"transaction flow(s) in the local subgraph"
        )

    # ── Assemble Rationale ───────────────────────────────────────────
    if not reasons:
        reasons.append(
            f"Risk score {risk_score:.3f} based on combined feature "
            f"and network analysis"
        )

    prefix = f"Flagged as {risk_level} risk ({predicted_label}, score: {risk_score:.3f})"
    rationale = f"{prefix}. " + ". ".join(reasons) + "."

    return rationale


def _humanize_feature_name(name: str) -> str:
    """
    Convert feature column names to human-readable labels.
    Only engineered features have interpretable semantics —
    raw features f1–f166 remain as-is (they are anonymized).
    """
    humanized = {
        "tx_freq": "Transaction frequency",
        "amount_mean": "Mean transaction amount",
        "amount_skew": "Amount distribution skewness",
        "address_age": "Address age (time steps)",
        "clustering_coeff": "Clustering coefficient",
        "burst_score": "Temporal burst score",
        "pageRank": "PageRank centrality",
        "communityId": "Community ID",
    }
    return humanized.get(name, name)


def generate_risk_summary(
    risk_score: float,
    predicted_label: str,
    shap_top_features: List[Dict],
) -> str:
    """
    Generate a brief one-line risk summary for dashboard display.

    Returns
    -------
    str
        Brief summary string, e.g. "High risk — driven by burst_score and tx_freq"
    """
    if risk_score >= 0.8:
        level = "High risk"
    elif risk_score >= 0.5:
        level = "Moderate risk"
    elif risk_score >= 0.2:
        level = "Low risk"
    else:
        level = "Minimal risk"

    # Top 2 driving features
    drivers = []
    for feat in shap_top_features[:2]:
        name = feat.get("feature_name", "")
        if name:
            drivers.append(_humanize_feature_name(name))

    if drivers:
        return f"{level} — driven by {' and '.join(drivers)}"
    else:
        return f"{level} (score: {risk_score:.3f})"
