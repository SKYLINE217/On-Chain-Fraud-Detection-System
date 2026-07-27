"""
rationale.py -- Generate human-readable explanation strings from SHAP and GNNExplainer outputs.
"""

FEATURE_READABLE_NAMES = {
    "tx_freq": "transaction frequency",
    "amount_mean": "mean transaction amount",
    "amount_skew": "transaction amount skewness",
    "address_age": "address age",
    "clustering_coeff": "local clustering coefficient",
    "burst_score": "temporal burst score",
    "pageRank": "network centrality (PageRank)",
    "communityId": "community membership",

}

def generate_rationale(
    shap_top_features: list[dict],
    edge_mask_top: list[tuple],   
    predicted_label: str,
) -> str:
    """
    Generate human-readable explanation string.

    Args:
        shap_top_features: output of compute_shap_for_node()
        edge_mask_top: top-K important incident edges with neighbor info
        predicted_label: "illicit" | "licit" | "unknown"

    Returns:
        Non-empty string starting with "Flagged due to:" or "Classified as licit due to:"
    """
    reasons = []

    prefix = "Flagged due to" if predicted_label == "illicit" else "Classified as licit due to"

    for feat in shap_top_features[:5]:
        name = FEATURE_READABLE_NAMES.get(feat["feature_name"], feat["feature_name"])
        impact = feat["shap_value"]
        val = feat["feature_value"]

        if feat["feature_name"].startswith("f") and feat["feature_name"][1:].isdigit():

            reasons.append(
                f"Anonymized feature {feat['feature_name']} "
                f"(value: {val:.2f}, SHAP impact: {impact:+.3f})"
            )
        else:
            direction = "elevated" if impact > 0 else "low"
            reasons.append(
                f"{direction.capitalize()} {name} "
                f"(value: {val:.2f}, SHAP impact: {impact:+.3f})"
            )

    for neighbor_id, edge_imp, neighbor_label in edge_mask_top[:3]:
        reasons.append(
            f"Connected to {neighbor_label} node "
            f"(edge importance: {edge_imp:.3f})"
        )

    if not reasons:
        reasons = ["Insufficient feature signal for detailed explanation"]

    return f"{prefix}: " + "; ".join(reasons)
