# Model Card

## Model Details
- **Name:** GraphSAGE Fraud Classifier
- **Version:** 1.0 (see `checkpoints/model_config.json` for exact config)
- **Type:** Node classification GNN (inductive)
- **Framework:** PyTorch Geometric
- **Training data:** Elliptic Bitcoin Transaction Dataset (time steps 1–34, labeled only)

## Intended Use
- Portfolio / research demonstration of GNN-based fraud detection
- NOT intended for: regulatory use, enforcement, financial compliance decisions

## Performance
*(Placeholder: Metrics to be filled after W&B sweep)*

| Split | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Val (35–39) | *see model_config.json* | *TBD* | *TBD* | *TBD* |
| Test (40–49) | *see model_config.json* | *TBD* | *TBD* | *TBD* |

## Known Limitations
1. **Temporal degradation:** F1 degrades at time steps 45–49 due to distribution shift.
2. **High-volume exchange false positives:** Exchanges resemble mixers structurally (high `tx_freq`, `burst_score`) — common FP source.
3. **Unknown node boundary effects:** Nodes neighboring many unknown-class nodes have noisier neighborhood aggregation.
4. **SHAP approximation:** KernelExplainer SHAP does not capture message-passing structure — explanations are feature-attribution approximations only.
5. **Inductive vs static:** GraphSAGE generalizes to new nodes; however, Elliptic txIds are anonymized — real-world inference requires careful re-mapping.

## Explainability
- **GNNExplainer:** per-node feature and edge importance (1–5s per node).
- **SHAP TreeExplainer (XGBoost):** exact feature attribution for tabular baseline.
- **SHAP KernelExplainer (GNN):** approximate, treats GNN as black box.

## Ethical Considerations
- System labels transactions as "illicit" based on training patterns only.
- False positives could incorrectly flag legitimate high-volume activity.
- System should NOT be used for enforcement without human review.
- All txIds are anonymized hashes — no personally identifiable information.

## Training Details
See `checkpoints/model_config.json` for exact hyperparameters and Weights & Biases run ID.

---

> **Compliance Disclaimer:** This system is a research and portfolio demonstration only. It is NOT a certified AML/CFT compliance tool, a regulated financial product, or a legally defensible fraud-detection system. It must not be used for regulatory reporting, enforcement decisions, or any purpose requiring compliance with financial regulations (BSA, FinCEN, EU AMLD, or equivalent). The authors disclaim all liability for any such use.
