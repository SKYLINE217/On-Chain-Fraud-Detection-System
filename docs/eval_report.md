# Evaluation Report

## Overview
- **Dataset:** Elliptic Bitcoin Transaction Dataset
- **Task:** Node classification — illicit vs. licit transaction
- **Temporal split:** Train 1–34 / Val 35–39 / Test 40–49 (non-negotiable)
- **Primary metric:** PR-AUC on illicit class (class=1)
- **Secondary:** F1, Precision, Recall, ROC-AUC

## Dataset Statistics
| Property | Value |
|---|---|
| Nodes | 203,769 |
| Edges | 234,355 |
| Raw features | 166 (anonymized) |
| Engineered features | 8 |
| Total features | 174 |
| Labeled nodes | ~47,000 (23% of total) |
| Illicit (train) | ~4,500 (~9.5% of labeled) |
| Licit (train) | ~42,500 (~90.5% of labeled) |

## Model Comparison Table
*(Placeholder: Paste metrics from `docs/model_comparison.csv` here after completing the Weights & Biases sweep)*

| Model | Test PR-AUC | Test F1 | Test ROC-AUC |
|---|---|---|---|
| Logistic Regression | *TBD* | *TBD* | *TBD* |
| Random Forest | *TBD* | *TBD* | *TBD* |
| XGBoost Baseline | *TBD* | *TBD* | *TBD* |
| GraphSAGE | *TBD* | *TBD* | *TBD* |
| GAT | *TBD* | *TBD* | *TBD* |

## Temporal F1 Analysis
*(Placeholder: Insert `f1_over_time.png` figure here)*

**Key finding:** F1 degrades from step 43 onward, consistent with concept drift.
This is expected and documented in `model_card.md` §Known Limitations.

## PR Curve Comparison
*(Placeholder: Insert `pr_curves_all_models.png` figure here)*

## Known Result Caveats
- XGBoost / Random Forest may match or exceed GNN PR-AUC on this dataset (consistent with published Elliptic literature — reported honestly)
- ROC-AUC is optimistic under 2% class imbalance — always report alongside PR-AUC
- GNN advantage is inductive generalization to new nodes, not necessarily higher PR-AUC on this static dataset

## Compliance Disclaimer
> This system is a research and portfolio demonstration only. It is NOT a certified AML/CFT compliance tool, a regulated financial product, or a legally defensible fraud-detection system. It must not be used for regulatory reporting, enforcement decisions, or any purpose requiring compliance with financial regulations (BSA, FinCEN, EU AMLD, or equivalent). The authors disclaim all liability for any such use.
