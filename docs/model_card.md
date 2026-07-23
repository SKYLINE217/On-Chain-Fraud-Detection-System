# Model Card — On-Chain Fraud/AML Detection System

> **Status:** Draft — Stage 1 (baselines only). Will be updated at each stage.

> **Disclaimer:** This system is a **research/portfolio demonstration only**. It is explicitly **NOT** a certified AML/CFT compliance tool, a regulated financial product, or a legally defensible fraud-detection system. It must not be used for regulatory reporting, enforcement decisions, or any purpose requiring compliance with financial regulations (e.g., BSA, FinCEN, EU AMLD, or equivalent).

---

## 1. Model Details

### Architecture
- **Stage 1 (current):** Tabular baselines — Logistic Regression, Random Forest, XGBoost
- **Stage 2–3 (planned):** GraphSAGE (primary GNN) and GAT (secondary GNN) via PyTorch Geometric

### Training Data
- **Dataset:** Elliptic Bitcoin Transaction Dataset (Kaggle)
- **Nodes:** 203,769 transactions
- **Edges:** 234,355 directed (BTC flow direction)
- **Node features:** 166 per node (94 local + 72 aggregated from 1-hop neighbors)
- **Time steps:** 49 temporal snapshots
- **Class distribution:** ~2% illicit, ~21% licit, ~77% unknown

### Temporal Split (Non-Negotiable)

```
Time steps:  1 ──────────── 34 | 35 ─── 39 | 40 ──── 49
             TRAIN              VAL          TEST
```

This mirrors the standard Elliptic benchmark protocol to prevent future information leakage and reflect realistic deployment where predictions are made on unseen future data. All models — baselines and GNNs — use this identical split for valid comparison.

- **Train:** Time steps 1–34 (labeled nodes only)
- **Validation:** Time steps 35–39 (used for hyperparameter selection)
- **Test:** Time steps 40–49 (touched once for final evaluation only)

### Loss Function
- **Baselines:** Class-weighted objectives (balanced class weights for LR/RF, `scale_pos_weight` for XGBoost)
- **GNNs (planned):** Weighted `CrossEntropyLoss` with inverse frequency weighting; Focal Loss (γ=2) as alternative

---

## 2. Intended Use

- Research and portfolio demonstration of GNN-based on-chain fraud detection
- Technical evaluation by employers and reviewers
- Academic reference for Graph Neural Network approaches to AML

**This is NOT intended for:**
- Regulatory AML/CFT compliance
- Automated enforcement or blocking decisions
- Production financial crime detection without extensive further validation

---

## 3. Metrics

**Primary metric: PR-AUC on illicit class.**

With only ~2% of nodes labeled as illicit, accuracy is meaningless (a trivial "predict licit always" classifier achieves ~98%+ accuracy). All metrics are reported on the illicit class specifically.

| Metric | Description |
|--------|-------------|
| Precision (illicit) | Of nodes predicted illicit, what fraction truly are |
| Recall (illicit) | Of truly illicit nodes, what fraction were caught |
| F1 (illicit) | Harmonic mean of precision and recall |
| **PR-AUC (illicit)** | **Area under the Precision-Recall curve — PRIMARY** |
| ROC-AUC | Area under the ROC curve (secondary; optimistic under this imbalance) |

### Baseline Results (Val Set — Time Steps 35–39)

*To be filled after Stage 1 training runs:*

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|-------|-----------|--------|----|--------|---------|
| Logistic Regression | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| XGBoost | — | — | — | — | — |
| GraphSAGE (raw features) | — | — | — | — | — |
| GraphSAGE (+ engineered) | — | — | — | — | — |
| GAT (raw features) | — | — | — | — | — |
| GAT (+ engineered) | — | — | — | — | — |

---

## 4. Limitations

### Feature Semantics
- Raw Elliptic features `f1`–`f166` are **intentionally anonymized** by the dataset creators. We cannot assert what any raw feature represents (e.g., "transaction amount") with confidence.
- **Only Person A's engineered features** (tx_freq, amount_mean, amount_skew, address_age, clustering_coeff, burst_score) have interpretable semantics. Interpretability claims in the explainability layer are limited to these.

### Dataset
- Elliptic data is from **2019** and is **Bitcoin-specific**; illicit tactics evolve over time, and this dataset may not represent current laundering patterns.
- Applying the model to Ethereum or other chains without retraining would introduce **distribution shift**.

### Explainability (planned — Stage 4)
- `KernelExplainer` (SHAP on GNN) treats the GNN as a black box over node features only — it **does not account for message-passing structure**. This is an approximation. `TreeExplainer` on XGBoost is exact.
- GNNExplainer is instance-specific and takes 1–5s per node; explanations are not cached.

---

## 5. Ethical Considerations

- **False positives** can flag legitimate users as illicit. This system must be used as a **decision-support tool**, not an automated enforcement mechanism.
- Risk scores and labels should always be reviewed by a human before any action is taken.
- The system operates at address-level and cluster-level only; no individual deanonymization is attempted.

---

*Last updated: Stage 1 (Week 1). Update this card at each stage as models and metrics evolve.*
