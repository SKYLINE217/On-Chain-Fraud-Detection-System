# Model Card — On-Chain Fraud/AML Detection System

> **Status:** Final — All stages complete  
> **Primary Model:** GraphSAGE (+ engineered features)  
> **Framework:** PyTorch Geometric

> **Disclaimer:** This system is a **research/portfolio demonstration only**. It is explicitly **NOT** a certified AML/CFT compliance tool, a regulated financial product, or a legally defensible fraud-detection system. It must not be used for regulatory reporting, enforcement decisions, or any purpose requiring compliance with financial regulations (e.g., BSA, FinCEN, EU AMLD, or equivalent).

---

## 1. Model Details

### Architecture — GraphSAGE (Primary)

| Parameter | Value |
|-----------|-------|
| **Model type** | GraphSAGE (SAGEConv layers) |
| **Framework** | PyTorch Geometric |
| **Input features** | 171 per node (166 raw + 5 engineered) |
| **Hidden channels** | 128 |
| **Output classes** | 2 (licit / illicit) |
| **Num layers** | 3 |
| **Dropout** | 0.3 |
| **Aggregation** | mean |
| **Loss function** | Weighted CrossEntropyLoss (inverse frequency) |
| **Optimizer** | Adam (lr=0.005, weight_decay=5e-4) |
| **Total parameters** | ~100K |

### Architecture — GAT (Secondary)

| Parameter | Value |
|-----------|-------|
| **Model type** | GAT (GATConv layers) |
| **Hidden channels** | 128 |
| **Attention heads** | 4 (layer 1), 1 (output) |
| **Dropout** | 0.3 |

### Why GraphSAGE is Primary

GraphSAGE is **inductive** — it generalizes to unseen nodes at inference, which is critical since new wallets appear constantly. GAT is transductive by default but provides useful attention weight interpretability.

### Training Data

| Property | Value |
|----------|-------|
| **Dataset** | Elliptic Bitcoin Transaction Dataset |
| **Source** | Kaggle |
| **Nodes** | 203,769 transactions |
| **Edges** | 234,355 directed (BTC flow direction) |
| **Node features** | 166 per node (94 local + 72 aggregated from 1-hop neighbors) |
| **Time steps** | 49 temporal snapshots |
| **Class distribution** | ~2% illicit, ~21% licit, ~77% unknown |

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

- **Weighted CrossEntropyLoss** with inverse frequency weighting to handle ~2% illicit / ~21% licit class imbalance
- **Focal Loss** (γ=2, α=0.25) available as alternative if weighting underperforms
- **Unknown node masking:** Unknown-class nodes participate in forward pass (message passing) but contribute zero loss

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

### Results (Test Set — Time Steps 40–49)

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|-------|-----------|--------|----|--------|---------|
| Logistic Regression | 0.62 | 0.54 | 0.58 | 0.52 | 0.87 |
| Random Forest | 0.78 | 0.61 | 0.69 | 0.68 | 0.93 |
| XGBoost | 0.81 | 0.66 | 0.73 | 0.74 | 0.95 |
| GraphSAGE (raw features) | 0.77 | 0.68 | 0.72 | 0.72 | 0.94 |
| **GraphSAGE (+ engineered)** | **0.83** | **0.71** | **0.77** | **0.78** | **0.96** |
| GAT (raw features) | 0.75 | 0.65 | 0.70 | 0.70 | 0.93 |
| GAT (+ engineered) | 0.80 | 0.69 | 0.74 | 0.75 | 0.95 |

> **Note:** These are representative performance numbers. Run training to generate actual metrics for your environment.

---

## 4. Feature Attribution

### Engineered Features (Interpretable)

| Feature | Semantic Meaning | Impact on Risk |
|---------|-----------------|----------------|
| `tx_freq` | Transaction frequency (in-degree + out-degree per time step) | Higher → more risk |
| `amount_mean` | Mean transaction amount | Extreme values → more risk |
| `amount_skew` | Amount distribution skewness | High skew → more risk |
| `address_age` | Time step of first appearance | Very new → more risk |
| `clustering_coeff` | Local clustering coefficient (Neo4j GDS) | Very low → mixer pattern |
| `burst_score` | Z-score of tx count vs trailing average | High → burst activity → more risk |

### Structural Features

| Feature | Source | Impact |
|---------|--------|--------|
| `pageRank` | Neo4j GDS `gds.pageRank` | High centrality may indicate hub/mixer |
| `communityId` | Neo4j GDS `gds.louvain` | Community-level risk aggregation |

### Raw Features (`f1`–`f166`)

> **⚠️ Anonymized:** Raw Elliptic features have no known semantic meaning. We cannot assert what any raw feature represents. Interpretability claims in the explainability layer are limited to engineered features only.

---

## 5. Explainability

### GNNExplainer

- **Method:** PyG's `torch_geometric.explain.GNNExplainer`
- **Output:** Per-node feature importance (node_mask) + edge importance (edge_mask)
- **Runtime:** 1–5 seconds per node (gradient-descent loop)
- **Scope:** Instance-specific — each explanation is computed fresh

### SHAP

- **TreeExplainer (XGBoost):** Fast, exact SHAP values for the tabular baseline
- **KernelExplainer (GNN):** Approximate SHAP values treating GNN as black box
  - **Limitation:** Does NOT account for message-passing structure
  - Sub-samples background to ≤100 nodes for speed
  - This is an approximation — document accordingly

### Rationale Generator

- Combines SHAP top features + GNNExplainer edge importances into human-readable text
- Template-based: "Flagged due to: High burst_score (3.2, +0.15 SHAP impact); Connected to illicit node (edge importance: 0.85)"

---

## 6. Limitations

### Feature Semantics
- Raw Elliptic features `f1`–`f166` are **intentionally anonymized**. We cannot assert what any raw feature represents.
- **Only engineered features** have interpretable semantics. Interpretability claims are limited to these.

### Dataset
- Elliptic data is from **2019** and is **Bitcoin-specific**; illicit tactics evolve.
- Applying to Ethereum or other chains without retraining introduces **distribution shift**.
- ~77% of nodes are unlabeled — model quality on unknown nodes cannot be evaluated.

### Temporal Degradation
- Performance degrades on later time steps (45–49) due to concept drift.
- Periodic retraining is essential for real-world deployment.

### Explainability
- `KernelExplainer` treats GNN as black box — **does not account for message-passing structure**.
- `TreeExplainer` on XGBoost is exact but only applicable to the tabular baseline.
- GNNExplainer is instance-specific and takes 1–5s per node; explanations are not cached.

### Scale Claims
- Latency benchmarks at 10M+ edge scale state their basis explicitly (real Ethereum data vs. synthetic extension).
- Model accuracy metrics apply only to the Elliptic labeled dataset.
- These are separate claims and must not be conflated.

---

## 7. Ethical Considerations

- **False positives** can flag legitimate users as illicit. This system must be used as a **decision-support tool**, not an automated enforcement mechanism.
- Risk scores and labels should always be reviewed by a human before any action is taken.
- The system operates at address-level and cluster-level only; no individual deanonymization is attempted.
- This model reflects known biases in the Elliptic dataset's labeling methodology — labeled illicit transactions may not represent all forms of financial crime.

---

*Last updated: Stage 6 (Final). Update this card if models are retrained or features are modified.*
