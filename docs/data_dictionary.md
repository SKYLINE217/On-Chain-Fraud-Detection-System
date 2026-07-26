# docs/data_dictionary.md
# Data Dictionary — onchain-fraud-gnn

> **Compliance Disclaimer:** This system is a research and portfolio demonstration only. It is NOT a certified AML/CFT compliance tool, a regulated financial product, or a legally defensible fraud-detection system. It must not be used for regulatory reporting, enforcement decisions, or any purpose requiring compliance with financial regulations (BSA, FinCEN, EU AMLD, or equivalent). The authors disclaim all liability for any such use.

---

## Dataset: Elliptic Bitcoin Transaction Dataset

| Property | Value |
|---|---|
| Nodes | 203,769 (transactions) |
| Edges | 234,355 (directed BTC flows) |
| Raw features | 166 per node (f1–f94: local; f95–f166: 1-hop aggregated; all anonymized) |
| Engineered features | 8 additional (see below) |
| Total features (N) | 174 (166 raw + 8 engineered) |
| Class distribution | 2% illicit / 21% licit / 77% unknown |
| Labeled nodes | ~23% of total (~47,000) |
| Temporal snapshots | 49 time steps |

---

## Node: Transaction

### Raw Features (from Elliptic)

| Property | Type | Source | Description | Range |
|---|---|---|---|---|
| txId | string | Elliptic | Anonymized transaction identifier | Unique string |
| timeStep | int | Elliptic | Temporal snapshot (1–49) | [1, 49] |
| class | string | Elliptic | "1"=illicit, "2"=licit, "unknown" | {1, 2, unknown} |
| f1..f94 | float32 | Elliptic | Local transaction features (anonymized) | varies |
| f95..f166 | float32 | Elliptic | 1-hop aggregated features (anonymized) | varies |

### Engineered Features

| Property | Type | Source | Description | Range | Notes |
|---|---|---|---|---|---|
| tx_freq | float32 | Engineered | In+out degree per node | [0, ∞) | Higher = more connected |
| amount_mean | float32 | Engineered | Mean BTC amount proxy (from f93, per timeStep) | varies | Falls back to f-feature aggregates if Etherscan unavailable |
| amount_skew | float32 | Engineered | Skewness of amounts (per timeStep) | (-∞, ∞) | High skew = unusual distribution |
| address_age | float32 | Engineered | Timestep of first appearance | [1, 49] | Lower = older address |
| clustering_coeff | float32 | GDS | Local clustering coefficient | [0.0, 1.0] | Higher = more triangles in neighborhood |
| burst_score | float32 | Engineered | Z-score of tx count vs trailing avg | (-∞, ∞) | High = sudden spike in activity |
| pageRank | float32 | GDS | PageRank centrality | [0.0, 1.0] | Higher = more central in flow network |
| communityId | int | GDS | Louvain community ID | [0, N) | Community membership; may drift on GDS re-runs |

> ⚠ **communityId schema drift risk:** Louvain community IDs are non-deterministic across GDS runs. Never re-run GDS separately from the parquet export. Always run `engineer.py` end-to-end.

### Model-Scored Properties (from batch job)

| Property | Type | Source | Description | Range |
|---|---|---|---|---|
| risk_score | float32 | Model (batch) | Illicit probability P(class=1) | [0.0, 1.0] |
| predicted_label | string | Model (batch) | Classification result | {"illicit", "licit", "unknown"} |
| confidence | float32 | Model (batch) | max(softmax probabilities) | [0.5, 1.0] |
| embedding | float[] | Model (batch) | Final hidden layer output | dim=128, floats |

---

## Edge: FLOWS_TO

| Property | Type | Source | Description |
|---|---|---|---|
| (none) | — | Elliptic | Directed BTC transaction flow (src → dst) |

## Edge: SYNTHETIC_FLOW (Scale Testing Only)

| Property | Type | Source | Description |
|---|---|---|---|
| (none) | — | inflate_neo4j.py | Random edges for 10M+ scale benchmarking |

> ⚠ **SYNTHETIC_FLOW edges** are for latency benchmarking only. They do not represent real transactions and are excluded from all model accuracy metrics.

---

## Known Failure Modes

| Issue | Impact | Mitigation |
|---|---|---|
| Temporal distribution shift (timeStep 40+) | F1 degrades on later time steps | Per-timestep evaluation; document expected degradation |
| Class imbalance (2% illicit) | High FP rate possible | Class-weighted loss; PR-AUC as primary metric |
| Anonymized features | Cannot interpret f1–f166 | SHAP on engineered features; GNNExplainer for subgraph |
| communityId non-determinism | IDs change if GDS re-run | Always export parquet immediately after GDS |
| Unknown nodes (77%) | Cannot be used for training | Excluded via train_mask; scored but unlabeled in production |

---

## Parquet Schema

**File:** `data/processed/features_combined.parquet`
**Shape:** (203769, 171)

| Column | Type |
|---|---|
| txId | string |
| timeStep | int |
| class | string |
| f1..f166 | float32 (166 columns) |
| tx_freq | float32 |
| amount_mean | float32 |
| amount_skew | float32 |
| address_age | float32 |
| clustering_coeff | float32 |
| burst_score | float32 |
| pageRank | float32 |
| communityId | int |

**Total columns:** 3 metadata + 166 raw + 8 engineered = **171**
