# docs/architecture.md
# System Architecture — onchain-fraud-gnn

> **Compliance Disclaimer:** This system is a research and portfolio demonstration only. It is NOT a certified AML/CFT compliance tool, a regulated financial product, or a legally defensible fraud-detection system. It must not be used for regulatory reporting, enforcement decisions, or any purpose requiring compliance with financial regulations (BSA, FinCEN, EU AMLD, or equivalent). The authors disclaim all liability for any such use.

---

## System Overview

On-chain fraud/AML detection using Graph Neural Networks on the **Elliptic Bitcoin Transaction Dataset** (203,769 nodes, 234,355 edges, 49 temporal snapshots).

**Stack:** Neo4j (graph store) + Redis (cache) + FastAPI (ML API) + Node.js BFF (Express) + React 18 (dashboard)

---

## Service Diagram

```
                    ┌──────────────────┐
                    │   React 18 SPA   │
                    │   (Vite build)   │
                    └────────┬─────────┘
                             │ :5173 (dev) / static (prod)
                             ▼
                    ┌──────────────────┐
                    │   Node.js BFF    │
                    │  (Express :3000) │
                    │  JWT auth, rate  │
                    │  limit, proxy    │
                    └────────┬─────────┘
                             │ X-API-Key header
                             ▼
                    ┌──────────────────┐
                    │   FastAPI        │
                    │   (Uvicorn :8000)│
                    │   ML inference   │
                    └──┬────────┬──────┘
                       │        │
              ┌────────▼──┐  ┌──▼───────┐
              │  Neo4j :7687│  │Redis :6379│
              │  (graph DB) │  │ (cache)   │
              └────────────┘  └───────────┘
```

**Port exposure (production):**
- Only port **3000** (BFF) is externally accessible
- FastAPI :8000 is internal Docker network only
- Neo4j :7474/7687 is internal only
- Redis :6379 is internal only

---

## Data Flow

### 1. ETL Pipeline (Offline)
```
Kaggle Elliptic CSV → load_neo4j.py → Neo4j (203K nodes, 234K edges)
                    → engineer.py   → features_combined.parquet (203769, 171)
                    → build_pyg.py  → pyg_data.pt (PyG Data object)
```

### 2. Model Training (Person B)
```
pyg_data.pt → GraphSAGE/GAT training → best_model.pt + model_config.json
```

### 3. Nightly Batch Scoring
```
best_model.pt + pyg_data.pt → score_batch.py → Neo4j (risk_score, predicted_label, confidence, embedding)
                                             → Redis FLUSHDB (invalidate cache)
```

### 4. Real-Time Serving
```
User → React → BFF /api/wallet/:address → FastAPI → Redis cache check
                                                  → Neo4j lookup (txId_idx)
                                                  → Return WalletResponse
```

### 5. Subgraph Exploration
```
User → React → BFF /api/wallet/:address/subgraph → FastAPI → Neo4j APOC subgraphAll (max 200 nodes, max 2 hops)
```

### 6. Explainability (Person B)
```
User → React → BFF /api/explain/:address → FastAPI → GNNExplainer → SHAP features + subgraph attribution
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Graph DB | Neo4j 5 | Native graph store; APOC + GDS plugins; Cypher query language |
| GNN Framework | PyTorch Geometric | Mature; supports GraphSAGE, GAT, GNNExplainer natively |
| Primary Model | GraphSAGE (mean) | Inductive — generalizes to unseen nodes; production-ready |
| Secondary Model | GAT | Attention weights as free interpretability signal |
| API Framework | FastAPI | Async; Pydantic validation; OpenAPI auto-docs |
| BFF | Node.js Express | JWT auth; rate limiting; serves static React bundle |
| Frontend | React 18 + Vite | Fast build; sigma.js for graph viz; recharts for dashboards |
| Cache | Redis 7 | Sub-ms reads; TTL-based invalidation; flushable per batch |
| Primary Metric | PR-AUC (illicit) | Standard for extreme class imbalance (2% illicit) |
| Temporal Split | Train ≤34, Val 35-39, Test ≥40 | Non-negotiable; prevents temporal leakage |

---

## Scale Notes

- **Base dataset:** 203,769 nodes, 234,355 edges (Elliptic)
- **10M+ edge validation:** Achieved via synthetic edge inflation (SYNTHETIC_FLOW edges, Elliptic ×43)
- **Latency targets:** p50 /wallet <500ms (cached), p95 /wallet <5s, p95 /subgraph <5s
- **Batch scoring:** ~203K nodes scored in single batch run; Redis flushed after

> ⚠ **Important:** Latency benchmarks at 10M+ edge scale were validated on synthetic edge inflation. Model accuracy metrics apply only to the labeled Elliptic dataset (203,769 nodes, 234,355 FLOWS_TO edges). These are separate claims.

---

## Security Summary

| Layer | Controls |
|---|---|
| FastAPI | API key auth (X-API-Key); Pydantic validation; CORS locked to BFF |
| BFF | JWT auth for admin; rate limiting (100/min public, 5/min explain); Helmet.js; body size limits |
| Neo4j | Parameterized Cypher only; read-only app user; no external port exposure |
| Redis | Internal only; FLUSHDB after batch |
| Docker | Non-root containers; cap_drop ALL; read-only filesystems; no-new-privileges |

See [security.md](file:///d:/my%20stuff/VERSION%202/onchain-fraud-gnn/.project-docs/security.md) for the full threat model and implementation details.
