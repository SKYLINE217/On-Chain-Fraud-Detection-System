# onchain-fraud-gnn

**On-Chain Fraud/AML Detection using Graph Neural Networks**

> ⚠ **Research/portfolio demonstration only.** This system is NOT a certified AML/CFT compliance tool and must not be used for regulatory reporting, enforcement decisions, or any purpose requiring financial regulation compliance.

---

## Overview

Graph Neural Network-based fraud detection system built on the [Elliptic Bitcoin Transaction Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) (203,769 transactions, 234,355 directed BTC flows, 49 temporal snapshots).

### Architecture
- **Neo4j 5** — Graph store with APOC + GDS plugins
- **FastAPI** — ML inference API with Pydantic validation
- **Node.js BFF** — Express proxy with JWT auth, rate limiting, Helmet.js
- **React 18** — Dashboard with graph visualization (sigma.js) + analytics (recharts)
- **Redis 7** — Sub-millisecond cache layer

### Models
- **GraphSAGE** (primary) — Inductive, generalizes to unseen nodes
- **GAT** (secondary) — Attention-based interpretability signal
- **GNNExplainer** — Subgraph + feature attribution explanations
- **Baselines** — Logistic Regression, Random Forest, XGBoost

---

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/SKYLINE217/On-Chain-Fraud-Detection-System.git
cd On-Chain-Fraud-Detection-System
cp .env.example .env  # Fill in secrets

# 2. Start infrastructure
docker compose up neo4j redis -d

# 3. Download Elliptic dataset (requires Kaggle CLI)
bash scripts/download_elliptic.sh

# 4. Load data and engineer features
python src/etl/load_neo4j.py
python src/features/engineer.py
python src/features/build_pyg.py

# 5. Start API
docker compose up fastapi bff -d

# 6. Access dashboard
open http://localhost:3000
```

See [docs/runbook.md](docs/runbook.md) for the full cold-start guide.

---

## Project Structure

```
onchain-fraud-gnn/
├── api/                    # FastAPI serving layer
│   ├── main.py             # App factory + middleware
│   ├── deps.py             # Neo4j/Redis dependency injection
│   ├── middleware/auth.py   # API key verification
│   ├── models/             # Pydantic request/response models
│   └── routers/            # wallet, cluster, health endpoints
├── src/
│   ├── etl/load_neo4j.py   # Idempotent Neo4j loader
│   ├── features/           # Feature engineering + PyG builder
│   ├── models/             # GraphSAGE, GAT definitions
│   └── serving/            # Batch scoring pipeline
├── scripts/                # Download, inflate, validate scripts
├── tests/                  # Load tests + integration tests
├── docs/                   # Architecture, runbook, data dictionary
├── frontend/               # React 18 + Node.js BFF
├── docker-compose.yml      # Full service orchestration
└── requirements.txt        # Python 3.11 dependencies
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/wallet/{address}` | Lookup risk score + metadata (Redis cached) |
| GET | `/wallet/{address}/subgraph` | 2-hop neighborhood (max 200 nodes) |
| GET | `/wallet/path/find?src=...&dst=...` | Shortest path between addresses |
| GET | `/cluster/list` | Top clusters by avg risk |
| GET | `/cluster/{id}` | Cluster detail with top 20 wallets |
| POST | `/explain/{address}` | GNNExplainer + SHAP attribution |
| GET | `/health` | Service health (Neo4j + Redis) |

---

## Documentation

- [Architecture](docs/architecture.md) — System design, service diagram, data flow
- [Runbook](docs/runbook.md) — Cold start, batch job, troubleshooting
- [Data Dictionary](docs/data_dictionary.md) — All node/edge properties, schemas

---

## Key Metrics

| Metric | Target |
|---|---|
| Primary metric | PR-AUC (illicit class) |
| Temporal split | Train ≤34, Val 35-39, Test ≥40 |
| Latency p50 /wallet | <500ms (cached) |
| Latency p95 /wallet | <5000ms |
| Scale validation | 10M+ edges (synthetic inflation) |

---

## License

This project is for research and portfolio demonstration purposes only.

---

> ⚠ **Compliance Disclaimer:** This system is a research and portfolio demonstration only. It is NOT a certified AML/CFT compliance tool, a regulated financial product, or a legally defensible fraud-detection system. It must not be used for regulatory reporting, enforcement decisions, or any purpose requiring compliance with financial regulations (BSA, FinCEN, EU AMLD, or equivalent). The authors disclaim all liability for any such use.
