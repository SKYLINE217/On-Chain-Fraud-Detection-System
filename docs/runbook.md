# docs/runbook.md
# Operations Runbook — onchain-fraud-gnn

> **Compliance Disclaimer:** This system is a research and portfolio demonstration only. It is NOT a certified AML/CFT compliance tool, a regulated financial product, or a legally defensible fraud-detection system. It must not be used for regulatory reporting, enforcement decisions, or any purpose requiring compliance with financial regulations (BSA, FinCEN, EU AMLD, or equivalent). The authors disclaim all liability for any such use.

---

## Cold Start

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env — fill all CHANGE_ME values with real secrets
# Generate secrets:
#   node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"   # API_KEY
#   node -e "console.log(require('crypto').randomBytes(32).toString('base64url'))"  # JWT_SECRET
#   openssl rand -base64 24  # NEO4J_PASSWORD

# 2. Start infrastructure
docker compose up neo4j redis -d
# Wait for healthchecks (30-60 seconds)
docker compose ps

# 3. Download dataset (requires Kaggle CLI + credentials)
bash scripts/download_elliptic.sh
# Or on Windows: powershell scripts/download_elliptic.ps1

# 4. Load data into Neo4j
python src/etl/load_neo4j.py

# 5. Feature engineering + GDS algorithms
python src/features/engineer.py

# 6. Build PyG Data object (handoff for Person B)
python src/features/build_pyg.py

# 7. (Person B) Train model
# python src/models/train.py
# This produces checkpoints/best_model.pt + checkpoints/model_config.json

# 8. Run batch scoring
docker compose --profile batch run --rm batch-job

# 9. Start all services
docker compose up fastapi bff -d

# 10. Access the dashboard
# Browser: http://localhost:3000
```

---

## Nightly Batch Job

The batch job scores all 203,769 nodes and writes `risk_score`, `predicted_label`, `confidence`, and `embedding` back to Neo4j, then flushes Redis.

### Cron Setup
```bash
# Add to crontab: crontab -e
# Run nightly at 2:00 AM UTC
0 2 * * * cd /path/to/onchain-fraud-gnn && \
  docker compose --profile batch run --rm batch-job >> /var/log/batch_scoring.log 2>&1
```

### Manual Run
```bash
docker compose --profile batch run --rm batch-job
```

### Validation
```bash
python scripts/validate_scores.py
# Expected: "✅ Validation passed: 203769/203769 nodes scored, 0 unscored."
```

---

## Service Management

| Service | Start | Stop | Logs |
|---|---|---|---|
| Neo4j + Redis | `docker compose up neo4j redis -d` | `docker compose stop neo4j redis` | `docker compose logs neo4j` |
| FastAPI | `docker compose up fastapi -d` | `docker compose stop fastapi` | `docker compose logs fastapi` |
| BFF | `docker compose up bff -d` | `docker compose stop bff` | `docker compose logs bff` |
| All | `docker compose up -d` | `docker compose down` | `docker compose logs -f` |
| Batch Job | `docker compose --profile batch run --rm batch-job` | (exits on completion) | stdout |

---

## Scale Disclaimer

> ⚠ **Important:** Latency benchmarks at 10M+ edge scale were validated on synthetic edge inflation (Elliptic 234K edges × 43 via `SYNTHETIC_FLOW` relationships). Model accuracy metrics apply only to the labeled Elliptic dataset (203,769 nodes, 234,355 FLOWS_TO edges). These are separate claims.

To inflate to 10M+ edges for scale testing:
```bash
python scripts/inflate_neo4j.py
```

---

## Redis Cache Strategy

| Endpoint | Cached? | Key Pattern | TTL |
|---|---|---|---|
| `/wallet/{address}` | ✅ | `score:{address}` | 3600s (1h) |
| `/wallet/{address}/subgraph` | ❌ | — | real-time Cypher |
| `/explain/{address}` | ❌ | — | GNNExplainer instance-specific |
| `/cluster/{id}` | Optional | `cluster:{id}` | 3600s |

**Cache flush:** Always `FLUSHDB` after batch scoring completes.

Manual flush:
```bash
docker compose exec redis redis-cli FLUSHDB
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Neo4j slow queries | Missing index; label scan | `EXPLAIN` query — check for `NodeByLabelScan`; rebuild indexes with `CREATE INDEX ... IF NOT EXISTS` |
| Redis cache stale | Batch job didn't flush | `redis-cli FLUSHDB` or re-run batch job with `--flush-redis` |
| CORS error in browser | ALLOWED_ORIGINS mismatch | Check `ALLOWED_ORIGINS` in `.env`; restart BFF |
| GNNExplainer timeout | Expected up to 15s | Increase client timeout; rate limit is 5/min |
| "Address not found" 404 | txId not in Neo4j | Verify ETL loaded all 203,769 nodes; check `MATCH (t:Transaction {txId: "..."}) RETURN t` |
| JWT expired | Token TTL exceeded | Re-login via `/api/admin/login`; default TTL is 8h |
| Rate limit 429 | Too many requests | Wait 60s; public limit is 100/min, explain is 5/min |
| Docker container crash | Resource limits | Check `docker compose logs <service>`; increase Docker memory allocation |
| Neo4j OOM on GDS | Graph too large for heap | Increase `NEO4J_server_memory_heap_max__size` in docker-compose env |

---

## Neo4j Index Verification

```cypher
-- Verify txId lookup uses index (not label scan)
EXPLAIN MATCH (t:Transaction {txId: "some_id"}) RETURN t
-- Expected: NodeIndexSeek on txId_idx

-- Verify timeStep index
EXPLAIN MATCH (t:Transaction) WHERE t.timeStep = 42 RETURN count(t)
-- Expected: NodeIndexSeek on timeStep_idx

-- Verify communityId index
EXPLAIN MATCH (t:Transaction {communityId: 1847}) RETURN count(t)
-- Expected: NodeIndexSeek on communityId_idx

-- List all indexes
SHOW INDEXES
```

---

## Load Testing

```bash
# Run Locust load test (50 concurrent users, 120 seconds)
locust -f tests/load/locustfile.py \
  --host http://localhost:8000 \
  --users 50 --spawn-rate 5 \
  --run-time 120s --headless \
  --csv docs/load_test_results/locust_50users
```

**Targets (from system_design.md §7.3):**
- p50 `/wallet`: <500ms (cached)
- p95 `/wallet`: <5000ms
- p95 `/subgraph`: <5000ms (20 users)
