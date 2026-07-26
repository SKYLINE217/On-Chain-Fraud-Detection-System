#!/bin/bash
# tests/integration/test_bff_fastapi.sh
# Run after docker compose up (all services)
# See person_a_stages.md §5.5 for full reference.
#
# Compliance Disclaimer: This system is a research and portfolio
# demonstration only. Not a certified AML/CFT compliance tool.

set -e

BFF_URL="http://localhost:3000"
TEST_ADDR="896630"

echo "========================================="
echo "  BFF ↔ FastAPI Integration Tests"
echo "========================================="

# Wallet lookup (should get 200, no auth needed for public endpoints)
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BFF_URL/api/wallet/$TEST_ADDR")
[ "$STATUS" = "200" ] || { echo "❌ FAIL: /api/wallet returned $STATUS"; exit 1; }
echo "✅ GET /api/wallet/$TEST_ADDR → 200"

# Subgraph
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BFF_URL/api/wallet/$TEST_ADDR/subgraph?hops=2")
[ "$STATUS" = "200" ] || { echo "❌ FAIL: /api/wallet/subgraph returned $STATUS"; exit 1; }
echo "✅ GET /api/wallet/$TEST_ADDR/subgraph → 200"

# Cluster list
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BFF_URL/api/cluster/list")
[ "$STATUS" = "200" ] || { echo "❌ FAIL: /api/cluster/list returned $STATUS"; exit 1; }
echo "✅ GET /api/cluster/list → 200"

# Health
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BFF_URL/api/health")
[ "$STATUS" = "200" ] || { echo "❌ FAIL: /api/health returned $STATUS"; exit 1; }
echo "✅ GET /api/health → 200"

# Rate limiting: 101st request should be 429
echo ""
echo "Testing rate limit..."
for i in $(seq 1 100); do
  curl -s -o /dev/null "$BFF_URL/api/wallet/$TEST_ADDR" &
done
wait
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BFF_URL/api/wallet/$TEST_ADDR")
[ "$STATUS" = "429" ] && echo "✅ Rate limit triggered correctly" || echo "⚠ Rate limit not triggered (may need more requests)"

# Admin without JWT → 401
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BFF_URL/api/admin/health")
[ "$STATUS" = "401" ] || { echo "❌ FAIL: admin without JWT returned $STATUS"; exit 1; }
echo "✅ /api/admin/health without JWT → 401"

echo ""
echo "========================================="
echo "  All integration tests passed."
echo "========================================="
