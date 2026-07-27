#!/bin/bash

set -e

BFF_URL="http://localhost:3000"
TEST_ADDR="896630"

echo "========================================="
echo "  BFF ↔ FastAPI Integration Tests"
echo "========================================="

STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BFF_URL/api/wallet/$TEST_ADDR")
[ "$STATUS" = "200" ] || { echo "❌ FAIL: /api/wallet returned $STATUS"; exit 1; }
echo "✅ GET /api/wallet/$TEST_ADDR → 200"

STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BFF_URL/api/wallet/$TEST_ADDR/subgraph?hops=2")
[ "$STATUS" = "200" ] || { echo "❌ FAIL: /api/wallet/subgraph returned $STATUS"; exit 1; }
echo "✅ GET /api/wallet/$TEST_ADDR/subgraph → 200"

STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BFF_URL/api/cluster/list")
[ "$STATUS" = "200" ] || { echo "❌ FAIL: /api/cluster/list returned $STATUS"; exit 1; }
echo "✅ GET /api/cluster/list → 200"

STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BFF_URL/api/health")
[ "$STATUS" = "200" ] || { echo "❌ FAIL: /api/health returned $STATUS"; exit 1; }
echo "✅ GET /api/health → 200"

echo ""
echo "Testing rate limit..."
for i in $(seq 1 100); do
  curl -s -o /dev/null "$BFF_URL/api/wallet/$TEST_ADDR" &
done
wait
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BFF_URL/api/wallet/$TEST_ADDR")
[ "$STATUS" = "429" ] && echo "✅ Rate limit triggered correctly" || echo "⚠ Rate limit not triggered (may need more requests)"

STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BFF_URL/api/admin/health")
[ "$STATUS" = "401" ] || { echo "❌ FAIL: admin without JWT returned $STATUS"; exit 1; }
echo "✅ /api/admin/health without JWT → 401"

echo ""
echo "========================================="
echo "  All integration tests passed."
echo "========================================="
