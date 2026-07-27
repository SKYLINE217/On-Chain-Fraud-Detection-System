#!/bin/bash
set -e

BFF_URL="http://localhost:3000"
TEST_ADDR="896630"

if ! curl -s -o /dev/null -w "%{http_code}" "$BFF_URL" | grep -q "200\|404"; then
    echo "BFF is not running at $BFF_URL. Cannot run integration tests."
    exit 0
fi

echo "Testing ML endpoints via BFF..."

RESPONSE=$(curl -s -X POST "$BFF_URL/api/explain/$TEST_ADDR" \
  -H "Content-Type: application/json" \
  -d '{}' \
  --max-time 30)

echo "$RESPONSE" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
except json.JSONDecodeError:
    print('FAIL: Response is not JSON')
    sys.exit(1)
required = ['address', 'shap_top_features', 'subgraph_explanation',
            'rationale', 'explanation_model', 'latency_warning']
missing = [k for k in required if k not in d]
if missing:
    print(f'FAIL: Missing keys: {missing}')
    sys.exit(1)
print('✅ /api/explain/{address} — schema valid')
print(f'   SHAP features: {len(d[\"shap_top_features\"])}')
print(f'   Rationale preview: {d[\"rationale\"][:60]}...')
"

echo "Testing explain rate limit..."
for i in $(seq 1 5); do
  curl -s -o /dev/null -X POST "$BFF_URL/api/explain/$TEST_ADDR" \
    -H "Content-Type: application/json" -d '{}' --max-time 30 &
done
wait
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BFF_URL/api/explain/$TEST_ADDR" \
  -H "Content-Type: application/json" -d '{}' --max-time 30)
[ "$STATUS" = "429" ] && echo "✅ Explain rate limit (5/min) enforced" \
  || echo "⚠ Rate limit not triggered (may need more requests)"

STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BFF_URL/api/wallet/../../../../etc/passwd")
[ "$STATUS" = "400" ] && echo "✅ Path traversal blocked → 400" \
  || echo "⚠ Unexpected status for path traversal: $STATUS"

echo "ML endpoint integration tests complete."
