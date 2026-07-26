# Demo Script (approx. 5 min)

## 1. Overview Tab (30s)
- Open `http://localhost:3000`
- Show headline metrics: total wallets, high-risk count, last scored timestamp
- Point out risk distribution histogram — 94% low risk, 1.4% high risk
- Disclaimer banner visible

## 2. Wallet Lookup (90s)
- Search for a known high-risk address (use one from `tests/integration_addresses.json`)
- Show: red RiskGauge, ILLICIT badge, confidence %, community ID
- Show SHAP bar chart: top features in red (positive push toward illicit)
- Click "Explain This Wallet"
- Wait 5-15s (explain spinner visible)
- Show GraphCanvas: red nodes = illicit neighbors, edge widths = importance
- Read rationale aloud

## 3. Cluster Explorer (60s)
- Navigate to `/clusters`
- Sort by `avg_risk` — show top community
- Click "View" → slide-in drawer with 20 highest-risk wallets
- Click "View in Wallet Lookup" for one wallet

## 4. Transaction Path (60s)
- Navigate to `/path`
- Enter two addresses (use known connected pair)
- Show shortest path: 4 hops, colored by risk

## 5. Admin Dashboard (60s)
- Navigate to `/admin/login`
- Login with admin credentials
- Show System Health: all services green, latency chart
- Show Model Registry: deployed model stats, PR curve overlay
- Show Batch Job Manager: last run timestamp, log viewer

## 6. Close
- *"This is a research demonstration — not for regulatory use"*
