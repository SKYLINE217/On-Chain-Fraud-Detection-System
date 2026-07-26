# scripts/pull_etherscan.py
"""
Pull BTC amount proxies from Etherscan for a sample of txIds.
Only needed if amount features are missing from Elliptic raw features.
Elliptic f-features include aggregated amounts — check before running.

NOTE: Elliptic dataset txIds are anonymized hashes — Etherscan enrichment
may not resolve. Use as-is; the engineered `amount_mean`/`amount_skew`
features fall back to Elliptic f-feature aggregates if Etherscan returns
no data.

Usage:
    python scripts/pull_etherscan.py

Compliance Disclaimer: This system is a research and portfolio
demonstration only. Not a certified AML/CFT compliance tool.
"""
import requests
import time
import os
import pandas as pd

ETHERSCAN_API_KEY = os.environ.get("ETHERSCAN_API_KEY", "")
BASE_URL = "https://api.etherscan.io/api"
RATE_LIMIT_SLEEP = 0.25  # 4 req/s on free tier


def get_tx_value(session: requests.Session, tx_hash: str) -> float | None:
    """Fetch ETH value for a transaction hash. Returns None on failure."""
    try:
        r = session.get(BASE_URL, params={
            "module": "proxy",
            "action": "eth_getTransactionByHash",
            "txhash": tx_hash,
            "apikey": ETHERSCAN_API_KEY,
        }, timeout=(3.0, 10.0)) # connect timeout 3s, read timeout 10s
        data = r.json().get("result")
        if data and data.get("value"):
            return int(data["value"], 16) / 1e18  # Wei → ETH
    except Exception:
        pass
    return None


def enrich_sample(txids: list[str], output_path: str = "data/raw/etherscan_amounts.csv"):
    """Pull amounts for a sample and save to CSV."""
    results = []
    with requests.Session() as session:
        for i, tx in enumerate(txids):
            val = get_tx_value(session, tx)
            results.append({"txId": tx, "eth_value": val})
            time.sleep(RATE_LIMIT_SLEEP)
            if i % 100 == 0:
                print(f"  {i}/{len(txids)} pulled")

    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Saved {len(results)} rows to {output_path}")


if __name__ == "__main__":
    # Load a sample of txIds from the features CSV
    features_path = "data/raw/elliptic_txs_features.csv"
    if not os.path.exists(features_path):
        print(f"ERROR: {features_path} not found. Run download_elliptic.sh first.")
        exit(1)

    df = pd.read_csv(features_path, header=None, usecols=[0], names=["txId"])
    # Sample 100 txIds for enrichment (adjust as needed)
    sample = df["txId"].head(100).tolist()
    print(f"Pulling Etherscan data for {len(sample)} txIds...")
    enrich_sample(sample)
