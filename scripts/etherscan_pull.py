# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

"""
Etherscan API pull script — fetches transaction history for a given wallet address.
Used for the live-lookup demo path only, NOT for training data.
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY", "")
ETHERSCAN_BASE_URL = "https://api.etherscan.io/api"


def fetch_transactions(address: str, start_block: int = 0, end_block: int = 99999999) -> list:
    """
    Fetches normal (external) transaction history for a wallet address.

    Args:
        address: Ethereum wallet address (0x...).
        start_block: Starting block number.
        end_block: Ending block number.

    Returns:
        List of transaction dicts from Etherscan.
    """
    if not ETHERSCAN_API_KEY:
        raise ValueError("ETHERSCAN_API_KEY not set. Add it to .env file.")

    params = {
        "module": "account",
        "action": "txlist",
        "address": address,
        "startblock": start_block,
        "endblock": end_block,
        "sort": "desc",
        "apikey": ETHERSCAN_API_KEY,
    }

    response = requests.get(ETHERSCAN_BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    if data.get("status") != "1":
        print(f"[!] Etherscan API returned status {data.get('status')}: {data.get('message')}")
        return []

    transactions = data.get("result", [])
    print(f"[✓] Fetched {len(transactions)} transactions for {address}")
    return transactions


def fetch_internal_transactions(address: str) -> list:
    """Fetches internal (contract) transaction history for a wallet address."""
    if not ETHERSCAN_API_KEY:
        raise ValueError("ETHERSCAN_API_KEY not set. Add it to .env file.")

    params = {
        "module": "account",
        "action": "txlistinternal",
        "address": address,
        "sort": "desc",
        "apikey": ETHERSCAN_API_KEY,
    }

    response = requests.get(ETHERSCAN_BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    return data.get("result", [])


def fetch_token_transfers(address: str) -> list:
    """Fetches ERC-20 token transfer events for a wallet address."""
    if not ETHERSCAN_API_KEY:
        raise ValueError("ETHERSCAN_API_KEY not set. Add it to .env file.")

    params = {
        "module": "account",
        "action": "tokentx",
        "address": address,
        "sort": "desc",
        "apikey": ETHERSCAN_API_KEY,
    }

    response = requests.get(ETHERSCAN_BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    return data.get("result", [])


if __name__ == "__main__":
    # Test with a well-known address (Ethereum Foundation donation address)
    test_address = "0xde0B295669a9FD93d5F28D9Ec85E40f4cb697BAe"
    txs = fetch_transactions(test_address)
    print(json.dumps(txs[:3], indent=2) if txs else "No transactions returned.")
