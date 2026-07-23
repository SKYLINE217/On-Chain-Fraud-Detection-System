# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

"""
Locust load test for the On-Chain Fraud Detection API.

Targets:
    - Concurrency: 20 and 50 concurrent users
    - p95 < 5 seconds under 50 concurrent users
    - Endpoints tested:
        GET /wallet/{address}
        GET /wallet/{address}/subgraph?hops=2
        GET /cluster/top
        GET /cluster/{cluster_id}
        GET /path?src=...&dst=...
        GET /health

Usage:
    # Interactive UI mode (opens browser at http://localhost:8089)
    locust -f tests/locustfile.py --host=http://localhost:8000

    # Headless mode — 20 users
    locust -f tests/locustfile.py --host=http://localhost:8000 \
        --headless -u 20 -r 5 --run-time 60s \
        --csv=results/locust_20u

    # Headless mode — 50 users
    locust -f tests/locustfile.py --host=http://localhost:8000 \
        --headless -u 50 -r 10 --run-time 120s \
        --csv=results/locust_50u
"""

import random
from locust import HttpUser, task, between, tag

# Sample transaction IDs — replace with real Elliptic txIds from your loaded data
SAMPLE_TX_IDS = [
    "1", "2", "3", "100", "500", "1000", "5000", "10000",
    "50000", "100000", "150000", "200000",
]

SAMPLE_CLUSTER_IDS = [0, 1, 2, 3, 5, 10, 50, 100]


class FraudDetectionUser(HttpUser):
    """
    Simulates a user interacting with the Fraud Detection API.
    Weighted task distribution mirrors expected real traffic:
        - Wallet score lookups: most frequent (hot path, cached)
        - Subgraph queries: moderate
        - Cluster queries: moderate
        - Path queries: least frequent
    """
    wait_time = between(0.5, 2.0)

    @tag("hot-path")
    @task(10)
    def wallet_score(self):
        """GET /wallet/{address} — hot path, should be <5s (cached after first hit)."""
        tx_id = random.choice(SAMPLE_TX_IDS)
        with self.client.get(
            f"/wallet/{tx_id}",
            name="/wallet/[address]",
            catch_response=True,
        ) as response:
            if response.status_code == 404:
                response.success()  # 404 is valid for non-existent addresses
            elif response.status_code != 200:
                response.failure(f"Unexpected status: {response.status_code}")

    @tag("subgraph")
    @task(5)
    def wallet_subgraph(self):
        """GET /wallet/{address}/subgraph?hops=2 — bounded query, <5s."""
        tx_id = random.choice(SAMPLE_TX_IDS)
        hops = random.choice([1, 2])
        with self.client.get(
            f"/wallet/{tx_id}/subgraph?hops={hops}&max_nodes=200",
            name="/wallet/[address]/subgraph",
            catch_response=True,
        ) as response:
            if response.status_code == 404:
                response.success()
            elif response.status_code != 200:
                response.failure(f"Unexpected status: {response.status_code}")

    @tag("cluster")
    @task(3)
    def cluster_top(self):
        """GET /cluster/top — cluster explorer."""
        self.client.get("/cluster/top?limit=50", name="/cluster/top")

    @tag("cluster")
    @task(3)
    def cluster_detail(self):
        """GET /cluster/{cluster_id} — single cluster detail."""
        cluster_id = random.choice(SAMPLE_CLUSTER_IDS)
        with self.client.get(
            f"/cluster/{cluster_id}?limit=50",
            name="/cluster/[cluster_id]",
            catch_response=True,
        ) as response:
            if response.status_code == 404:
                response.success()
            elif response.status_code != 200:
                response.failure(f"Unexpected status: {response.status_code}")

    @tag("path")
    @task(2)
    def find_path(self):
        """GET /path?src=...&dst=... — shortest path lookup."""
        src = random.choice(SAMPLE_TX_IDS)
        dst = random.choice(SAMPLE_TX_IDS)
        with self.client.get(
            f"/path?src={src}&dst={dst}",
            name="/path",
            catch_response=True,
        ) as response:
            if response.status_code in (200, 404):
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @tag("health")
    @task(1)
    def health(self):
        """GET /health — sanity check."""
        self.client.get("/health", name="/health")


class CacheWarmupUser(HttpUser):
    """
    Secondary user class that warms up the Redis cache by hitting
    every sample address once. Useful for measuring cached vs uncached latency.
    """
    wait_time = between(0.1, 0.5)
    fixed_count = 1  # Only spawn 1 instance

    @task
    def warmup_cache(self):
        """Hit every sample address to prime the cache."""
        for tx_id in SAMPLE_TX_IDS:
            self.client.get(f"/wallet/{tx_id}", name="/wallet/[warmup]")
        self.environment.runner.quit()  # Stop after warmup
