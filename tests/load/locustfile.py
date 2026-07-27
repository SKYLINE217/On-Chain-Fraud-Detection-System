
"""
Locust load test for FastAPI endpoints.
See person_a_stages.md §3.6 for full reference.

Run:
    locust -f tests/load/locustfile.py \
      --host http://localhost:8000 \
      --users 50 --spawn-rate 5 \
      --run-time 120s --headless \
      --csv docs/load_test_results/locust_50users

Targets from system_design.md §7.3:
    p50 /wallet: <500ms (cached)
    p95 /wallet: <5000ms
    p95 /subgraph: <5000ms (20 users)

Compliance Disclaimer: This system is a research and portfolio
demonstration only. Not a certified AML/CFT compliance tool.
"""
from locust import HttpUser, task, between
import random
import os

SAMPLE_ADDRESSES = [
    "896630", "901217", "912345", "843201", "772334",
    "856321", "788912", "923456", "834567", "745678",
]

API_KEY = os.environ.get("API_KEY", "dev_key_change_me")

class WalletUser(HttpUser):
    wait_time = between(0.5, 2.0)

    def on_start(self):
        self.headers = {"X-API-Key": API_KEY}

    @task(8)
    def wallet_lookup(self):
        addr = random.choice(SAMPLE_ADDRESSES)
        self.client.get(f"/wallet/{addr}", headers=self.headers)

    @task(2)
    def subgraph_lookup(self):
        addr = random.choice(SAMPLE_ADDRESSES)
        self.client.get(f"/wallet/{addr}/subgraph?hops=2", headers=self.headers)

    @task(1)
    def health_check(self):
        self.client.get("/health", headers=self.headers)
