import httpx
import os
from dotenv import load_dotenv

load_dotenv()

BASE = os.environ.get("FASTAPI_URL", "http://localhost:8000")
HEADERS = {"X-API-Key": os.environ.get("API_KEY", "dev_key_change_me")}
TEST_ADDRESS = "896630"  

def check_wallet():
    r = httpx.get(f"{BASE}/wallet/{TEST_ADDRESS}", headers=HEADERS)
    assert r.status_code == 200, f"wallet returned {r.status_code}: {r.text}"
    d = r.json()
    assert all(k in d for k in ["address", "risk_score", "predicted_label", "confidence", "timeStep", "communityId"]),        f"Missing keys in wallet response: {d.keys()}"
    assert isinstance(d["risk_score"], (int, float)) and 0.0 <= d["risk_score"] <= 1.0,        f"risk_score out of range: {d['risk_score']}"
    print("✅ /wallet/{address}")

def check_subgraph():
    r = httpx.get(f"{BASE}/wallet/{TEST_ADDRESS}/subgraph?hops=2", headers=HEADERS)
    assert r.status_code == 200, f"subgraph returned {r.status_code}: {r.text}"
    d = r.json()
    assert all(k in d for k in ["nodes", "edges", "hops", "node_count"]),        f"Missing keys in subgraph response: {d.keys()}"
    assert d["node_count"] <= 200, f"node_count exceeded 200: {d['node_count']}"
    assert d["hops"] <= 2, f"hops exceeded 2: {d['hops']}"
    print("✅ /wallet/{address}/subgraph")

def check_cluster_list():
    r = httpx.get(f"{BASE}/cluster/list", headers=HEADERS)
    assert r.status_code == 200, f"cluster/list returned {r.status_code}: {r.text}"
    d = r.json()
    assert isinstance(d, list), f"Expected list, got {type(d)}"
    if d:
        assert all(k in d[0] for k in ["communityId", "size", "avg_risk", "max_risk"]),            f"Missing keys in cluster response: {d[0].keys()}"
    print("✅ /cluster/list")

def check_health():
    r = httpx.get(f"{BASE}/health", headers=HEADERS)
    assert r.status_code == 200, f"health returned {r.status_code}: {r.text}"
    d = r.json()
    assert "status" in d, f"Missing 'status' in health response"
    print("✅ /health")

if __name__ == "__main__":
    check_wallet()
    check_subgraph()
    check_cluster_list()
    check_health()
    print("\n✅ All schema validations passed.")
