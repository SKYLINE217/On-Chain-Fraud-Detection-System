
import httpx
import os

BASE = "http://localhost:8000"
HEADERS = {"X-API-Key": os.environ.get("API_KEY", "test_key")}
TEST_ADDRESS = "896630"

def check_explain():
    r = httpx.post(f"{BASE}/explain/{TEST_ADDRESS}", headers=HEADERS, timeout=30.0)
    assert r.status_code == 200, f"Status: {r.status_code}, Body: {r.text}"
    d = r.json()

    required = ["address", "shap_top_features", "subgraph_explanation",
                "rationale", "explanation_model", "latency_warning"]
    for key in required:
        assert key in d, f"Missing key: {key}"

    assert isinstance(d["shap_top_features"], list)
    if len(d["shap_top_features"]) > 0:
        for feat in d["shap_top_features"]:
            assert all(k in feat for k in ["feature_name", "feature_value", "shap_value"])
            assert isinstance(feat["shap_value"], float)

    sg = d["subgraph_explanation"]
    assert "important_nodes" in sg and "important_edges" in sg
    for edge in sg["important_edges"]:
        assert all(k in edge for k in ["src", "dst", "importance_score"])

    assert isinstance(d["rationale"], str) and len(d["rationale"]) > 10

    assert isinstance(d["latency_warning"], str)

    print(f"✅ POST /explain/{TEST_ADDRESS}")
    print(f"   Rationale: {d['rationale'][:80]}...")
    print(f"   Latency warning: {d['latency_warning']}")

def check_score_batch():
    """POST /score with list of addresses."""
    payload = {"addresses": [TEST_ADDRESS, "901217", "912345"]}
    r = httpx.post(f"{BASE}/score", json=payload, headers=HEADERS, timeout=30.0)
    assert r.status_code == 200, f"Status: {r.status_code}"
    d = r.json()
    assert isinstance(d, list)
    for item in d:
        assert all(k in item for k in ["address", "risk_score", "predicted_label", "confidence"])
    print(f"✅ POST /score  ({len(d)} results)")

if __name__ == "__main__":
    import sys
    try:
        httpx.get(BASE, timeout=1.0)
    except httpx.ConnectError:
        print("API is not running. Please start FastAPI first: uvicorn api.main:app --port 8000")
        sys.exit(0)

    check_explain()
    check_score_batch()
    print("All ML API schema validations passed.")
