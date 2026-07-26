# test_api_endpoint.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.testclient import TestClient

# Mock the dependencies
from api.routers.explain import router

app = FastAPI()
app.include_router(router, prefix="/explain")

from api.middleware.auth import verify_api_key
app.dependency_overrides[verify_api_key] = lambda: "mock_api_key"

client = TestClient(app)

def test_explain_endpoint():
    # We will pick a txId that exists in our mock data
    import pandas as pd
    df = pd.read_parquet("mocks/person_a/mock_features_combined.parquet")
    sample_address = str(df["txId"].iloc[0])
    
    print(f"Testing /explain endpoint for address {sample_address}...")
    response = client.post(f"/explain/{sample_address}")
    
    if response.status_code == 200:
        print("Success! Response:")
        data = response.json()
        print(f"Address: {data['address']}")
        print(f"Rationale: {data['rationale']}")
        print(f"Model: {data['explanation_model']}")
        print("Test PASSED.")
    else:
        print(f"Failed with status {response.status_code}")
        print(response.text)
        
if __name__ == "__main__":
    test_explain_endpoint()
