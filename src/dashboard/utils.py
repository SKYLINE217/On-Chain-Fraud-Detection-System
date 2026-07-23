import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Cache to simulate Redis
@st.cache_data
def load_mock_data():
    """Load the raw feature CSV to mock the backend for the prototype."""
    data_dir = Path("data/raw")
    features_path = data_dir / "elliptic_txs_features.csv"
    classes_path = data_dir / "elliptic_txs_classes.csv"
    
    if not features_path.exists() or not classes_path.exists():
        # Return empty dummy if not available locally
        return pd.DataFrame(), pd.DataFrame()
        
    features = pd.read_csv(features_path, header=None)
    classes = pd.read_csv(classes_path)
    
    # Rename for sanity
    features.columns = ["txId", "timeStep"] + [f"f{i}" for i in range(1, 167)]
    
    # Merge classes
    df = features.merge(classes, on="txId", how="left")
    return df

def get_mock_risk_score(tx_id: str, df: pd.DataFrame) -> float:
    """Return a mock risk score based on the true class for demonstration."""
    if df.empty:
        return 0.5
        
    row = df[df["txId"] == tx_id]
    if row.empty:
        return np.random.uniform(0, 1)
        
    label = row.iloc[0]["class"]
    if label == "1": # illicit
        return np.random.uniform(0.75, 0.99)
    elif label == "2": # licit
        return np.random.uniform(0.01, 0.35)
    else:
        return np.random.uniform(0.4, 0.6)

def get_mock_shap_values():
    """Return mock SHAP values for the horizontal bar chart."""
    features = ["burst_score", "tx_freq", "amount_skew", "f42", "f91", "clustering_coeff", "f12", "address_age", "f101", "f33"]
    values = np.random.uniform(-0.5, 0.8, 10)
    # Sort by absolute value descending
    indices = np.argsort(np.abs(values))
    return [features[i] for i in indices], [values[i] for i in indices]
