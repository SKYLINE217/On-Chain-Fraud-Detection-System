"""
export_predictions.py — Train GNN and export predictions to CSV for Neo4j serving.
"""

import os
import logging
import torch
import pandas as pd

from src.models.gnn import GAT
from src.models.train import train_gnn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PYG_DATA_FILE = os.path.join(PROCESSED_DIR, "pyg_data.pt")
FEATURES_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "elliptic_txs_features.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "predictions.csv")

def main():
    if not os.path.exists(PYG_DATA_FILE):
        log.error(f"PyG data not found at {PYG_DATA_FILE}")
        return
        
    data = torch.load(PYG_DATA_FILE, weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    in_channels = data.num_features
    out_channels = 2
    
    log.info("Training GAT for prediction export...")
    gat = GAT(in_channels, 128, out_channels, num_layers=2, heads=4, dropout=0.5).to(device)
    gat = train_gnn(gat, data, epochs=100, lr=0.005)
    
    log.info("Generating predictions for all nodes...")
    gat.eval()
    with torch.no_grad():
        out = gat(data.x, data.edge_index)
        probs = torch.softmax(out, dim=-1)
        preds = out.argmax(dim=-1)
        
        # Risk score is the probability of class 1 (illicit)
        risk_scores = probs[:, 1].cpu().numpy()
        confidence = probs.max(dim=-1).values.cpu().numpy()
        predicted_labels = preds.cpu().numpy()
        
    features = pd.read_csv(FEATURES_FILE, header=None, usecols=[0])
    tx_ids = features[0].astype(str).values
    
    assert len(tx_ids) == len(risk_scores), "Mismatch in node count between raw CSV and PyG data"
    
    log.info(f"Exporting {len(tx_ids)} predictions to {OUTPUT_FILE}...")
    
    label_map = {0: "licit", 1: "illicit"}
    pred_str = [label_map[p] for p in predicted_labels]
    
    df = pd.DataFrame({
        "txId": tx_ids,
        "risk_score": risk_scores,
        "confidence": confidence,
        "predicted_label": pred_str
    })
    
    df.to_csv(OUTPUT_FILE, index=False)
    log.info("✓ Predictions exported successfully.")

if __name__ == "__main__":
    main()
