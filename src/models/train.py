import os
import logging
import torch
import numpy as torch_np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score

from src.models.gnn import GraphSAGE, GAT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PYG_DATA_FILE = os.path.join(PROCESSED_DIR, "pyg_data.pt")

def evaluate_model(model, data, mask, name):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.softmax(out, dim=-1)
        preds = out.argmax(dim=-1)
        
        y_true = data.y[mask].cpu().numpy()
        y_pred = preds[mask].cpu().numpy()
        y_prob = probs[mask][:, 1].cpu().numpy()
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        pr_auc = average_precision_score(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        
        log.info(f"--- {name} Results ---")
        log.info(f"Precision: {precision:.4f}")
        log.info(f"Recall:    {recall:.4f}")
        log.info(f"F1 Score:  {f1:.4f}")
        log.info(f"PR-AUC:    {pr_auc:.4f}")
        log.info(f"ROC-AUC:   {roc_auc:.4f}")
        
        return {"precision": precision, "recall": recall, "f1": f1, "pr_auc": pr_auc, "roc_auc": roc_auc}

def train_gnn(model, data, epochs=100, lr=0.01, weight_decay=5e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Class weights for focal / cross entropy
    labels = data.y[data.train_mask]
    n_licit = (labels == 0).sum().item()
    n_illicit = (labels == 1).sum().item()
    
    weight = torch.tensor([1.0, max(1.0, n_licit / max(1, n_illicit))], dtype=torch.float32).to(data.x.device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            log.info(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")
            
    return model

def main():
    if not os.path.exists(PYG_DATA_FILE):
        log.error(f"PyG data not found at {PYG_DATA_FILE}")
        return
        
    data = torch.load(PYG_DATA_FILE, weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    in_channels = data.num_features
    out_channels = 2
    
    # Train GraphSAGE
    log.info("Training GraphSAGE...")
    sage = GraphSAGE(in_channels, 128, out_channels, num_layers=2, dropout=0.5).to(device)
    sage = train_gnn(sage, data, epochs=100, lr=0.01)
    res_sage = evaluate_model(sage, data, data.test_mask, "GraphSAGE")
    
    # Train GAT
    log.info("Training GAT...")
    gat = GAT(in_channels, 128, out_channels, num_layers=2, heads=4, dropout=0.5).to(device)
    gat = train_gnn(gat, data, epochs=100, lr=0.005)
    res_gat = evaluate_model(gat, data, data.test_mask, "GAT")
    
    # Print comparison table
    df = pd.DataFrame([res_sage, res_gat], index=["GraphSAGE", "GAT"])
    print("\n--- Model Comparison ---")
    print(df.to_markdown())

if __name__ == "__main__":
    main()
