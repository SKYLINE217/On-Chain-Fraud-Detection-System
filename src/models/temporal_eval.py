"""
temporal_eval.py -- Evaluate F1 per time step to reveal temporal degradation.
"""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import os
import argparse
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@torch.no_grad()
def evaluate_per_timestep(
    model: torch.nn.Module,
    data: Data,
    device: torch.device,
    time_steps: range = range(35, 50),
) -> dict:
    """
    Evaluate F1 per time step (val+test range: 35-49).
    Reveals temporal degradation from distribution shift.
    """
    model.eval()
    out = model(data.x.to(device), data.edge_index.to(device)).cpu()
    probs = F.softmax(out, dim=1).numpy()
    preds = probs.argmax(axis=1)

    results = {}
    for t in time_steps:
        mask = (data.time_step == t) & (data.y >= 0)
        if mask.sum() == 0:
            continue
        y_t = data.y[mask].numpy()
        p_t = preds[mask]
        f1 = f1_score(y_t, p_t, labels=[1], average="binary", zero_division=0)
        n_illicit = (y_t == 1).sum()
        results[t] = {"f1": float(f1), "n_illicit": int(n_illicit), "n_total": int(mask.sum())}
        
    return results

def plot_f1_over_time(results: dict, output_path: str = "docs/figures/f1_over_time.png"):
    steps = sorted(results.keys())
    f1_vals = [results[t]["f1"] for t in steps]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, f1_vals, marker="o", color="#ef4444", linewidth=2)
    ax.axvline(x=39.5, color="#6b7280", linestyle="--", label="Val/Test boundary")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("F1 (Illicit Class)")
    ax.set_title("F1 Score over Time Steps -- GNN on Elliptic")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    args = parser.parse_args()
    
    from src.features.build_pyg import load_pyg_data
    from src.models.graphsage import GraphSAGE
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading mock data for temporal evaluation...")
    # Because we don't know if real parquet exists, try to load mock if real doesn't exist.
    # Actually load_pyg_data has mock_edges handling, but for features we should just let it load.
    try:
        data = load_pyg_data()
    except FileNotFoundError:
        print("Using mock data")
        data = load_pyg_data(parquet_path="mocks/person_a/mock_features_combined.parquet", mock_edges=True)
    
    model = GraphSAGE(in_channels=data.x.shape[1], hidden_channels=128, out_channels=2, num_layers=3)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    model.to(device)
    
    results = evaluate_per_timestep(model, data, device)
    out_path = plot_f1_over_time(results)
    print(f"Saved temporal evaluation plot to {out_path}")
