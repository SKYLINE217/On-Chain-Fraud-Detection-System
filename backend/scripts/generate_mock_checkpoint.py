"""
Generate a mock model checkpoint for end-to-end pipeline testing.
Creates checkpoints/best_model.pt with random weights.

Usage:
    python scripts/generate_mock_checkpoint.py
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models.graphsage import GraphSAGE


def main():
    config_path = Path("checkpoints/model_config.json")
    with open(config_path) as f:
        config = json.load(f)

    model = GraphSAGE(
        in_channels=config["in_channels"],
        hidden_channels=config["hidden_channels"],
        out_channels=config["out_channels"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "best_metrics": {
            "pr_auc_illicit": 0.0,
            "f1_illicit": 0.0,
            "precision_illicit": 0.0,
            "recall_illicit": 0.0,
            "roc_auc": 0.0,
            "epoch": 0,
        },
    }

    output_path = Path("checkpoints/best_model.pt")
    torch.save(checkpoint, output_path)
    print(f"[✓] Mock checkpoint saved to {output_path}")
    print(f"    Model: {config['model_type']}")
    print(f"    in_channels: {config['in_channels']}")
    print(f"    hidden_channels: {config['hidden_channels']}")
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
