"""
sweep_runner.py -- W&B Agent execution script for hyperparameter sweep.

Run this as:
    wandb agent your-username/onchain-fraud-gnn/SWEEP_ID
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import wandb
from src.models.graphsage import GraphSAGE
from src.models.gat import GAT
from src.models.train import train
from src.features.build_pyg import load_pyg_data

def sweep_train():
    wandb.init()
    cfg = wandb.config

    data = load_pyg_data("data/processed/features_combined.parquet")

    if cfg.model_type == "graphsage":
        model = GraphSAGE(
            in_channels=data.x.shape[1],
            hidden_channels=cfg.hidden_channels,
            out_channels=2,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            aggr=cfg.aggr,
        )
    else:
        model = GAT(
            in_channels=data.x.shape[1],
            hidden_channels=cfg.hidden_channels,
            out_channels=2,
            dropout=cfg.dropout,

        )

    os.makedirs("checkpoints", exist_ok=True)
    train(
        model=model,
        data=data,
        config=dict(cfg),
        checkpoint_path=f"checkpoints/sweep_{wandb.run.id}.pt",
        use_wandb=True,
    )

if __name__ == "__main__":

    sweep_id = wandb.sweep(
        sweep=open("src/models/sweep_config.yaml"),
        project="onchain-fraud-gnn"
    )
    wandb.agent(sweep_id, function=sweep_train, count=30)
