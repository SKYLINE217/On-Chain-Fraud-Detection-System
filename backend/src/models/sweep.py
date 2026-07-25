# src/models/sweep.py
# ──────────────────────────────────────────────────────────────────────────
# W&B Sweep runner for GNN models.
#
# Called by the W&B agent to run a single sweep configuration.
# Usage (run via W&B agent, not directly):
#   wandb sweep src/models/sweep_config_graphsage.yaml
#   wandb agent <sweep_id>
# ──────────────────────────────────────────────────────────────────────────

import logging
import wandb
from src.models.train import train_gnn

logger = logging.getLogger(__name__)

def sweep_train():
    """Initialize W&B sweep run and launch training."""
    # Initialize a new W&B run from the sweep agent
    # config will be populated automatically by W&B from the sweep definition
    wandb.init()
    config = wandb.config

    logger.info(f"Starting sweep run with config: {dict(config)}")

    model_type = config.model_type
    
    # Extract GraphSAGE specific params
    num_layers = config.get("num_layers", 3)
    aggr = config.get("aggr", "mean")
    
    # Extract GAT specific params
    heads = config.get("heads", 4)

    # Common params
    hidden_channels = config.get("hidden_channels", 128)
    dropout = config.get("dropout", 0.3)
    lr = config.get("lr", 0.005)
    loss_type = config.get("loss_type", "weighted_ce")
    epochs = config.get("epochs", 200)
    patience = config.get("patience", 20)

    try:
        model, metrics = train_gnn(
            model_type=model_type,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            heads=heads,
            aggr=aggr,
            lr=lr,
            epochs=epochs,
            patience=patience,
            loss_type=loss_type,
            use_wandb=False, # We initialized W&B manually above
        )
        # Log final best metrics to the sweep run
        wandb.log({
            "val_pr_auc_illicit": metrics.get("pr_auc_illicit", 0.0),
            "val_f1_illicit": metrics.get("f1_illicit", 0.0),
            "best_epoch": metrics.get("epoch", 0)
        })
    except Exception as e:
        logger.error(f"Sweep run failed: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    sweep_train()
