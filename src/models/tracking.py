# src/models/tracking.py
# ──────────────────────────────────────────────────────────────────────────
# Experiment tracking utilities for W&B (Weights & Biases).
# Every baseline and GNN run in this project logs to the same W&B project
# so that comparison tables are always available in one place.
# ──────────────────────────────────────────────────────────────────────────

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ── Project-level constants ──────────────────────────────────────────────
WANDB_PROJECT = "onchain-fraud-gnn"
WANDB_ENTITY = None  # Set to your W&B team/username if needed

# ── Required metrics — every run must log these ─────────────────────────
REQUIRED_METRICS = [
    "precision_illicit",
    "recall_illicit",
    "f1_illicit",
    "pr_auc_illicit",   # ← PRIMARY metric
    "roc_auc",          # secondary, note optimism under 2% imbalance
]


def init_wandb_run(
    run_name: str,
    config: Dict[str, Any],
    tags: Optional[list] = None,
    notes: Optional[str] = None,
):
    """
    Initialize a W&B run for this project.

    Parameters
    ----------
    run_name : str
        Human-readable name (e.g. "baseline-xgb", "graphsage-sweep-42").
    config : dict
        Hyperparameters and run configuration to log.
    tags : list, optional
        Tags for filtering runs (e.g. ["baseline", "stage-1"]).
    notes : str, optional
        Free-text notes visible in the W&B dashboard.

    Returns
    -------
    wandb.Run or None
        The initialized run object, or None if W&B is unavailable.
    """
    try:
        import wandb
    except ImportError:
        logger.warning(
            "wandb not installed — metrics will be logged to console only. "
            "Install with: pip install wandb"
        )
        return None

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        config=config,
        tags=tags or [],
        notes=notes or "",
        reinit=True,
    )
    logger.info(f"W&B run initialized: {run.name} ({run.id})")
    return run


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    """
    Log a dictionary of metrics to W&B (if active) and to the console.

    Parameters
    ----------
    metrics : dict
        Metric name → value. Should include all REQUIRED_METRICS.
    step : int, optional
        Step number (epoch) for time-series logging.
    """
    # Warn if any required metric is missing
    missing = [m for m in REQUIRED_METRICS if m not in metrics]
    if missing:
        logger.warning(f"Missing required metrics: {missing}")

    # Always log to console
    metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    prefix = f"[step {step}] " if step is not None else ""
    logger.info(f"{prefix}{metrics_str}")

    # Log to W&B if available
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass


def log_config(config: Dict[str, Any]):
    """
    Update the W&B run config (e.g. after determining class weights).
    """
    try:
        import wandb
        if wandb.run is not None:
            wandb.config.update(config, allow_val_change=True)
    except ImportError:
        pass


def finish_run():
    """
    Finalize the current W&B run.
    """
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
            logger.info("W&B run finished.")
    except ImportError:
        pass
