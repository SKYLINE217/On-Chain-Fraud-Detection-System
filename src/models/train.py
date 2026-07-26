"""
train.py -- Full GNN training loop with early stopping on val PR-AUC.

Key design decisions:
    - Loss masking: only compute loss on labeled training nodes (unknown excluded)
    - Class weights: inverse-frequency weighting to handle 2% illicit imbalance
    - Early stopping: patience on val PR-AUC (primary metric)
    - Gradient clipping: max_norm=1.0 to prevent exploding gradients
    - LR scheduler: ReduceLROnPlateau on val PR-AUC

Usage:
    from src.models.graphsage import GraphSAGE
    from src.models.train import train
    from src.features.build_pyg import load_pyg_data

    data = load_pyg_data()
    model = GraphSAGE(in_channels=174, hidden_channels=128, out_channels=2)
    test_metrics = train(model, data, config={"epochs": 300}, checkpoint_path="checkpoints/best_model.pt")

Compliance Disclaimer:
    This system is a research and portfolio demonstration only. It is NOT
    a certified AML/CFT compliance tool, a regulated financial product, or
    a legally defensible fraud-detection system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
)
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_class_weights(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for labeled nodes only.

    Args:
        y: Full label tensor (N,) -- values: 0 (licit), 1 (illicit), -1 (unknown)
        mask: Boolean mask -- True for labeled nodes used in this split

    Returns:
        weight tensor [w_licit, w_illicit] for CrossEntropyLoss
    """
    labeled_y = y[mask]
    n_licit = (labeled_y == 0).sum().float()
    n_illicit = (labeled_y == 1).sum().float()

    assert n_licit > 0 and n_illicit > 0, "Split contains only one class -- check mask"

    # Upweight illicit (minority class)
    # weight[0] = licit weight, weight[1] = illicit weight
    weight = torch.tensor([n_illicit / n_licit, 1.0])
    return weight


def train_epoch(
    model: nn.Module,
    data: Data,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Single training epoch."""
    model.train()
    optimizer.zero_grad()

    out = model(data.x.to(device), data.edge_index.to(device))

    # CRITICAL: Only compute loss on labeled training nodes
    # Unknown nodes (y == -1) are excluded via train_mask
    loss = criterion(
        out[data.train_mask],
        data.y[data.train_mask].to(device)
    )

    loss.backward()

    # Gradient clipping -- prevents exploding gradients on deep models
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: Data,
    mask: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Evaluate on given mask (val or test).

    Returns:
        dict with precision, recall, f1, pr_auc, roc_auc on illicit class
    """
    model.eval()
    out = model(data.x.to(device), data.edge_index.to(device))

    # Get predictions for masked nodes only
    logits_masked = out[mask].cpu()
    y_masked = data.y[mask].cpu()

    probs = F.softmax(logits_masked, dim=1).numpy()
    preds = probs.argmax(axis=1)
    y_true = y_masked.numpy()

    # Metrics on illicit class (class=1)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, preds, labels=[1], average="binary", zero_division=0
    )

    # PR-AUC (primary metric)
    pr_auc = average_precision_score(
        (y_true == 1).astype(int), probs[:, 1]
    )

    # ROC-AUC (secondary -- note optimism under 2% imbalance)
    try:
        roc_auc = roc_auc_score((y_true == 1).astype(int), probs[:, 1])
    except ValueError:
        roc_auc = 0.0  # Only one class in mask

    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "n_evaluated": int(mask.sum()),
        "n_illicit": int((y_true == 1).sum()),
        "n_licit": int((y_true == 0).sum()),
    }


def train(
    model: nn.Module,
    data: Data,
    config: dict,
    checkpoint_path: str,
    use_wandb: bool = True,
) -> dict:
    """
    Full training loop with early stopping on val PR-AUC.

    Args:
        model: GraphSAGE or GAT instance
        data: PyG Data object (all 203,769 nodes)
        config: hyperparameters dict
        checkpoint_path: where to save best_model.pt
        use_wandb: whether to log to W&B

    Returns:
        dict with best test metrics
    """
    wandb = None
    if use_wandb:
        try:
            import wandb as _wandb
            wandb = _wandb
        except ImportError:
            logger.warning("wandb not installed -- skipping experiment tracking")
            use_wandb = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Verify masks before training -- critical safety check
    assert (data.y[data.train_mask] == -1).sum() == 0, \
        "CRITICAL: Unknown nodes in train_mask -- abort training"
    assert data.train_mask.sum() + data.val_mask.sum() + data.test_mask.sum() == \
        (data.y >= 0).sum(), \
        "Mask counts don't match labeled node count"

    logger.info(f"Training on {device}")
    logger.info(
        f"Labeled nodes: train={data.train_mask.sum()}, "
        f"val={data.val_mask.sum()}, test={data.test_mask.sum()}"
    )

    # Compute class weights from training split
    weight = compute_class_weights(data.y, data.train_mask).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("lr", 0.001),
        weight_decay=config.get("weight_decay", 5e-4),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6
    )

    best_val_pr_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    max_patience = config.get("patience", 30)

    for epoch in range(1, config.get("epochs", 300) + 1):
        train_loss = train_epoch(model, data, optimizer, criterion, device)
        val_metrics = evaluate(model, data, data.val_mask, device)
        val_pr_auc = val_metrics["pr_auc"]

        scheduler.step(val_pr_auc)

        if use_wandb and wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_pr_auc": val_pr_auc,
                "val_f1": val_metrics["f1"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_roc_auc": val_metrics["roc_auc"],
                "lr": optimizer.param_groups[0]["lr"],
            })

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            patience_counter = 0
            # Save checkpoint
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(
                f"Epoch {epoch}: New best val PR-AUC={val_pr_auc:.4f} -- saved checkpoint"
            )
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(
                    f"Early stopping at epoch {epoch} (best was epoch {best_epoch})"
                )
                break

        if epoch % 10 == 0:
            logger.info(
                f"Epoch {epoch} | Loss: {train_loss:.4f} | "
                f"Val PR-AUC: {val_pr_auc:.4f} | F1: {val_metrics['f1']:.4f}"
            )

    # Final evaluation on test set (touch ONCE only)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    test_metrics = evaluate(model, data, data.test_mask, device)

    logger.info("=" * 50)
    logger.info("FINAL TEST RESULTS (time steps 40-49)")
    logger.info(f"PR-AUC: {test_metrics['pr_auc']:.4f}")
    logger.info(f"F1:     {test_metrics['f1']:.4f}")
    logger.info(f"Prec:   {test_metrics['precision']:.4f}")
    logger.info(f"Recall: {test_metrics['recall']:.4f}")

    if use_wandb and wandb:
        wandb.summary.update({
            "best_val_pr_auc": best_val_pr_auc,
            "best_epoch": best_epoch,
            "test_pr_auc": test_metrics["pr_auc"],
            "test_f1": test_metrics["f1"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
        })

    return test_metrics
