# src/models/train.py
# ──────────────────────────────────────────────────────────────────────────
# GNN training loop for node classification on the Elliptic dataset.
#
# Key design decisions (from aim.md + person_b.md):
#   - Loss mask: ONLY labeled nodes contribute to loss. Unknown nodes
#     participate in forward pass (message passing) but get zero loss.
#     This is the most common silent bug on Elliptic — verified via assert.
#   - Class-weighted CrossEntropyLoss (inverse frequency)
#   - Focal Loss as alternative if weighting underperforms
#   - Temporal split: train 1–34, val 35–39, test 40–49
#   - Primary metric: PR-AUC on illicit class
#   - Test set is NEVER touched during training — only at final evaluation
# ──────────────────────────────────────────────────────────────────────────

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from src.models.graphsage import GraphSAGE
from src.models.gat import GAT
from src.models.tracking import init_wandb_run, log_metrics, finish_run

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────
RAW_DATA_DIR = Path("data/raw")
CHECKPOINTS_DIR = Path("checkpoints")

TRAIN_STEPS = range(1, 35)   # time steps 1–34
VAL_STEPS = range(35, 40)    # time steps 35–39
TEST_STEPS = range(40, 50)   # time steps 40–49

# Label encoding (blend.md Contract 2)
LABEL_MAP = {"1": 1, "2": 0}  # illicit=1, licit=0
UNKNOWN_LABEL = -1


# ── Focal Loss ───────────────────────────────────────────────────────────

class FocalLoss(torch.nn.Module):
    """
    Focal Loss for handling severe class imbalance.
    Falls back to this if weighted CrossEntropyLoss underperforms.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    alpha : float
        Weighting factor for the positive (illicit) class.
    gamma : float
        Focusing parameter — higher gamma down-weights easy examples more.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # probability of the correct class

        # Per-sample alpha
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ── PyG Data Construction ────────────────────────────────────────────────

def build_pyg_data(
    data_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Data:
    """
    Build a PyG Data object from the raw Elliptic CSVs.

    This constructs the graph for Person B's GNN training using
    raw features only (166 features). When Person A delivers
    features_combined.parquet, use build_pyg_data_from_parquet() instead.

    Returns a Data object matching blend.md Contract 2:
        data.x:          (203769, 166) float32
        data.edge_index: (2, 234355)  long
        data.y:          (203769,)    long — 0=licit, 1=illicit, -1=unknown
        data.train_mask: bool, True for labeled nodes in steps 1–34
        data.val_mask:   bool, True for labeled nodes in steps 35–39
        data.test_mask:  bool, True for labeled nodes in steps 40–49
    """
    data_dir = data_dir or RAW_DATA_DIR

    logger.info("Building PyG Data object from raw Elliptic CSVs...")

    # ── Load features ────────────────────────────────────────────────
    features_df = pd.read_csv(data_dir / "elliptic_txs_features.csv", header=None)
    col_names = ["txId", "timeStep"] + [f"f{i}" for i in range(1, 167)]
    features_df.columns = col_names

    # ── Load labels ──────────────────────────────────────────────────
    classes_df = pd.read_csv(data_dir / "elliptic_txs_classes.csv")

    # ── Load edges ───────────────────────────────────────────────────
    edges_df = pd.read_csv(data_dir / "elliptic_txs_edgelist.csv")

    # ── Build node ID mapping (txId → contiguous integer index) ──────
    all_tx_ids = features_df["txId"].values
    tx_id_to_idx = {tx_id: idx for idx, tx_id in enumerate(all_tx_ids)}
    n_nodes = len(all_tx_ids)

    logger.info(f"Nodes: {n_nodes}")

    # ── Feature tensor ───────────────────────────────────────────────
    feature_cols = [f"f{i}" for i in range(1, 167)]
    x = torch.tensor(
        features_df[feature_cols].values, dtype=torch.float32
    )

    # ── Label tensor ─────────────────────────────────────────────────
    # Merge to get labels aligned with feature order
    merged = features_df[["txId", "timeStep"]].merge(
        classes_df, on="txId", how="left"
    )
    # Map labels: "1" → 1 (illicit), "2" → 0 (licit), "unknown"/NaN → -1
    y_values = merged["class"].map(LABEL_MAP).fillna(UNKNOWN_LABEL).astype(int)
    y = torch.tensor(y_values.values, dtype=torch.long)

    # ── Edge index ───────────────────────────────────────────────────
    # Map edge txIds to contiguous indices
    valid_edges = edges_df[
        edges_df["txId1"].isin(tx_id_to_idx) & edges_df["txId2"].isin(tx_id_to_idx)
    ]
    src = torch.tensor(
        [tx_id_to_idx[tx] for tx in valid_edges["txId1"]], dtype=torch.long
    )
    dst = torch.tensor(
        [tx_id_to_idx[tx] for tx in valid_edges["txId2"]], dtype=torch.long
    )
    edge_index = torch.stack([src, dst], dim=0)

    logger.info(f"Edges: {edge_index.shape[1]}")

    # ── Masks ────────────────────────────────────────────────────────
    time_steps = merged["timeStep"].values
    is_labeled = (y != UNKNOWN_LABEL)

    train_mask = torch.tensor(
        is_labeled.numpy() & np.isin(time_steps, list(TRAIN_STEPS)),
        dtype=torch.bool,
    )
    val_mask = torch.tensor(
        is_labeled.numpy() & np.isin(time_steps, list(VAL_STEPS)),
        dtype=torch.bool,
    )
    test_mask = torch.tensor(
        is_labeled.numpy() & np.isin(time_steps, list(TEST_STEPS)),
        dtype=torch.bool,
    )

    # ── Critical verification (aim.md §13, person_b.md §2.4) ────────
    n_labeled = is_labeled.sum().item()
    mask_total = train_mask.sum().item() + val_mask.sum().item() + test_mask.sum().item()
    assert mask_total == n_labeled, (
        f"Mask counts ({mask_total}) != labeled count ({n_labeled}). "
        f"Unknown nodes are leaking into masks — this is a critical bug!"
    )
    assert (y[train_mask] == UNKNOWN_LABEL).sum() == 0, (
        "Unknown nodes found in train_mask — critical bug!"
    )
    assert (y[val_mask] == UNKNOWN_LABEL).sum() == 0, (
        "Unknown nodes found in val_mask — critical bug!"
    )
    assert (y[test_mask] == UNKNOWN_LABEL).sum() == 0, (
        "Unknown nodes found in test_mask — critical bug!"
    )

    logger.info(
        f"Masks — train: {train_mask.sum().item()}, "
        f"val: {val_mask.sum().item()}, "
        f"test: {test_mask.sum().item()}, "
        f"unknown (no mask): {(~is_labeled).sum().item()}"
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    if device is not None:
        data = data.to(device)

    return data


def build_pyg_data_from_parquet(
    parquet_path: Path,
    edges_csv_path: Path,
    device: Optional[torch.device] = None,
) -> Tuple[Data, StandardScaler]:
    """
    Build PyG Data object from Person A's combined Parquet file (Stage 3).
    Includes StandardScaler fit on train set to prevent leakage.
    
    Returns
    -------
    data : Data
        PyG Data object.
    scaler : StandardScaler
        Fitted scaler, to be saved with the model checkpoint.
    """
    logger.info(f"Loading combined features from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # ── Verify Contract 1 ────────────────────────────────────────────
    expected_meta = {"txId", "timeStep", "class"}
    feature_cols = [c for c in df.columns if c not in expected_meta]
    logger.info(f"Loaded {len(feature_cols)} features ({df.shape[0]} nodes).")
    
    if df[feature_cols].isna().sum().sum() > 0:
        raise ValueError("NaNs found in features! Contract 1 violated.")

    # ── ID Mapping ───────────────────────────────────────────────────
    all_tx_ids = df["txId"].values
    tx_id_to_idx = {tx_id: idx for idx, tx_id in enumerate(all_tx_ids)}
    
    # ── Labels & Masks ───────────────────────────────────────────────
    y_values = df["class"].map(LABEL_MAP).fillna(UNKNOWN_LABEL).astype(int)
    y = torch.tensor(y_values.values, dtype=torch.long)
    
    time_steps = df["timeStep"].values
    is_labeled = (y != UNKNOWN_LABEL)

    train_mask = torch.tensor(is_labeled.numpy() & np.isin(time_steps, list(TRAIN_STEPS)), dtype=torch.bool)
    val_mask = torch.tensor(is_labeled.numpy() & np.isin(time_steps, list(VAL_STEPS)), dtype=torch.bool)
    test_mask = torch.tensor(is_labeled.numpy() & np.isin(time_steps, list(TEST_STEPS)), dtype=torch.bool)

    # ── Feature Scaling (Stage 3 Requirement) ────────────────────────
    # Fit scaler on train split ONLY
    logger.info("Fitting StandardScaler on train split only...")
    scaler = StandardScaler()
    
    X_numpy = df[feature_cols].values.astype(np.float32)
    scaler.fit(X_numpy[train_mask.numpy()])
    
    # Transform entire dataset
    X_scaled = scaler.transform(X_numpy)
    x = torch.tensor(X_scaled, dtype=torch.float32)

    # ── Edges ────────────────────────────────────────────────────────
    logger.info(f"Loading edges from {edges_csv_path}...")
    edges_df = pd.read_csv(edges_csv_path)
    valid_edges = edges_df[
        edges_df["txId1"].isin(tx_id_to_idx) & edges_df["txId2"].isin(tx_id_to_idx)
    ]
    src = torch.tensor([tx_id_to_idx[tx] for tx in valid_edges["txId1"]], dtype=torch.long)
    dst = torch.tensor([tx_id_to_idx[tx] for tx in valid_edges["txId2"]], dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    if device is not None:
        data = data.to(device)

    return data, scaler


# ── Training ─────────────────────────────────────────────────────────────

def compute_class_weights(data: Data) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from the training labels.
    Weight vector: [w_licit, w_illicit] where illicit is upweighted.

    From aim.md §6:
        weight = torch.tensor([n_illicit / n_licit, 1.0])
    """
    y_train = data.y[data.train_mask]
    n_licit = (y_train == 0).sum().float()
    n_illicit = (y_train == 1).sum().float()

    # Inverse frequency: upweight the minority (illicit) class
    weight = torch.tensor([n_illicit / n_licit, 1.0])

    logger.info(
        f"Class weights — licit: {weight[0]:.4f}, illicit: {weight[1]:.4f} "
        f"(n_licit={n_licit.item():.0f}, n_illicit={n_illicit.item():.0f})"
    )
    return weight


def train_epoch(
    model: torch.nn.Module,
    data: Data,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
) -> float:
    """
    Single training epoch.

    All nodes participate in the forward pass (message passing).
    Loss is computed ONLY on labeled nodes in the train mask.
    Unknown nodes contribute zero loss.
    """
    model.train()
    optimizer.zero_grad()

    # Forward pass — ALL nodes participate (including unknown)
    out = model(data.x, data.edge_index)

    # Loss — ONLY labeled nodes in train split
    loss = criterion(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data: Data,
    mask: torch.Tensor,
    split_name: str = "val",
) -> Dict[str, float]:
    """
    Evaluate the model on a given split.

    Parameters
    ----------
    model : torch.nn.Module
        Trained GNN model.
    data : Data
        PyG Data object.
    mask : torch.Tensor
        Boolean mask for the split to evaluate.
    split_name : str
        Name for logging.

    Returns
    -------
    metrics : dict
        All 5 required metrics (precision, recall, F1, PR-AUC, ROC-AUC
        on the illicit class).
    """
    model.eval()
    out = model(data.x, data.edge_index)

    # Predictions for masked nodes only
    probs = F.softmax(out[mask], dim=1)
    y_true = data.y[mask].cpu().numpy()
    y_proba = probs[:, 1].cpu().numpy()  # probability of illicit
    y_pred = probs.argmax(dim=1).cpu().numpy()

    metrics = {
        "precision_illicit": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_illicit": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_illicit": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "pr_auc_illicit": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }

    return metrics


def train_gnn(
    model_type: str = "graphsage",
    hidden_channels: int = 128,
    num_layers: int = 3,
    dropout: float = 0.3,
    heads: int = 4,
    aggr: str = "mean",
    lr: float = 0.005,
    epochs: int = 200,
    patience: int = 20,
    loss_type: str = "weighted_ce",
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    data_dir: Optional[Path] = None,
    use_parquet: bool = False,
    use_wandb: bool = True,
    device: Optional[str] = None,
) -> Tuple[torch.nn.Module, Dict[str, float]]:
    """
    Full GNN training pipeline.

    Parameters
    ----------
    model_type : str
        "graphsage" or "gat".
    hidden_channels : int
        Hidden layer dimension.
    num_layers : int
        Number of GNN layers (GraphSAGE only; GAT is fixed at 2).
    dropout : float
        Dropout probability.
    heads : int
        Number of attention heads (GAT only).
    aggr : str
        Aggregation function (GraphSAGE only: "mean" or "max").
    lr : float
        Learning rate.
    epochs : int
        Maximum training epochs.
    patience : int
        Early stopping patience (epochs without val improvement).
    loss_type : str
        "weighted_ce" or "focal".
    data_dir : Path, optional
        Override data directory.
    use_wandb : bool
        Whether to log to W&B.
    device : str, optional
        "cuda" or "cpu". Auto-detects if None.

    Returns
    -------
    model : torch.nn.Module
        Best trained model (by val PR-AUC).
    best_metrics : dict
        Best validation metrics.
    """
    # ── Device ───────────────────────────────────────────────────────
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info(f"Training on device: {device}")

    # ── Data ─────────────────────────────────────────────────────────
    data_dir = data_dir or RAW_DATA_DIR
    scaler = None
    if use_parquet:
        parquet_path = Path("data/processed/features_combined.parquet")
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found at {parquet_path}")
        edges_path = data_dir / "elliptic_txs_edgelist.csv"
        data, scaler = build_pyg_data_from_parquet(parquet_path, edges_path, device)
    else:
        data = build_pyg_data(data_dir, device)

    in_channels = data.x.shape[1]
    logger.info(f"Input features: {in_channels}")

    # ── Model ────────────────────────────────────────────────────────
    if model_type.lower() == "graphsage":
        model = GraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=2,
            num_layers=num_layers,
            dropout=dropout,
            aggr=aggr,
        )
    elif model_type.lower() == "gat":
        model = GAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=2,
            heads=heads,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {model_type}, Parameters: {total_params:,}")

    # ── Loss ─────────────────────────────────────────────────────────
    if loss_type == "weighted_ce":
        class_weight = compute_class_weights(data).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
    elif loss_type == "focal":
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # ── Optimizer ────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # ── W&B ──────────────────────────────────────────────────────────
    config = {
        "model_type": model_type,
        "hidden_channels": hidden_channels,
        "num_layers": num_layers if model_type == "graphsage" else 2,
        "dropout": dropout,
        "heads": heads if model_type == "gat" else "N/A",
        "aggr": aggr if model_type == "graphsage" else "attention",
        "lr": lr,
        "epochs": epochs,
        "patience": patience,
        "loss_type": loss_type,
        "in_channels": in_channels,
        "total_params": total_params,
        "device": str(device),
        "n_train": data.train_mask.sum().item(),
        "n_val": data.val_mask.sum().item(),
    }

    if use_wandb:
        init_wandb_run(
            f"{model_type}-train",
            config,
            tags=["gnn", "stage-2", model_type],
        )

    # ── Training loop ────────────────────────────────────────────────
    best_val_pr_auc = 0.0
    best_metrics = {}
    best_state_dict = None
    epochs_without_improvement = 0

    logger.info(f"Starting training — {epochs} max epochs, patience={patience}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train
        loss = train_epoch(model, data, optimizer, criterion)

        # Evaluate on val set
        val_metrics = evaluate(model, data, data.val_mask, "val")
        val_metrics["loss"] = loss
        val_metrics["epoch"] = epoch
        val_metrics["epoch_time"] = time.time() - t0

        # Log
        if use_wandb:
            log_metrics(val_metrics, step=epoch)

        # Early stopping on PR-AUC
        if val_metrics["pr_auc_illicit"] > best_val_pr_auc:
            best_val_pr_auc = val_metrics["pr_auc_illicit"]
            best_metrics = val_metrics.copy()
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Log progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"loss={loss:.4f} | "
                f"val PR-AUC={val_metrics['pr_auc_illicit']:.4f} | "
                f"val F1={val_metrics['f1_illicit']:.4f} | "
                f"best PR-AUC={best_val_pr_auc:.4f} | "
                f"{val_metrics['epoch_time']:.1f}s"
            )

        if epochs_without_improvement >= patience:
            logger.info(
                f"Early stopping at epoch {epoch} "
                f"(no improvement for {patience} epochs). "
                f"Best val PR-AUC: {best_val_pr_auc:.4f}"
            )
            break

    # ── Restore best model ───────────────────────────────────────────
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # ── Save checkpoint ──────────────────────────────────────────────
    _save_checkpoint(model, model_type, best_metrics, config, scaler)

    if use_wandb:
        finish_run()

    logger.info(
        f"\nTraining complete — {model_type}\n"
        f"  Best val PR-AUC: {best_val_pr_auc:.4f}\n"
        f"  Best val F1:     {best_metrics.get('f1_illicit', 0):.4f}\n"
        f"  Best epoch:      {best_metrics.get('epoch', 'N/A')}"
    )

    return model, best_metrics


def _save_checkpoint(
    model: torch.nn.Module,
    model_type: str,
    best_metrics: Dict[str, float],
    config: Dict,
    scaler: Optional[StandardScaler] = None,
):
    """
    Save the best model checkpoint and model_config.json.

    Checkpoint format matches blend.md Contract 3:
    - checkpoints/best_model.pt (or {model_type}_best.pt)
    - checkpoints/model_config.json
    """
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save state dict
    checkpoint_path = CHECKPOINTS_DIR / f"{model_type}_best.pt"
    
    ckpt_dict = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "best_metrics": best_metrics,
    }
    if scaler is not None:
        ckpt_dict["feature_scaler"] = scaler

    torch.save(ckpt_dict, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")

    # Save model config (blend.md Contract 3)
    model_config = model.get_config()
    model_config["best_val_pr_auc"] = best_metrics.get("pr_auc_illicit", 0)
    model_config["label_encoding"] = {"licit": 0, "illicit": 1, "unknown": -1}

    config_path = CHECKPOINTS_DIR / f"{model_type}_config.json"
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    logger.info(f"Model config saved to {config_path}")


# ── CLI entrypoint ───────────────────────────────────────────────────────

def main():
    """Train both GraphSAGE and GAT with default hyperparameters."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("=" * 60)
    logger.info("Training GraphSAGE...")
    logger.info("=" * 60)
    sage_model, sage_metrics = train_gnn(
        model_type="graphsage",
        hidden_channels=128,
        num_layers=3,
        dropout=0.3,
        lr=0.005,
        epochs=200,
        patience=20,
    )

    logger.info("=" * 60)
    logger.info("Training GAT...")
    logger.info("=" * 60)
    gat_model, gat_metrics = train_gnn(
        model_type="gat",
        hidden_channels=128,
        heads=4,
        dropout=0.3,
        lr=0.005,
        epochs=200,
        patience=20,
    )

    # Print comparison
    print("\n" + "=" * 60)
    print("  GNN Training Complete — Val Set Results")
    print("=" * 60)
    print(f"{'Model':<15} {'PR-AUC':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 55)
    for name, metrics in [("GraphSAGE", sage_metrics), ("GAT", gat_metrics)]:
        print(
            f"{name:<15} "
            f"{metrics.get('pr_auc_illicit', 0):>10.4f} "
            f"{metrics.get('f1_illicit', 0):>10.4f} "
            f"{metrics.get('precision_illicit', 0):>10.4f} "
            f"{metrics.get('recall_illicit', 0):>10.4f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
