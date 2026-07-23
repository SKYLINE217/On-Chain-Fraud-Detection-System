# src/models/evaluate.py
# ──────────────────────────────────────────────────────────────────────────
# Stage 3: Final Model Evaluation on the Test Set
#
# This script performs the one-time evaluation on the untouched test set
# (time steps 40–49) to produce the final comparison table for eval_report.md.
#
# It expects the models to be already trained and checkpoints saved in
# the `checkpoints/` directory.
# ──────────────────────────────────────────────────────────────────────────

import json
import logging
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.baselines import load_and_split
from src.models.gat import GAT
from src.models.graphsage import GraphSAGE
from src.models.train import build_pyg_data, build_pyg_data_from_parquet

logger = logging.getLogger(__name__)

CHECKPOINTS_DIR = Path("checkpoints")
RAW_DATA_DIR = Path("data/raw")
PARQUET_PATH = Path("data/processed/features_combined.parquet")


def evaluate_tabular(model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Evaluate a tabular sklearn/xgboost model."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    return {
        "precision": float(precision_score(y, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y, y_pred, pos_label=1, zero_division=0)),
        "pr_auc": float(average_precision_score(y, y_proba)),
        "roc_auc": float(roc_auc_score(y, y_proba)),
    }


@torch.no_grad()
def evaluate_gnn(model, data, mask) -> Dict[str, float]:
    """Evaluate a PyTorch Geometric GNN model."""
    model.eval()
    out = model(data.x, data.edge_index)
    
    probs = torch.nn.functional.softmax(out[mask], dim=1)
    y_true = data.y[mask].cpu().numpy()
    y_proba = probs[:, 1].cpu().numpy()
    y_pred = probs.argmax(dim=1).cpu().numpy()
    
    return {
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def load_gnn_checkpoint(checkpoint_path: Path, model_type: str, device: torch.device):
    """Load a trained GNN from a checkpoint dict."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = ckpt["config"]
    
    if model_type == "graphsage":
        model = GraphSAGE(
            in_channels=config["in_channels"],
            hidden_channels=config["hidden_channels"],
            out_channels=config["out_channels"],
            num_layers=config.get("num_layers", 3),
            dropout=config.get("dropout", 0.3),
        )
    elif model_type == "gat":
        model = GAT(
            in_channels=config["in_channels"],
            hidden_channels=config["hidden_channels"],
            out_channels=config["out_channels"],
            heads=config.get("heads", 4),
            dropout=config.get("dropout", 0.3),
        )
    else:
        raise ValueError(f"Unknown GNN type: {model_type}")
        
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    
    return model, ckpt.get("feature_scaler")


def run_final_evaluation():
    """Run evaluation for all required model variants on the test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running evaluation on device: {device}")
    
    results = {}
    
    # ── Tabular Baselines (Raw Features) ─────────────────────────────
    logger.info("Evaluating tabular baselines (test set)...")
    try:
        splits = load_and_split(RAW_DATA_DIR, verify=False)
        X_test, y_test = splits["test"]
        
        for name, filename in [
            ("Logistic Regression", "lr_baseline.pkl"),
            ("Random Forest", "rf_baseline.pkl"),
            ("XGBoost", "xgb_baseline.pkl"),
        ]:
            path = CHECKPOINTS_DIR / filename
            if path.exists():
                with open(path, "rb") as f:
                    model = pickle.load(f)
                results[name] = evaluate_tabular(model, X_test, y_test)
                logger.info(f"✓ {name} evaluated.")
            else:
                logger.warning(f"Missing {name} checkpoint at {path}")
    except Exception as e:
        logger.error(f"Failed to evaluate tabular baselines: {e}")

    # ── GNN Models (Raw Features) ────────────────────────────────────
    logger.info("Evaluating GNNs (raw features)...")
    try:
        raw_data = build_pyg_data(RAW_DATA_DIR, device)
        mask = raw_data.test_mask
        
        # We need raw-feature trained GNNs. Assuming they were saved as {model}_raw_best.pt
        for name, prefix in [
            ("GraphSAGE (raw features)", "graphsage_raw"),
            ("GAT (raw features)", "gat_raw"),
        ]:
            path = CHECKPOINTS_DIR / f"{prefix}_best.pt"
            if path.exists():
                model, _ = load_gnn_checkpoint(path, prefix.split("_")[0], device)
                results[name] = evaluate_gnn(model, raw_data, mask)
                logger.info(f"✓ {name} evaluated.")
            else:
                logger.warning(f"Missing {name} checkpoint at {path}")
    except Exception as e:
        logger.error(f"Failed to evaluate raw-feature GNNs: {e}")

    # ── GNN Models (+ Engineered Features) ───────────────────────────
    logger.info("Evaluating GNNs (+ engineered features)...")
    if PARQUET_PATH.exists():
        try:
            edges_path = RAW_DATA_DIR / "elliptic_txs_edgelist.csv"
            # We don't use the newly fitted scaler from build_pyg_data_from_parquet, 
            # we must use the one saved in the checkpoint during training.
            # However, building the data object is easiest with the function, we'll
            # just overwrite `x` if a scaler is found in the checkpoint.
            eng_data, _ = build_pyg_data_from_parquet(PARQUET_PATH, edges_path, device)
            
            # Need original unscaled numpy features to apply the loaded scaler
            df = pd.read_parquet(PARQUET_PATH)
            feature_cols = [c for c in df.columns if c not in {"txId", "timeStep", "class"}]
            X_numpy = df[feature_cols].values.astype(np.float32)
            
            for name, prefix in [
                ("GraphSAGE (+ engineered)", "graphsage_eng"),
                ("GAT (+ engineered)", "gat_eng"),
            ]:
                path = CHECKPOINTS_DIR / f"{prefix}_best.pt"
                if path.exists():
                    model, scaler = load_gnn_checkpoint(path, prefix.split("_")[0], device)
                    
                    if scaler is not None:
                        X_scaled = scaler.transform(X_numpy)
                        eng_data.x = torch.tensor(X_scaled, dtype=torch.float32).to(device)
                        
                    results[name] = evaluate_gnn(model, eng_data, eng_data.test_mask)
                    logger.info(f"✓ {name} evaluated.")
                else:
                    logger.warning(f"Missing {name} checkpoint at {path}")
        except Exception as e:
            logger.error(f"Failed to evaluate engineered-feature GNNs: {e}")
    else:
        logger.warning(f"Engineered parquet not found at {PARQUET_PATH}. Skipping those evaluations.")

    # ── Print Final Report ───────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  FINAL EVALUATION REPORT — Test Set (Time Steps 40–49)")
    print("  Primary metric: PR-AUC on illicit class")
    print("=" * 80)
    
    header = f"{'Model':<28} | {'Precision':>9} | {'Recall':>9} | {'F1':>9} | {'PR-AUC':>9} | {'ROC-AUC':>9}"
    print(header)
    print("-" * len(header))
    
    expected_order = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "GraphSAGE (raw features)",
        "GraphSAGE (+ engineered)",
        "GAT (raw features)",
        "GAT (+ engineered)",
    ]
    
    for name in expected_order:
        if name in results:
            m = results[name]
            print(
                f"{name:<28} | "
                f"{m['precision']:>9.4f} | "
                f"{m['recall']:>9.4f} | "
                f"{m['f1']:>9.4f} | "
                f"{m['pr_auc']:>9.4f} | "
                f"{m['roc_auc']:>9.4f}"
            )
        else:
            print(f"{name:<28} | {'—':>9} | {'—':>9} | {'—':>9} | {'—':>9} | {'—':>9}")
            
    print("=" * 80 + "\n")
    
    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_final_evaluation()
