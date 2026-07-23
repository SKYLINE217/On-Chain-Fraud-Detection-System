# src/models/baselines.py
# ──────────────────────────────────────────────────────────────────────────
# Baseline tabular models for the Elliptic Bitcoin dataset.
#
# Models: Logistic Regression, Random Forest, XGBoost
# Split:  Temporal — train steps 1–34, val 35–39, test 40–49
#         (canonical split from aim.md §4, non-negotiable)
#
# Primary metric: PR-AUC on illicit class
# ──────────────────────────────────────────────────────────────────────────

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

from src.models.tracking import (
    init_wandb_run,
    log_metrics,
    log_config,
    finish_run,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────
RAW_DATA_DIR = Path("data/raw")
CHECKPOINTS_DIR = Path("checkpoints")

# Elliptic CSV filenames
FEATURES_FILE = "elliptic_txs_features.csv"
CLASSES_FILE = "elliptic_txs_classes.csv"
# (edgelist not needed for tabular baselines)

# Temporal split boundaries (from aim.md §4)
TRAIN_STEPS = range(1, 35)   # time steps 1–34
VAL_STEPS = range(35, 40)    # time steps 35–39
TEST_STEPS = range(40, 50)   # time steps 40–49

# Label encoding (from blend.md Contract 2)
# Elliptic: "1" = illicit, "2" = licit, "unknown" = unlabeled
LABEL_MAP = {"1": 1, "2": 0}  # illicit=1 (positive class), licit=0

# Feature column indices in the raw CSV
# Column 0 = txId, Column 1 = timeStep, Columns 2–167 = f1..f166
TXID_COL = 0
TIMESTEP_COL = 1
FEATURE_START_COL = 1  # We include timeStep as a feature? No — see below
RAW_FEATURE_COLS = list(range(2, 168))  # f1..f166 (166 features)


# ── Data Loading ─────────────────────────────────────────────────────────

def load_elliptic_data(
    data_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the Elliptic dataset from CSVs.

    Returns
    -------
    features : pd.DataFrame
        Columns: [txId, timeStep, f1, f2, ..., f166]
    labels : pd.DataFrame
        Columns: [txId, class]
    """
    data_dir = data_dir or RAW_DATA_DIR

    features_path = data_dir / FEATURES_FILE
    classes_path = data_dir / CLASSES_FILE

    if not features_path.exists():
        raise FileNotFoundError(
            f"Elliptic features file not found at {features_path}. "
            f"Download the dataset from Kaggle ('Elliptic Data Set') "
            f"and place CSVs in {data_dir}/"
        )
    if not classes_path.exists():
        raise FileNotFoundError(
            f"Elliptic classes file not found at {classes_path}. "
            f"Download the dataset from Kaggle ('Elliptic Data Set') "
            f"and place CSVs in {data_dir}/"
        )

    logger.info(f"Loading features from {features_path}...")
    features = pd.read_csv(features_path, header=None)
    # Assign readable column names
    col_names = ["txId", "timeStep"] + [f"f{i}" for i in range(1, 167)]
    features.columns = col_names

    logger.info(f"Loading labels from {classes_path}...")
    labels = pd.read_csv(classes_path)

    logger.info(
        f"Loaded {len(features)} nodes, "
        f"{len(labels)} labels "
        f"({(labels['class'] == '1').sum()} illicit, "
        f"{(labels['class'] == '2').sum()} licit, "
        f"{(labels['class'] == 'unknown').sum()} unknown)"
    )

    return features, labels


def prepare_splits(
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Merge features with labels, filter to labeled nodes, and split
    temporally into train/val/test.

    Returns
    -------
    splits : dict
        Keys: "train", "val", "test"
        Values: (X, y) tuples where X is the feature matrix and y is the
                label array (0=licit, 1=illicit).
    """
    # Merge
    merged = features.merge(labels, left_on="txId", right_on="txId")

    # Filter to labeled only (exclude "unknown")
    labeled = merged[merged["class"].isin(["1", "2"])].copy()
    labeled["label"] = labeled["class"].map(LABEL_MAP)

    logger.info(
        f"Labeled nodes: {len(labeled)} "
        f"(illicit: {(labeled['label'] == 1).sum()}, "
        f"licit: {(labeled['label'] == 0).sum()})"
    )

    # Feature columns: f1..f166 (the 166 anonymized features)
    feature_cols = [f"f{i}" for i in range(1, 167)]

    # Temporal split
    train = labeled[labeled["timeStep"].isin(TRAIN_STEPS)]
    val = labeled[labeled["timeStep"].isin(VAL_STEPS)]
    test = labeled[labeled["timeStep"].isin(TEST_STEPS)]

    logger.info(
        f"Split sizes — "
        f"train: {len(train)} (steps 1-34), "
        f"val: {len(val)} (steps 35-39), "
        f"test: {len(test)} (steps 40-49)"
    )
    logger.info(
        f"Class balance — "
        f"train illicit: {(train['label'] == 1).sum()} "
        f"({(train['label'] == 1).mean():.2%}), "
        f"val illicit: {(val['label'] == 1).sum()} "
        f"({(val['label'] == 1).mean():.2%})"
    )

    splits = {
        "train": (train[feature_cols].values.astype(np.float32),
                  train["label"].values),
        "val": (val[feature_cols].values.astype(np.float32),
                val["label"].values),
        "test": (test[feature_cols].values.astype(np.float32),
                 test["label"].values),
    }

    return splits


# ── Evaluation ───────────────────────────────────────────────────────────

def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str = "val",
) -> Dict[str, float]:
    """
    Evaluate a trained model on the illicit-class metrics.

    Parameters
    ----------
    model : sklearn-compatible classifier
        Must have `.predict()` and `.predict_proba()` methods.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        True labels (0=licit, 1=illicit).
    split_name : str
        Name of the split for logging purposes.

    Returns
    -------
    metrics : dict
        precision_illicit, recall_illicit, f1_illicit,
        pr_auc_illicit (PRIMARY), roc_auc
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]  # probability of illicit class

    metrics = {
        "precision_illicit": precision_score(y, y_pred, pos_label=1, zero_division=0),
        "recall_illicit": recall_score(y, y_pred, pos_label=1, zero_division=0),
        "f1_illicit": f1_score(y, y_pred, pos_label=1, zero_division=0),
        "pr_auc_illicit": average_precision_score(y, y_proba),
        "roc_auc": roc_auc_score(y, y_proba),
    }

    # Log confusion matrix
    cm = confusion_matrix(y, y_pred)
    logger.info(
        f"[{split_name}] Confusion matrix:\n"
        f"  TN={cm[0, 0]}, FP={cm[0, 1]}\n"
        f"  FN={cm[1, 0]}, TP={cm[1, 1]}"
    )

    return metrics


# ── Baseline Models ──────────────────────────────────────────────────────

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weight: str = "balanced",
) -> LogisticRegression:
    """Train a Logistic Regression baseline with balanced class weights."""
    logger.info("Training Logistic Regression...")
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info("Logistic Regression training complete.")
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    class_weight: str = "balanced",
) -> RandomForestClassifier:
    """Train a Random Forest baseline with balanced class weights."""
    logger.info("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info("Random Forest training complete.")
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    scale_pos_weight: Optional[float] = None,
) -> XGBClassifier:
    """
    Train an XGBoost baseline with scale_pos_weight for class imbalance.

    If scale_pos_weight is None, it is computed automatically as
    n_negative / n_positive (standard inverse-frequency weighting).
    """
    if scale_pos_weight is None:
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        scale_pos_weight = n_neg / n_pos
        logger.info(f"XGBoost scale_pos_weight = {scale_pos_weight:.2f}")

    logger.info("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=n_estimators,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        use_label_encoder=False,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info("XGBoost training complete.")
    return model


# ── Orchestration ────────────────────────────────────────────────────────

def run_all_baselines(
    data_dir: Optional[Path] = None,
    use_wandb: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Run all three baseline models end-to-end:
    1. Load data
    2. Temporal split
    3. Train LR, RF, XGBoost
    4. Evaluate on val set (NOT test set — reserved for Stage 3)
    5. Log all metrics to W&B

    Parameters
    ----------
    data_dir : Path, optional
        Override data directory (default: data/raw/).
    use_wandb : bool
        Whether to log to W&B (disable for tests).

    Returns
    -------
    results : dict
        Model name → metrics dict
    """
    # Load and split
    features, labels = load_elliptic_data(data_dir)
    splits = prepare_splits(features, labels)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]

    # Class balance info for logging
    n_illicit_train = (y_train == 1).sum()
    n_licit_train = (y_train == 0).sum()
    class_info = {
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_illicit_train": int(n_illicit_train),
        "n_licit_train": int(n_licit_train),
        "illicit_ratio_train": float(n_illicit_train / len(y_train)),
        "split_strategy": "temporal",
        "train_steps": "1-34",
        "val_steps": "35-39",
        "test_steps": "40-49 (untouched)",
        "n_features": X_train.shape[1],
    }

    results = {}

    # ── 1. Logistic Regression ───────────────────────────────────────────
    model_name = "Logistic Regression"
    config = {**class_info, "model_type": model_name, "class_weight": "balanced"}

    if use_wandb:
        init_wandb_run("baseline-lr", config, tags=["baseline", "stage-1"])

    lr_model = train_logistic_regression(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_val, y_val, "val")
    results[model_name] = lr_metrics

    logger.info(f"[{model_name}] Val PR-AUC: {lr_metrics['pr_auc_illicit']:.4f}")
    if use_wandb:
        log_metrics(lr_metrics)
        finish_run()

    # Save model
    _save_model(lr_model, "lr_baseline.pkl")

    # ── 2. Random Forest ─────────────────────────────────────────────────
    model_name = "Random Forest"
    config = {
        **class_info,
        "model_type": model_name,
        "n_estimators": 100,
        "class_weight": "balanced",
    }

    if use_wandb:
        init_wandb_run("baseline-rf", config, tags=["baseline", "stage-1"])

    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_val, y_val, "val")
    results[model_name] = rf_metrics

    logger.info(f"[{model_name}] Val PR-AUC: {rf_metrics['pr_auc_illicit']:.4f}")
    if use_wandb:
        log_metrics(rf_metrics)
        finish_run()

    _save_model(rf_model, "rf_baseline.pkl")

    # ── 3. XGBoost ───────────────────────────────────────────────────────
    model_name = "XGBoost"
    spw = n_licit_train / n_illicit_train
    config = {
        **class_info,
        "model_type": model_name,
        "n_estimators": 200,
        "scale_pos_weight": float(spw),
    }

    if use_wandb:
        init_wandb_run("baseline-xgb", config, tags=["baseline", "stage-1"])

    xgb_model = train_xgboost(X_train, y_train)
    xgb_metrics = evaluate_model(xgb_model, X_val, y_val, "val")
    results[model_name] = xgb_metrics

    logger.info(f"[{model_name}] Val PR-AUC: {xgb_metrics['pr_auc_illicit']:.4f}")
    if use_wandb:
        log_metrics(xgb_metrics)
        finish_run()

    _save_model(xgb_model, "xgb_baseline.pkl")

    # ── Summary ──────────────────────────────────────────────────────────
    _print_comparison_table(results)

    return results


def _save_model(model, filename: str):
    """Save a trained model checkpoint to the checkpoints directory."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINTS_DIR / filename
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {path}")


def _print_comparison_table(results: Dict[str, Dict[str, float]]):
    """Print a formatted comparison table to the console."""
    header = f"{'Model':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'PR-AUC':>10} {'ROC-AUC':>10}"
    separator = "─" * len(header)
    print(f"\n{separator}")
    print("  BASELINE COMPARISON — Val Set (Time Steps 35–39)")
    print(f"  Primary metric: PR-AUC on illicit class")
    print(f"{separator}")
    print(header)
    print(f"{'─' * 25} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10}")
    for model_name, metrics in results.items():
        print(
            f"{model_name:<25} "
            f"{metrics['precision_illicit']:>10.4f} "
            f"{metrics['recall_illicit']:>10.4f} "
            f"{metrics['f1_illicit']:>10.4f} "
            f"{metrics['pr_auc_illicit']:>10.4f} "
            f"{metrics['roc_auc']:>10.4f}"
        )
    print(separator)
    print("  Note: Test set (steps 40–49) is reserved for final evaluation (Stage 3).")
    print(f"{separator}\n")


# ── Verification helper (used by tests and manual checks) ────────────────

def load_and_split(
    data_dir: Optional[Path] = None,
    verify: bool = False,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load and split data, optionally running integrity checks.

    Parameters
    ----------
    data_dir : Path, optional
        Override data directory.
    verify : bool
        If True, run assertions that validate split integrity.

    Returns
    -------
    splits : dict
        Same as prepare_splits().
    """
    features, labels = load_elliptic_data(data_dir)
    splits = prepare_splits(features, labels)

    if verify:
        X_train, y_train = splits["train"]
        X_val, y_val = splits["val"]
        X_test, y_test = splits["test"]

        # No overlap between splits
        # (verified by construction via temporal steps, but double-check sizes)
        total_labeled = len(y_train) + len(y_val) + len(y_test)
        labeled_in_data = labels[labels["class"].isin(["1", "2"])].shape[0]
        assert total_labeled == labeled_in_data, (
            f"Split sizes don't sum to labeled count: "
            f"{total_labeled} != {labeled_in_data}"
        )

        # Only valid labels
        assert set(np.unique(y_train)).issubset({0, 1})
        assert set(np.unique(y_val)).issubset({0, 1})
        assert set(np.unique(y_test)).issubset({0, 1})

        # Feature shape
        assert X_train.shape[1] == 166, f"Expected 166 features, got {X_train.shape[1]}"
        assert X_val.shape[1] == 166
        assert X_test.shape[1] == 166

        # No NaN in features
        assert not np.isnan(X_train).any(), "NaN found in train features"
        assert not np.isnan(X_val).any(), "NaN found in val features"

        logger.info("✓ All verification checks passed.")

    return splits


# ── CLI entrypoint ───────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_all_baselines(use_wandb=True)
