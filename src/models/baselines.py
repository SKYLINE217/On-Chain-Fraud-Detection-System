import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
    precision_recall_curve,
)
from xgboost import XGBClassifier
import logging
from typing import Optional

logger = logging.getLogger(__name__)

FEATURE_COLS = [f"f{i}" for i in range(1, 167)] + [
    "tx_freq", "amount_mean", "amount_skew", "address_age",
    "clustering_coeff", "burst_score", "pageRank",

]

def load_and_split(parquet_path: str) -> tuple:
    """
    Load features_combined.parquet and apply temporal split.

    Split (non-negotiable per Elliptic benchmark standard):
        Train: timeStep 1-34  (labeled only)
        Val:   timeStep 35-39 (labeled only)
        Test:  timeStep 40-49 (labeled only)

    Args:
        parquet_path: Path to features_combined.parquet.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    df = pd.read_parquet(parquet_path)

    assert df.shape == (203769, 177), f"Unexpected shape: {df.shape}"
    assert df[FEATURE_COLS].isna().sum().sum() == 0, "NaNs found in features"

    labeled = df[df["class"] != "unknown"].copy()
    labeled["y"] = (labeled["class"] == "1").astype(int)  

    train = labeled[labeled["timeStep"] <= 34]
    val = labeled[(labeled["timeStep"] >= 35) & (labeled["timeStep"] <= 39)]
    test = labeled[labeled["timeStep"] >= 40]

    logger.info(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    logger.info(f"Train illicit rate: {train['y'].mean():.3f}")

    return (
        train[FEATURE_COLS].values, train["y"].values,
        val[FEATURE_COLS].values, val["y"].values,
        test[FEATURE_COLS].values, test["y"].values,
    )

def eval_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    model_name: str,
) -> dict:
    """
    Compute evaluation metrics for illicit class detection.

    Args:
        y_true: Ground truth labels (0=licit, 1=illicit).
        probs: Predicted probabilities for illicit class.
        model_name: Name for logging.

    Returns:
        Dict with precision, recall, f1, pr_auc on illicit class.
    """
    preds = (probs >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, preds, labels=[1], average="binary", zero_division=0
    )
    pr_auc = average_precision_score(y_true, probs)

    metrics = {
        "model": model_name,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
    }
    logger.info(
        f"{model_name} → PR-AUC: {pr_auc:.4f}  F1: {f1:.4f}  "
        f"P: {prec:.4f}  R: {rec:.4f}"
    )
    return metrics

def run_all_baselines(
    parquet_path: str,
    use_wandb: bool = True,
) -> tuple[list[dict], object]:
    """
    Train and evaluate all baseline models.

    Models:
        1. Logistic Regression (class_weight="balanced")
        2. Random Forest (200 trees, class_weight="balanced")
        3. XGBoost (scale_pos_weight, early stopping on val PR-AUC)

    Args:
        parquet_path: Path to features_combined.parquet.
        use_wandb: Whether to log to Weights & Biases.

    Returns:
        Tuple of (results_list, xgboost_model).
        results_list: List of dicts with metrics per model.
        xgboost_model: Fitted XGBClassifier (saved for SHAP in Stage 4).
    """
    wandb = None
    if use_wandb:
        try:
            import wandb as _wandb
            wandb = _wandb
        except ImportError:
            logger.warning("wandb not installed — skipping experiment tracking")
            use_wandb = False

    X_train, y_train, X_val, y_val, X_test, y_test = load_and_split(parquet_path)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    results = []
    pr_curve_data = {}  

    if use_wandb:
        wandb.init(project="onchain-fraud-gnn", name="baseline-lr", reinit=True)

    logger.info("Training Logistic Regression...")
    lr = LogisticRegression(
        class_weight="balanced", max_iter=1000, C=0.1, solver="lbfgs"
    )
    lr.fit(X_train_s, y_train)
    lr_probs = lr.predict_proba(X_test_s)[:, 1]
    lr_metrics = eval_metrics(y_test, lr_probs, "LogisticRegression")
    results.append(lr_metrics)

    lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
    pr_curve_data["LogisticRegression"] = (lr_recall, lr_precision, lr_probs, y_test)

    if use_wandb:
        wandb.log(lr_metrics)
        wandb.finish()

    if use_wandb:
        wandb.init(project="onchain-fraud-gnn", name="baseline-rf", reinit=True)

    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_depth=None,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_metrics = eval_metrics(y_test, rf_probs, "RandomForest")
    results.append(rf_metrics)

    rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_probs)
    pr_curve_data["RandomForest"] = (rf_recall, rf_precision, rf_probs, y_test)

    if use_wandb:
        wandb.log(rf_metrics)
        wandb.finish()

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    if use_wandb:
        wandb.init(project="onchain-fraud-gnn", name="baseline-xgb", reinit=True)

    logger.info("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=42,
        tree_method="hist",     
        device="cpu",
    )
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    xgb_metrics = eval_metrics(y_test, xgb_probs, "XGBoost")
    results.append(xgb_metrics)

    xgb_precision, xgb_recall, _ = precision_recall_curve(y_test, xgb_probs)
    pr_curve_data["XGBoost"] = (xgb_recall, xgb_precision, xgb_probs, y_test)

    if use_wandb:
        wandb.log(xgb_metrics)
        wandb.finish()

    return results, xgb, pr_curve_data
