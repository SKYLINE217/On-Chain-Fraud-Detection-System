import os
import logging
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PARQUET_FILE = os.path.join(PROCESSED_DIR, "features_combined.parquet")

LABEL_MAP = {"2": 0, "1": 1, "unknown": -1}

def encode_labels(class_series: pd.Series) -> np.ndarray:
    return class_series.map(LABEL_MAP).values.astype(np.int64)

def evaluate(y_true, y_pred, y_prob, model_name):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    log.info(f"--- {model_name} Results ---")
    log.info(f"Precision: {precision:.4f}")
    log.info(f"Recall:    {recall:.4f}")
    log.info(f"F1 Score:  {f1:.4f}")
    log.info(f"PR-AUC:    {pr_auc:.4f}")
    log.info(f"ROC-AUC:   {roc_auc:.4f}")
    
    return {"precision": precision, "recall": recall, "f1": f1, "pr_auc": pr_auc, "roc_auc": roc_auc}

def train_baselines(use_engineered=True):
    if not os.path.isfile(PARQUET_FILE):
        log.error(f"Parquet file not found: {PARQUET_FILE}")
        return

    df = pd.read_parquet(PARQUET_FILE)
    
    y = encode_labels(df["class"])
    
    # Exclude unknowns
    is_labeled = y != -1
    
    df_labeled = df[is_labeled]
    y_labeled = y[is_labeled]
    
    time_steps = df_labeled["timeStep"].values
    
    # Train 1-34, Val 35-39, Test 40-49
    train_mask = (time_steps >= 1) & (time_steps <= 34)
    val_mask = (time_steps >= 35) & (time_steps <= 39)
    test_mask = (time_steps >= 40) & (time_steps <= 49)
    
    # Features
    feature_names = [f"f{i}" for i in range(1, 167)]
    if use_engineered:
        engineered_names = [
            "tx_freq", "amount_mean", "amount_skew", "address_age",
            "clustering_coeff", "burst_score", "pageRank", "communityId",
        ]
        feature_names += [c for c in engineered_names if c in df.columns]
        
    X = df_labeled[feature_names].values
    
    X_train, y_train = X[train_mask], y_labeled[train_mask]
    X_val, y_val = X[val_mask], y_labeled[val_mask]
    X_test, y_test = X[test_mask], y_labeled[test_mask]
    
    log.info(f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")
    
    # Class weights for imbalance
    illicit_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    
    # 1. Logistic Regression
    lr = LogisticRegression(class_weight="balanced", max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    evaluate(y_test, y_pred_lr, y_prob_lr, "Logistic Regression")
    
    # 2. XGBoost
    xgb_clf = xgb.XGBClassifier(
        scale_pos_weight=illicit_weight,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="aucpr",
        early_stopping_rounds=10
    )
    xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_pred_xgb = xgb_clf.predict(X_test)
    y_prob_xgb = xgb_clf.predict_proba(X_test)[:, 1]
    evaluate(y_test, y_pred_xgb, y_prob_xgb, "XGBoost")

if __name__ == "__main__":
    train_baselines()
