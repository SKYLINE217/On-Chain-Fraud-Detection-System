# tests/test_baselines.py
# ──────────────────────────────────────────────────────────────────────────
# Sanity tests for the baseline models and temporal split integrity.
# Run with: python -m pytest tests/test_baselines.py -v
# ──────────────────────────────────────────────────────────────────────────

import sys
import os
import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.baselines import (
    load_elliptic_data,
    prepare_splits,
    evaluate_model,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    TRAIN_STEPS,
    VAL_STEPS,
    TEST_STEPS,
    LABEL_MAP,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def elliptic_data():
    """Load the Elliptic dataset once for the entire test module."""
    try:
        features, labels = load_elliptic_data()
        return features, labels
    except FileNotFoundError:
        pytest.skip(
            "Elliptic dataset not found in data/raw/. "
            "Download from Kaggle and place CSVs there."
        )


@pytest.fixture(scope="module")
def splits(elliptic_data):
    """Prepare temporal splits once for the entire test module."""
    features, labels = elliptic_data
    return prepare_splits(features, labels)


# ── Data Integrity Tests ─────────────────────────────────────────────────

class TestDataIntegrity:
    """Verify the raw dataset loads correctly."""

    def test_features_shape(self, elliptic_data):
        """Features CSV should have 203,769 rows and 168 columns (txId + timeStep + 166 features)."""
        features, _ = elliptic_data
        assert features.shape[0] == 203_769, f"Expected 203,769 nodes, got {features.shape[0]}"
        assert features.shape[1] == 168, f"Expected 168 columns, got {features.shape[1]}"

    def test_labels_class_distribution(self, elliptic_data):
        """Verify the expected class distribution."""
        _, labels = elliptic_data
        class_counts = labels["class"].value_counts()
        assert "1" in class_counts.index, "Missing illicit class ('1')"
        assert "2" in class_counts.index, "Missing licit class ('2')"
        assert "unknown" in class_counts.index, "Missing unknown class"

    def test_labels_match_features(self, elliptic_data):
        """Every txId in features should have a label entry."""
        features, labels = elliptic_data
        assert set(features["txId"]).issubset(set(labels["txId"]))


# ── Temporal Split Tests ─────────────────────────────────────────────────

class TestTemporalSplit:
    """Verify the temporal split is correct and non-overlapping."""

    def test_no_overlap_train_val(self):
        """Train and val time steps must not overlap."""
        overlap = set(TRAIN_STEPS) & set(VAL_STEPS)
        assert len(overlap) == 0, f"Train/val overlap on steps: {overlap}"

    def test_no_overlap_train_test(self):
        """Train and test time steps must not overlap."""
        overlap = set(TRAIN_STEPS) & set(TEST_STEPS)
        assert len(overlap) == 0, f"Train/test overlap on steps: {overlap}"

    def test_no_overlap_val_test(self):
        """Val and test time steps must not overlap."""
        overlap = set(VAL_STEPS) & set(TEST_STEPS)
        assert len(overlap) == 0, f"Val/test overlap on steps: {overlap}"

    def test_split_boundaries(self):
        """Verify exact temporal boundaries from aim.md §4."""
        assert list(TRAIN_STEPS) == list(range(1, 35)), "Train should be steps 1-34"
        assert list(VAL_STEPS) == list(range(35, 40)), "Val should be steps 35-39"
        assert list(TEST_STEPS) == list(range(40, 50)), "Test should be steps 40-49"

    def test_splits_sum_to_labeled(self, splits, elliptic_data):
        """Total split sizes must equal the number of labeled nodes."""
        _, labels = elliptic_data
        n_labeled = labels[labels["class"].isin(["1", "2"])].shape[0]
        X_train, y_train = splits["train"]
        X_val, y_val = splits["val"]
        X_test, y_test = splits["test"]
        total = len(y_train) + len(y_val) + len(y_test)
        assert total == n_labeled, (
            f"Split sizes ({total}) don't sum to labeled count ({n_labeled})"
        )


# ── Label Tests ──────────────────────────────────────────────────────────

class TestLabels:
    """Verify label encoding is correct."""

    def test_only_binary_labels_in_splits(self, splits):
        """Splits should contain only 0 (licit) and 1 (illicit), no -1 or 'unknown'."""
        for split_name, (X, y) in splits.items():
            unique_labels = set(np.unique(y))
            assert unique_labels.issubset({0, 1}), (
                f"{split_name} contains invalid labels: {unique_labels}"
            )

    def test_no_unknown_in_training(self, splits):
        """No unknown labels (-1 or otherwise) should leak into training data."""
        _, y_train = splits["train"]
        assert (y_train == -1).sum() == 0, "Unknown labels found in train split"
        assert (y_train < 0).sum() == 0, "Negative labels found in train split"

    def test_label_encoding_matches_spec(self):
        """Label map must match blend.md Contract 2: '2'→0 (licit), '1'→1 (illicit)."""
        assert LABEL_MAP == {"1": 1, "2": 0}, (
            f"Label map doesn't match spec: {LABEL_MAP}"
        )


# ── Feature Tests ────────────────────────────────────────────────────────

class TestFeatures:
    """Verify feature matrix integrity."""

    def test_feature_count(self, splits):
        """Feature matrix should have 166 columns (raw Elliptic features)."""
        X_train, _ = splits["train"]
        assert X_train.shape[1] == 166, f"Expected 166 features, got {X_train.shape[1]}"

    def test_no_nan_in_features(self, splits):
        """No NaN values should be present in any split's feature matrix."""
        for split_name, (X, _) in splits.items():
            assert not np.isnan(X).any(), f"NaN found in {split_name} features"

    def test_feature_dtype(self, splits):
        """Features should be float32."""
        X_train, _ = splits["train"]
        assert X_train.dtype == np.float32, f"Expected float32, got {X_train.dtype}"


# ── Model Training Smoke Tests ───────────────────────────────────────────

class TestBaselineModels:
    """Smoke tests — verify models train and produce valid predictions."""

    def test_logistic_regression_trains(self, splits):
        """LR should train without error and produce probabilities in [0, 1]."""
        X_train, y_train = splits["train"]
        X_val, y_val = splits["val"]
        model = train_logistic_regression(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        assert proba.min() >= 0.0, "Probabilities below 0"
        assert proba.max() <= 1.0, "Probabilities above 1"

    def test_random_forest_trains(self, splits):
        """RF should train without error and produce probabilities in [0, 1]."""
        X_train, y_train = splits["train"]
        X_val, y_val = splits["val"]
        model = train_random_forest(X_train, y_train, n_estimators=10)  # fast
        proba = model.predict_proba(X_val)[:, 1]
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_xgboost_trains(self, splits):
        """XGBoost should train without error and produce probabilities in [0, 1]."""
        X_train, y_train = splits["train"]
        X_val, y_val = splits["val"]
        model = train_xgboost(X_train, y_train, n_estimators=10)  # fast
        proba = model.predict_proba(X_val)[:, 1]
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_evaluation_returns_all_metrics(self, splits):
        """evaluate_model() should return all 5 required metrics."""
        X_train, y_train = splits["train"]
        X_val, y_val = splits["val"]
        model = train_logistic_regression(X_train, y_train)
        metrics = evaluate_model(model, X_val, y_val)
        required = [
            "precision_illicit", "recall_illicit", "f1_illicit",
            "pr_auc_illicit", "roc_auc",
        ]
        for key in required:
            assert key in metrics, f"Missing metric: {key}"
            assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}"
