"""
verify_parquet.py — Validate features_combined.parquet against integration contract.

Run this BEFORE any model training.
Validates that the parquet matches the exact contract from PLAN.md.

Usage:
    python scripts/verify_parquet.py [path_to_parquet]

Contract (from PLAN.md → Integration Contracts):
    Shape: (203769, 177)
    NOTE: PLAN.md states 171 but explicitly lists 177 columns
    (txId + timeStep + class + f1..f166 + 8 engineered = 3+166+8 = 177).
    The column list is authoritative.
    Zero NaNs in numeric feature columns
    class values ∈ {"1", "2", "unknown"}
    timeStep range: 1-49

Compliance Disclaimer:
    This system is a research and portfolio demonstration only. It is NOT
    a certified AML/CFT compliance tool, a regulated financial product, or
    a legally defensible fraud-detection system.
"""

import pandas as pd
import sys
import os

# Default path — override with CLI arg
DEFAULT_PATH = "data/processed/features_combined.parquet"

# Feature columns per PLAN.md contract
FEATURE_COLS = [f"f{i}" for i in range(1, 167)] + [
    "tx_freq", "amount_mean", "amount_skew", "address_age",
    "clustering_coeff", "burst_score", "pageRank", "communityId",
]

# All required columns
REQUIRED_COLS = ["txId", "timeStep", "class"] + FEATURE_COLS

# Valid class values
VALID_CLASSES = {"1", "2", "unknown"}


def verify_parquet(parquet_path: str) -> bool:
    """
    Validate features_combined.parquet against integration contract.

    Args:
        parquet_path: Path to the parquet file.

    Returns:
        True if all checks pass.

    Raises:
        AssertionError: If any contract check fails.
    """
    if not os.path.exists(parquet_path):
        print(f"[FAIL] File not found: {parquet_path}")
        return False

    print(f"Validating: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    # 1. Shape check
    # NOTE: PLAN.md states shape (203769, 171) but explicitly lists 177 columns
    # (txId + timeStep + class + f1..f166 + 8 engineered = 3 + 166 + 8 = 177).
    # The column list is authoritative.
    assert df.shape == (203769, 177), \
        f"[FAIL] Shape mismatch: {df.shape}  expected (203769, 177)"
    print(f"  [PASS] Shape: {df.shape}")

    # 2. Column existence
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    assert len(missing) == 0, f"[FAIL] Missing columns: {missing}"
    print(f"  [PASS] All {len(REQUIRED_COLS)} required columns present")

    # 3. Zero NaNs in feature columns
    nan_count = df[FEATURE_COLS].isna().sum().sum()
    assert nan_count == 0, f"[FAIL] NaN count in features: {nan_count}  (expected 0)"
    print(f"  [PASS] Zero NaNs in feature columns")

    # 4. Class values
    actual_classes = set(df["class"].unique())
    assert actual_classes.issubset(VALID_CLASSES), \
        f"[FAIL] Unexpected class values: {actual_classes - VALID_CLASSES}"
    print(f"  [PASS] Class values: {actual_classes}")

    # 5. Temporal range
    ts_min, ts_max = df["timeStep"].min(), df["timeStep"].max()
    assert ts_min == 1 and ts_max == 49, \
        f"[FAIL] timeStep range: {ts_min}-{ts_max}  expected 1-49"
    print(f"  [PASS] timeStep range: {ts_min}-{ts_max}")

    # 6. Class distribution summary
    class_dist = df["class"].value_counts().to_dict()
    print(f"\n  Class distribution:")
    for cls, count in sorted(class_dist.items()):
        pct = count / len(df) * 100
        print(f"    {cls}: {count:,} ({pct:.1f}%)")

    # 7. Labeled node statistics (for temporal split)
    labeled = df[df["class"] != "unknown"]
    train = labeled[labeled["timeStep"] <= 34]
    val = labeled[(labeled["timeStep"] >= 35) & (labeled["timeStep"] <= 39)]
    test = labeled[labeled["timeStep"] >= 40]

    print(f"\n  Temporal split (labeled only):")
    print(f"    Train (1-34):   {len(train):,} nodes")
    print(f"    Val   (35-39):  {len(val):,} nodes")
    print(f"    Test  (40-49):  {len(test):,} nodes")

    if len(train) > 0:
        train_illicit = (train["class"] == "1").sum()
        print(f"    Train illicit:  {train_illicit:,} ({train_illicit/len(train)*100:.1f}%)")

    print(f"\n[PASS] Parquet validated successfully: {parquet_path}")
    return True


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    success = verify_parquet(path)
    sys.exit(0 if success else 1)
