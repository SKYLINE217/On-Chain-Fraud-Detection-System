#!/bin/bash

set -euo pipefail

DATA_DIR="data/raw"
mkdir -p "$DATA_DIR"

echo "========================================="
echo "  Elliptic Dataset Download"
echo "========================================="

if [ -f "$DATA_DIR/elliptic_txs_features.csv" ] && \
   [ -f "$DATA_DIR/elliptic_txs_classes.csv" ] && \
   [ -f "$DATA_DIR/elliptic_txs_edgelist.csv" ]; then
    echo "Dataset files already exist in $DATA_DIR. Skipping download."
    echo "Delete them manually if you want to re-download."
    exit 0
fi

echo "Downloading Elliptic dataset via Kaggle CLI..."
echo "(Requires ~/.kaggle/kaggle.json to be configured)"

kaggle datasets download -d ellipticco/elliptic-data-set -p "$DATA_DIR" --unzip

echo ""
echo "Validating downloaded files..."

MISSING=0
for f in elliptic_txs_features.csv elliptic_txs_classes.csv elliptic_txs_edgelist.csv; do
    if [ -f "$DATA_DIR/$f" ]; then
        echo "  ✅ $f exists"
    else
        echo "  ❌ MISSING: $f"
        MISSING=1
    fi
done

if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "ERROR: Some files are missing. Download may have failed."
    exit 1
fi

echo ""
echo "Verifying SHA-256 hashes (BC-02)..."

declare -A EXPECTED_HASHES
EXPECTED_HASHES["elliptic_txs_features.csv"]="fd7f83573443c9e302e371d3f110e3b6224160f5d1ed8a287757936127800ff0"
EXPECTED_HASHES["elliptic_txs_classes.csv"]="93e2e7b2405c735ba752bf6ba06b947561deddd1f5a8fc91e46f6a4c0e439493"
EXPECTED_HASHES["elliptic_txs_edgelist.csv"]="a35053ba68a98e4382cae2ba65b9d9e36b23b6439e02dff084971b1b72a5156e"

HASH_FAILED=0
for f in elliptic_txs_features.csv elliptic_txs_classes.csv elliptic_txs_edgelist.csv; do
    actual=$(sha256sum "$DATA_DIR/$f" | awk '{print $1}')
    expected=${EXPECTED_HASHES[$f]}
    if [[ "$expected" == EXPECTED_HASH* ]]; then
        echo "  ⚠️ HASH SKIPPED (Placeholder): $f -> $actual"
    elif [ "$expected" == "$actual" ]; then
        echo "  ✅ HASH OK: $f"
    else
        echo "  ❌ HASH MISMATCH: $f (Expected $expected, got $actual)"
        HASH_FAILED=1
    fi
done

if [ "$HASH_FAILED" -eq 1 ]; then
    echo "ERROR: Data integrity verification failed. Potential MITM or corrupted download."
    exit 1
fi

echo ""
echo "Row counts:"
wc -l "$DATA_DIR"/*.csv

echo ""
echo "========================================="
echo "  Download complete!"
echo "  Files saved to: $DATA_DIR/"
echo "========================================="
