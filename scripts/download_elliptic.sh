#!/bin/bash
# scripts/download_elliptic.sh
# Downloads the Elliptic Bitcoin Transaction Dataset from Kaggle.
#
# Prerequisites:
#   - Kaggle CLI installed: pip install kaggle
#   - Kaggle credentials configured: ~/.kaggle/kaggle.json
#
# Usage:
#   bash scripts/download_elliptic.sh
#
# Compliance Disclaimer: This system is a research and portfolio
# demonstration only. Not a certified AML/CFT compliance tool.

set -euo pipefail

DATA_DIR="data/raw"
mkdir -p "$DATA_DIR"

echo "========================================="
echo "  Elliptic Dataset Download"
echo "========================================="

# Check if files already exist
if [ -f "$DATA_DIR/elliptic_txs_features.csv" ] && \
   [ -f "$DATA_DIR/elliptic_txs_classes.csv" ] && \
   [ -f "$DATA_DIR/elliptic_txs_edgelist.csv" ]; then
    echo "Dataset files already exist in $DATA_DIR. Skipping download."
    echo "Delete them manually if you want to re-download."
    exit 0
fi

echo "Downloading Elliptic dataset via Kaggle CLI..."
echo "(Requires ~/.kaggle/kaggle.json to be configured)"

# Via Kaggle CLI (requires ~/.kaggle/kaggle.json)
kaggle datasets download -d ellipticco/elliptic-data-set -p "$DATA_DIR" --unzip

echo ""
echo "Validating downloaded files..."

# Validate files present
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
echo "Row counts:"
wc -l "$DATA_DIR"/*.csv
# Expected:
#   203770 elliptic_txs_features.csv  (header + 203769 rows)
#   203770 elliptic_txs_classes.csv
#   234356 elliptic_txs_edgelist.csv

echo ""
echo "========================================="
echo "  Download complete!"
echo "  Files saved to: $DATA_DIR/"
echo "========================================="
