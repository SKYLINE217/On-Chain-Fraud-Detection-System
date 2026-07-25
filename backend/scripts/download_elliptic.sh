#!/usr/bin/env bash
# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.

DATA_DIR="data/raw"
mkdir -p "$DATA_DIR"

echo "Downloading Elliptic Dataset..."
# Elliptic dataset URL placeholder or Kaggle download command
curl -sL "https://www.kaggle.com/api/v1/datasets/download/eltorin/elliptic-data-set" -o "$DATA_DIR/elliptic.zip" || true

echo "Download completed or skipped if dataset is manually provided."
