# scripts/download_elliptic.ps1
# Downloads the Elliptic Bitcoin Transaction Dataset from Kaggle.
#
# Prerequisites:
#   - Kaggle CLI installed: pip install kaggle
#   - Kaggle credentials configured: ~/.kaggle/kaggle.json
#
# Usage:
#   .\scripts\download_elliptic.ps1
#
# Compliance Disclaimer: This system is a research and portfolio
# demonstration only. Not a certified AML/CFT compliance tool.

$ErrorActionPreference = "Stop"

$DATA_DIR = "data/raw"
if (-not (Test-Path $DATA_DIR)) {
    New-Item -ItemType Directory -Path $DATA_DIR -Force | Out-Null
}

Write-Host "=========================================" -ForegroundColor Green
Write-Host "  Elliptic Dataset Download" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""

# Check if files already exist
$filesExist = (Test-Path "$DATA_DIR/elliptic_txs_features.csv") -and `
              (Test-Path "$DATA_DIR/elliptic_txs_classes.csv") -and `
              (Test-Path "$DATA_DIR/elliptic_txs_edgelist.csv")

if ($filesExist) {
    Write-Host "Dataset files already exist in $DATA_DIR. Skipping download."
    Write-Host "Delete them manually if you want to re-download."
    exit 0
}

Write-Host "Downloading Elliptic dataset via Kaggle CLI..."
Write-Host "(Requires ~/.kaggle/kaggle.json to be configured)"
Write-Host ""

# Via Kaggle CLI (requires ~/.kaggle/kaggle.json)
kaggle datasets download -d ellipticco/elliptic-data-set -p "$DATA_DIR" --unzip

Write-Host ""
Write-Host "Validating downloaded files..."
Write-Host ""

# Validate files present
$MISSING = 0
$files = @("elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv")
foreach ($f in $files) {
    if (Test-Path "$DATA_DIR/$f") {
        Write-Host "  ✅ $f exists"
    } else {
        Write-Host "  ❌ MISSING: $f"
        $MISSING = 1
    }
}

if ($MISSING -eq 1) {
    Write-Host ""
    Write-Host "ERROR: Some files are missing. Download may have failed."
    exit 1
}

Write-Host ""
Write-Host "Verifying SHA-256 hashes (BC-02)..."

$EXPECTED_HASHES = @{
    "elliptic_txs_features.csv" = "EXPECTED_HASH_FEATURES"
    "elliptic_txs_classes.csv" = "EXPECTED_HASH_CLASSES"
    "elliptic_txs_edgelist.csv" = "EXPECTED_HASH_EDGELIST"
}

$HASH_FAILED = 0
foreach ($f in $files) {
    $expected = $EXPECTED_HASHES[$f]
    $actual = (Get-FileHash "$DATA_DIR/$f" -Algorithm SHA256).Hash.ToLower()
    if ($expected -like "EXPECTED_HASH*") {
        Write-Host "  ⚠️ HASH SKIPPED (Placeholder): $f -> $actual" -ForegroundColor Yellow
    } elseif ($expected.ToLower() -eq $actual) {
        Write-Host "  ✅ HASH OK: $f" -ForegroundColor Green
    } else {
        Write-Host "  ❌ HASH MISMATCH: $f (Expected $expected, got $actual)" -ForegroundColor Red
        $HASH_FAILED = 1
    }
}

if ($HASH_FAILED -eq 1) {
    Write-Host ""
    Write-Host "ERROR: Data integrity verification failed. Potential MITM or corrupted download." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Row counts:"
Get-ChildItem "$DATA_DIR/*.csv" | ForEach-Object {
    $lineCount = @(Get-Content $_.FullName | Measure-Object -Line).Lines
    Write-Host "  $lineCount $($_.Name)"
}
Write-Host ""
Write-Host "# Expected:"
Write-Host "#   203770 elliptic_txs_features.csv  (header + 203769 rows)"
Write-Host "#   203770 elliptic_txs_classes.csv"
Write-Host "#   234356 elliptic_txs_edgelist.csv"
Write-Host ""

Write-Host "=========================================" -ForegroundColor Green
Write-Host "  Download complete!" -ForegroundColor Green
Write-Host "  Files saved to: $DATA_DIR/" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
