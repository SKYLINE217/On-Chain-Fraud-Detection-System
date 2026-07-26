# scripts/download_elliptic.ps1
# PowerShell alternative for Windows users.
# Downloads the Elliptic Bitcoin Transaction Dataset from Kaggle.
#
# Prerequisites:
#   - Kaggle CLI installed: pip install kaggle
#   - Kaggle credentials configured: %USERPROFILE%\.kaggle\kaggle.json
#
# Usage:
#   .\scripts\download_elliptic.ps1
#
# Compliance Disclaimer: This system is a research and portfolio
# demonstration only. Not a certified AML/CFT compliance tool.

$ErrorActionPreference = "Stop"

$DATA_DIR = "data\raw"

if (-not (Test-Path $DATA_DIR)) {
    New-Item -ItemType Directory -Path $DATA_DIR -Force | Out-Null
}

Write-Host "========================================="
Write-Host "  Elliptic Dataset Download"
Write-Host "========================================="

# Check if files already exist
$files = @(
    "elliptic_txs_features.csv",
    "elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv"
)

$allExist = $true
foreach ($f in $files) {
    if (-not (Test-Path (Join-Path $DATA_DIR $f))) {
        $allExist = $false
        break
    }
}

if ($allExist) {
    Write-Host "Dataset files already exist in $DATA_DIR. Skipping download."
    Write-Host "Delete them manually if you want to re-download."
    exit 0
}

Write-Host "Downloading Elliptic dataset via Kaggle CLI..."
Write-Host "(Requires %USERPROFILE%\.kaggle\kaggle.json to be configured)"

# Download via Kaggle CLI
kaggle datasets download -d ellipticco/elliptic-data-set -p $DATA_DIR --unzip

Write-Host ""
Write-Host "Validating downloaded files..."

$missing = 0
foreach ($f in $files) {
    $filePath = Join-Path $DATA_DIR $f
    if (Test-Path $filePath) {
        Write-Host "  ✅ $f exists"
    } else {
        Write-Host "  ❌ MISSING: $f"
        $missing++
    }
}

if ($missing -gt 0) {
    Write-Host ""
    Write-Host "ERROR: Some files are missing. Download may have failed."
    exit 1
}

Write-Host ""
Write-Host "Row counts:"
foreach ($f in $files) {
    $filePath = Join-Path $DATA_DIR $f
    $lineCount = (Get-Content $filePath | Measure-Object -Line).Lines
    Write-Host "  $lineCount $f"
}
# Expected:
#   203770 elliptic_txs_features.csv  (header + 203769 rows)
#   203770 elliptic_txs_classes.csv
#   234356 elliptic_txs_edgelist.csv

Write-Host ""
Write-Host "========================================="
Write-Host "  Download complete!"
Write-Host "  Files saved to: $DATA_DIR\"
Write-Host "========================================="
