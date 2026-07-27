import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay

from scripts.verify_parquet import verify_parquet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

REAL_PARQUET = "data/processed/features_combined.parquet"
MOCK_PARQUET = "mocks/person_a/mock_features_combined.parquet"
CHECKPOINT_DIR = "checkpoints"
XGB_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "xgb_baseline.pkl")
PR_CURVE_PATH = "docs/figures/pr_curve_baselines.png"

def generate_mock_if_needed(mock_path: str) -> str:
    """Generate mock parquet if it doesn't exist."""
    if os.path.exists(mock_path):
        logger.info(f"Mock parquet already exists: {mock_path}")
        return mock_path

    logger.info("Generating mock parquet data...")
    from mocks.person_a.generate_mock_parquet import generate_mock_parquet
    return generate_mock_parquet(mock_path)

def plot_pr_curves(pr_curve_data: dict, output_path: str) -> str:
    """
    Generate overlaid PR curves for all baseline models.

    Args:
        pr_curve_data: Dict of {model_name: (recall, precision, probs, y_test)}.
        output_path: Where to save the figure.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {
        "LogisticRegression": "#3b82f6",   
        "RandomForest": "#22c55e",          
        "XGBoost": "#ef4444",               
    }

    for model_name, (recall, precision, probs, y_test) in pr_curve_data.items():
        color = colors.get(model_name, "#6b7280")
        ax.plot(
            recall, precision,
            label=model_name,
            color=color,
            linewidth=2,
        )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("PR Curves — Baseline Models (Illicit Class)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"PR curve saved: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Baseline Models")
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock parquet data (for development before Person A delivers)",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--parquet-path", type=str, default=None,
        help="Custom path to features_combined.parquet",
    )
    args = parser.parse_args()

    use_wandb = not args.no_wandb

    if args.parquet_path:
        parquet_path = args.parquet_path
    elif args.mock:
        parquet_path = MOCK_PARQUET
        generate_mock_if_needed(parquet_path)
    else:
        parquet_path = REAL_PARQUET

    logger.info("=" * 60)
    logger.info("STEP 1: Verifying parquet data")
    logger.info("=" * 60)

    if not verify_parquet(parquet_path):
        logger.error("Parquet verification failed — aborting")
        sys.exit(1)

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: Training baseline models")
    logger.info("=" * 60)

    from src.models.baselines import run_all_baselines

    results, xgb_model, pr_curve_data = run_all_baselines(
        parquet_path=parquet_path,
        use_wandb=use_wandb,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: Saving XGBoost checkpoint")
    logger.info("=" * 60)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    joblib.dump(xgb_model, XGB_CHECKPOINT)
    ckpt_size = os.path.getsize(XGB_CHECKPOINT) / (1024 * 1024)
    logger.info(f"XGBoost checkpoint saved: {XGB_CHECKPOINT} ({ckpt_size:.1f} MB)")

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: Generating PR curve figure")
    logger.info("=" * 60)

    plot_pr_curves(pr_curve_data, PR_CURVE_PATH)

    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS — Baseline Model Comparison")
    logger.info("=" * 60)

    df_results = pd.DataFrame(results)
    print("\n" + df_results.to_string(index=False))

    if args.mock:
        print("\n⚠ NOTE: Results above are from MOCK data — not real Elliptic features.")
        print("  Re-run against real data after Person A delivers features_combined.parquet.")

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 1 CHECKLIST")
    logger.info("=" * 60)

    checks = {
        "verify_parquet.py passes": True,  
        "XGBoost checkpoint saved": os.path.exists(XGB_CHECKPOINT),
        "PR curve figure generated": os.path.exists(PR_CURVE_PATH),
        "3 baseline results logged": len(results) == 3,
    }

    all_pass = True
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        if not passed:
            all_pass = False
        logger.info(f"  {status} {check}")

    if not use_wandb:
        logger.info("  ⚠ W&B logging disabled — run with W&B to complete checklist")

    if all_pass:
        logger.info("\n🎉 Stage 1 complete!")
    else:
        logger.error("\n❌ Some checks failed — review output above")
        sys.exit(1)

if __name__ == "__main__":
    main()
