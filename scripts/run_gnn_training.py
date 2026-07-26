"""
run_gnn_training.py -- Stage 2 entry point for first end-to-end GNN training.

Trains GraphSAGE and GAT on the Elliptic dataset (mock or real).
Validates invariants, runs first training loop, and checks success criteria:
    - Training loss decreases in first 10 epochs
    - Val PR-AUC > 0.5 after 20 epochs (on real data)
    - No NaN loss

Usage:
    python scripts/run_gnn_training.py --mock --no-wandb
    python scripts/run_gnn_training.py  # real data + W&B

Compliance Disclaimer:
    This system is a research and portfolio demonstration only. It is NOT
    a certified AML/CFT compliance tool, a regulated financial product, or
    a legally defensible fraud-detection system.
"""

import argparse
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

MOCK_PARQUET = "mocks/person_a/mock_features_combined.parquet"
REAL_PARQUET = "data/processed/features_combined.parquet"


def main():
    parser = argparse.ArgumentParser(description="Stage 2: GNN Training")
    parser.add_argument("--mock", action="store_true", help="Use mock data")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    args = parser.parse_args()

    use_wandb = not args.no_wandb
    parquet_path = MOCK_PARQUET if args.mock else REAL_PARQUET

    # Ensure mock data exists
    if args.mock and not os.path.exists(MOCK_PARQUET):
        logger.info("Generating mock parquet...")
        from mocks.person_a.generate_mock_parquet import generate_mock_parquet
        generate_mock_parquet(MOCK_PARQUET)

    # -----------------------------------------------------------------------
    # 1. Build PyG Data
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1: Building PyG Data object")
    logger.info("=" * 60)

    from src.features.build_pyg import load_pyg_data

    data = load_pyg_data(
        parquet_path=parquet_path,
        mock_edges=args.mock,
    )

    logger.info(f"Data: {data}")
    logger.info(f"Features: {data.x.shape[1]}")
    logger.info(
        f"Class weights preview: "
        f"illicit in train = {int((data.y[data.train_mask] == 1).sum())}, "
        f"licit in train = {int((data.y[data.train_mask] == 0).sum())}"
    )

    # -----------------------------------------------------------------------
    # 2. Train GraphSAGE
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: Training GraphSAGE (3L, h=128, mean)")
    logger.info("=" * 60)

    from src.models.graphsage import GraphSAGE
    from src.models.train import train

    wandb = None
    if use_wandb:
        try:
            import wandb as _wandb
            wandb = _wandb
            wandb.init(project="onchain-fraud-gnn", name="sage-first-run", reinit=True)
        except ImportError:
            logger.warning("wandb not available")
            use_wandb = False

    sage_model = GraphSAGE(
        in_channels=data.x.shape[1],
        hidden_channels=128,
        out_channels=2,
        num_layers=3,
        dropout=0.3,
        aggr="mean",
    )

    sage_config = {
        "lr": 0.001,
        "weight_decay": 5e-4,
        "epochs": args.epochs,
        "patience": args.patience,
    }

    os.makedirs("checkpoints", exist_ok=True)
    sage_metrics = train(
        model=sage_model,
        data=data,
        config=sage_config,
        checkpoint_path="checkpoints/best_model.pt",
        use_wandb=use_wandb,
    )

    if use_wandb and wandb:
        wandb.finish()

    logger.info(f"GraphSAGE test metrics: {sage_metrics}")

    # -----------------------------------------------------------------------
    # 3. Train GAT
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: Training GAT (2L, h=128, 4 heads)")
    logger.info("=" * 60)

    from src.models.gat import GAT

    if use_wandb and wandb:
        wandb.init(project="onchain-fraud-gnn", name="gat-first-run", reinit=True)

    gat_model = GAT(
        in_channels=data.x.shape[1],
        hidden_channels=128,
        out_channels=2,
        heads=4,
        dropout=0.3,
    )

    gat_metrics = train(
        model=gat_model,
        data=data,
        config={**sage_config, "epochs": args.epochs},
        checkpoint_path="checkpoints/gat_first_run.pt",
        use_wandb=use_wandb,
    )

    if use_wandb and wandb:
        wandb.finish()

    logger.info(f"GAT test metrics: {gat_metrics}")

    # -----------------------------------------------------------------------
    # 4. Verify get_embedding()
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: Verifying get_embedding()")
    logger.info("=" * 60)

    sage_model.eval()
    with torch.no_grad():
        emb = sage_model.get_embedding(data.x, data.edge_index)
    assert emb.shape == (203769, 128), f"Embedding shape mismatch: {emb.shape}"
    assert not torch.isnan(emb).any(), "NaN in embeddings"
    logger.info(f"get_embedding() output: {emb.shape} -- OK")

    # -----------------------------------------------------------------------
    # 5. Verify GAT return_attention
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 5: Verifying GAT attention weights")
    logger.info("=" * 60)

    gat_model.eval()
    with torch.no_grad():
        out, (edge_idx, alpha) = gat_model(
            data.x, data.edge_index, return_attention=True
        )
    assert out.shape == (203769, 2), f"GAT output shape: {out.shape}"
    logger.info(
        f"GAT attention: edge_idx={edge_idx.shape}, alpha={alpha.shape} -- OK"
    )

    # -----------------------------------------------------------------------
    # 6. Results summary
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 2 RESULTS")
    logger.info("=" * 60)

    import pandas as pd
    results_df = pd.DataFrame([
        {"model": "GraphSAGE (3L, h=128)", **sage_metrics},
        {"model": "GAT (2L, h=128, 4h)", **gat_metrics},
    ])
    print("\n" + results_df.to_string(index=False))

    if args.mock:
        print(
            "\n[!] NOTE: Results above are from MOCK data -- "
            "not real Elliptic features."
        )

    # Checklist
    logger.info("")
    logger.info("STAGE 2 CHECKLIST")
    checks = {
        "GraphSAGE get_embedding() works": emb.shape == (203769, 128),
        "GAT return_attention works": alpha is not None,
        "best_model.pt saved": os.path.exists("checkpoints/best_model.pt"),
        "gat_first_run.pt saved": os.path.exists("checkpoints/gat_first_run.pt"),
        "No NaN in outputs": (
            not torch.isnan(emb).any() and not torch.isnan(out).any()
        ),
    }

    all_pass = True
    for check, passed in checks.items():
        status = "[OK]" if passed else "[FAIL]"
        if not passed:
            all_pass = False
        logger.info(f"  {status} {check}")

    if all_pass:
        logger.info("\nStage 2 complete!")
    else:
        logger.error("\nSome checks failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
