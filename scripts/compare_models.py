"""
compare_models.py -- Fetch all run results from W&B and produce comparison table.
"""
import argparse
import pandas as pd
import sys

def compare_models(entity_project: str):
    try:
        import wandb
    except ImportError:
        print("wandb not installed. Run `pip install wandb`.")
        sys.exit(1)
        
    api = wandb.Api()
    try:
        runs = api.runs(entity_project)
    except Exception as e:
        print(f"Error fetching runs from {entity_project}: {e}")
        print("Make sure you are logged in (wandb login) and the project name is correct.")
        sys.exit(1)

    rows = []
    for run in runs:
        if run.state == "finished" and run.summary.get("test_pr_auc"):
            rows.append({
                "Run": run.name,
                "Model": run.config.get("model_type", run.name.split("-")[0]),
                "PR-AUC": round(run.summary.get("test_pr_auc", 0), 4),
                "F1": round(run.summary.get("test_f1", 0), 4),
                "Precision": round(run.summary.get("test_precision", 0), 4),
                "Recall": round(run.summary.get("test_recall", 0), 4),
                "hidden_dim": run.config.get("hidden_channels", "-"),
                "layers": run.config.get("num_layers", "-"),
            })

    if not rows:
        print("No finished runs with 'test_pr_auc' found.")
        sys.exit(0)

    df = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False)
    print(df.to_string(index=False))
    
    # Save to docs/model_comparison.csv
    import os
    os.makedirs("docs", exist_ok=True)
    df.to_csv("docs/model_comparison.csv", index=False)
    print("\nSaved comparison to docs/model_comparison.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="your-username/onchain-fraud-gnn",
                        help="W&B project path (entity/project)")
    args = parser.parse_args()
    
    compare_models(args.project)
