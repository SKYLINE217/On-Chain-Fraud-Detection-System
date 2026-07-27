"""
plot_pr_curves.py -- Fetch PR curve artifacts from W&B and plot them.
"""
import argparse
import matplotlib.pyplot as plt
import sys
import os

def plot_curves(entity_project: str, best_sage: str, best_gat: str):
    try:
        import wandb
    except ImportError:
        print("wandb not installed. Run `pip install wandb`.")
        sys.exit(1)

    api = wandb.Api()

    fig, ax = plt.subplots(figsize=(8, 6))

    model_runs = {
        "GraphSAGE (Best)": best_sage,
        "GAT (Best)": best_gat,
        "XGBoost": "baseline-xgb",
        "Random Forest": "baseline-rf",
    }

    found_any = False
    for label, run_id in model_runs.items():
        if not run_id:
            continue

        try:

            runs = api.runs(entity_project, filters={"display_name": run_id})
            if len(runs) > 0:
                run = runs[0]
            else:
                run = api.run(f"{entity_project}/{run_id}")

            artifacts = run.logged_artifacts()
            for art in artifacts:

                if "pr_curve" in art.name or art.type == "run_table":

                    art_dir = art.download()
                    import json

                    for file in os.listdir(art_dir):
                        if file.endswith(".json"):
                            with open(os.path.join(art_dir, file), 'r') as f:
                                data = json.load(f)

                                if "data" in data and "columns" in data:
                                    recall_idx = data["columns"].index("recall")
                                    precision_idx = data["columns"].index("precision")
                                    recall_vals = [row[recall_idx] for row in data["data"]]
                                    precision_vals = [row[precision_idx] for row in data["data"]]

                                    ax.plot(recall_vals, precision_vals, label=label)
                                    found_any = True
                                    break
        except Exception as e:
            print(f"Could not load PR curve for {label} ({run_id}): {e}")

    if not found_any:
        print("No PR curve data could be loaded. Did you run the models with W&B?")
        sys.exit(1)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curves -- All Models (Illicit Class)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs("docs/figures", exist_ok=True)
    plt.savefig("docs/figures/pr_curves_all_models.png", dpi=150)
    print("Saved docs/figures/pr_curves_all_models.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="your-username/onchain-fraud-gnn",
                        help="W&B project path (entity/project)")
    parser.add_argument("--sage", type=str, required=True,
                        help="Run ID or name for best GraphSAGE run")
    parser.add_argument("--gat", type=str, required=True,
                        help="Run ID or name for best GAT run")
    args = parser.parse_args()

    plot_curves(args.project, args.sage, args.gat)
