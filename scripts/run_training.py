import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.graphsage import GraphSAGE
from src.models.train import train
from src.features.build_pyg import load_pyg_data
import torch

def main():
    print("Loading PyG data...")

    data = load_pyg_data()

    print(f"Data loaded: {data.num_nodes} nodes, {data.num_edges} edges.")

    in_channels = data.num_features
    print(f"Input features: {in_channels}")

    model = GraphSAGE(in_channels=in_channels, hidden_channels=128, out_channels=2)

    config = {
        "epochs": 50, 
        "lr": 0.001,
        "weight_decay": 5e-4,
        "patience": 10
    }

    checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')

    print("Starting training...")

    test_metrics = train(
        model=model, 
        data=data, 
        config=config, 
        checkpoint_path=checkpoint_path,
        use_wandb=False
    )

    print("Training complete!")
    print(test_metrics)

if __name__ == "__main__":
    main()
