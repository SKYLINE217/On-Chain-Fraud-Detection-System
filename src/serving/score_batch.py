"""
score_batch.py -- Model loading for batch scoring and API.
"""
import json
import torch
from src.models.graphsage import GraphSAGE
from src.models.gat import GAT

def load_model_from_config(checkpoint_path: str, config_path: str) -> tuple[torch.nn.Module, dict]:
    with open(config_path) as f:
        config = json.load(f)

    if config["model_type"] == "graphsage" or config["model_type"] == "GraphSAGE":
        model = GraphSAGE(
            in_channels=config["in_channels"],
            hidden_channels=config["hidden_channels"],
            out_channels=config["out_channels"],
            num_layers=config["num_layers"],
            dropout=config.get("dropout", 0.0),
            aggr=config.get("aggr", "mean"),
        )
    elif config["model_type"] == "GAT" or config["model_type"] == "gat":
        model = GAT(
            in_channels=config["in_channels"],
            hidden_channels=config["hidden_channels"],
            out_channels=config["out_channels"],
            dropout=config.get("dropout", 0.0),
        )
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config
