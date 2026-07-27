
import torch
import pytest
from src.models.graphsage import GraphSAGE
from src.models.gat import GAT

@pytest.fixture
def dummy_data():
    n_nodes, n_features = 100, 174
    x = torch.randn(n_nodes, n_features)

    edge_index = torch.randint(0, n_nodes, (2, 300))
    return x, edge_index

def test_graphsage_output_shape(dummy_data):
    x, edge_index = dummy_data
    model = GraphSAGE(in_channels=174, hidden_channels=64, out_channels=2, num_layers=2)
    out = model(x, edge_index)
    assert out.shape == (100, 2), f"Expected (100, 2), got {out.shape}"

def test_graphsage_embedding_shape(dummy_data):
    x, edge_index = dummy_data
    model = GraphSAGE(in_channels=174, hidden_channels=64, out_channels=2, num_layers=3)
    emb = model.get_embedding(x, edge_index)
    assert emb.shape == (100, 64), f"Expected (100, 64), got {emb.shape}"

def test_gat_output_shape(dummy_data):
    x, edge_index = dummy_data
    model = GAT(in_channels=174, hidden_channels=64, out_channels=2)
    out = model(x, edge_index)
    assert out.shape == (100, 2)

def test_gat_attention_weights(dummy_data):
    x, edge_index = dummy_data
    model = GAT(in_channels=174, hidden_channels=64, out_channels=2, heads=4)
    out, (edge_idx, alpha) = model(x, edge_index, return_attention=True)
    assert out.shape == (100, 2)
    assert alpha.shape[1] == 4  

def test_graphsage_no_nan(dummy_data):
    x, edge_index = dummy_data
    model = GraphSAGE(in_channels=174, hidden_channels=64, out_channels=2, num_layers=2)
    out = model(x, edge_index)
    assert not torch.isnan(out).any(), "NaN in GraphSAGE output"

def test_graphsage_eval_mode_deterministic(dummy_data):
    x, edge_index = dummy_data
    model = GraphSAGE(in_channels=174, hidden_channels=64, out_channels=2, num_layers=2)
    model.eval()
    with torch.no_grad():
        out1 = model(x, edge_index)
        out2 = model(x, edge_index)
    assert torch.allclose(out1, out2), "Non-deterministic in eval mode"
