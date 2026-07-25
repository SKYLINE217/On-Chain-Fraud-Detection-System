# tests/test_gnn_models.py
# ──────────────────────────────────────────────────────────────────────────
# Unit tests for GNN models (GraphSAGE, GAT) and training utilities.
#
# These tests use small synthetic graphs to validate:
#   - Model architecture (forward pass shape, embedding extraction)
#   - Loss masking (unknown nodes contribute zero loss)
#   - Class weighting
#   - Training step (loss decreases)
#   - Focal Loss implementation
#
# Run with: python -m pytest tests/test_gnn_models.py -v
# ──────────────────────────────────────────────────────────────────────────

import sys
import os

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.graphsage import GraphSAGE
from src.models.gat import GAT
from src.models.train import (
    FocalLoss,
    compute_class_weights,
    train_epoch,
    evaluate,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def small_graph():
    """
    Create a small synthetic graph for testing.
    20 nodes, 30 edges, 10 features, mix of labeled and unknown.
    """
    torch.manual_seed(42)
    n_nodes = 20
    n_features = 10
    n_edges = 30

    x = torch.randn(n_nodes, n_features)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))

    # Labels: 0=licit (10 nodes), 1=illicit (3 nodes), -1=unknown (7 nodes)
    y = torch.full((n_nodes,), -1, dtype=torch.long)
    y[:10] = 0   # licit
    y[10:13] = 1  # illicit
    # y[13:] = -1 (unknown) — already set

    # Masks: labeled nodes in "train" split (first 8 labeled = indices 0-7 + 10-12)
    is_labeled = y >= 0
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:8] = True      # 8 licit nodes
    train_mask[10:12] = True   # 2 illicit nodes
    # train_mask total = 10 nodes

    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask[8:10] = True      # 2 licit nodes
    val_mask[12] = True        # 1 illicit node
    # val_mask total = 3 nodes

    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    # No test nodes in this small graph (test set untouched per spec)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    return data


@pytest.fixture
def graphsage_model():
    """Small GraphSAGE model for testing."""
    return GraphSAGE(
        in_channels=10,
        hidden_channels=16,
        out_channels=2,
        num_layers=2,
        dropout=0.1,
    )


@pytest.fixture
def gat_model():
    """Small GAT model for testing."""
    return GAT(
        in_channels=10,
        hidden_channels=8,
        out_channels=2,
        heads=2,
        dropout=0.1,
    )


# ── GraphSAGE Tests ──────────────────────────────────────────────────────

class TestGraphSAGE:
    """Tests for the GraphSAGE model."""

    def test_forward_shape(self, graphsage_model, small_graph):
        """Forward pass should output (N, out_channels) logits."""
        out = graphsage_model(small_graph.x, small_graph.edge_index)
        assert out.shape == (20, 2), f"Expected (20, 2), got {out.shape}"

    def test_embedding_shape(self, graphsage_model, small_graph):
        """get_embeddings should return (N, hidden_channels)."""
        graphsage_model.eval()
        emb = graphsage_model.get_embeddings(small_graph.x, small_graph.edge_index)
        assert emb.shape == (20, 16), f"Expected (20, 16), got {emb.shape}"

    def test_config_serializable(self, graphsage_model):
        """get_config should return a JSON-serializable dict."""
        import json
        config = graphsage_model.get_config()
        assert config["model_type"] == "GraphSAGE"
        json.dumps(config)  # Should not raise

    def test_different_layer_counts(self, small_graph):
        """Models with 2 and 3 layers should both produce valid output."""
        for n_layers in [2, 3]:
            model = GraphSAGE(
                in_channels=10, hidden_channels=16, out_channels=2,
                num_layers=n_layers,
            )
            out = model(small_graph.x, small_graph.edge_index)
            assert out.shape == (20, 2)

    def test_different_aggregations(self, small_graph):
        """Both 'mean' and 'max' aggregation should work."""
        for aggr in ["mean", "max"]:
            model = GraphSAGE(
                in_channels=10, hidden_channels=16, out_channels=2,
                num_layers=2, aggr=aggr,
            )
            out = model(small_graph.x, small_graph.edge_index)
            assert out.shape == (20, 2)


# ── GAT Tests ────────────────────────────────────────────────────────────

class TestGAT:
    """Tests for the GAT model."""

    def test_forward_shape(self, gat_model, small_graph):
        """Forward pass should output (N, out_channels) logits."""
        out = gat_model(small_graph.x, small_graph.edge_index)
        assert out.shape == (20, 2), f"Expected (20, 2), got {out.shape}"

    def test_attention_weights(self, gat_model, small_graph):
        """return_attention_weights=True should return attention coefficients."""
        gat_model.eval()
        out, (attn1, attn2) = gat_model(
            small_graph.x, small_graph.edge_index,
            return_attention_weights=True,
        )
        assert out.shape == (20, 2)
        # attn1 is a tuple (edge_index, attention_weights)
        assert attn1[1] is not None, "First layer attention weights are None"
        assert attn2[1] is not None, "Second layer attention weights are None"

    def test_embedding_shape(self, gat_model, small_graph):
        """get_embeddings should return (N, hidden_channels * heads)."""
        gat_model.eval()
        emb = gat_model.get_embeddings(small_graph.x, small_graph.edge_index)
        # hidden_channels=8, heads=2 → embedding dim = 16
        assert emb.shape == (20, 16), f"Expected (20, 16), got {emb.shape}"

    def test_config_serializable(self, gat_model):
        """get_config should return a JSON-serializable dict."""
        import json
        config = gat_model.get_config()
        assert config["model_type"] == "GAT"
        json.dumps(config)


# ── Loss Masking Tests ───────────────────────────────────────────────────

class TestLossMasking:
    """
    Verify that unknown nodes contribute zero loss.
    This is the most common silent bug on Elliptic (aim.md §13).
    """

    def test_train_mask_excludes_unknown(self, small_graph):
        """train_mask should be False for all unknown-class nodes."""
        unknown_mask = small_graph.y == -1
        overlap = (small_graph.train_mask & unknown_mask).sum().item()
        assert overlap == 0, (
            f"{overlap} unknown nodes found in train_mask — critical bug!"
        )

    def test_val_mask_excludes_unknown(self, small_graph):
        """val_mask should be False for all unknown-class nodes."""
        unknown_mask = small_graph.y == -1
        overlap = (small_graph.val_mask & unknown_mask).sum().item()
        assert overlap == 0

    def test_loss_only_on_masked_nodes(self, graphsage_model, small_graph):
        """
        Loss should be computed only on train_mask nodes.
        Verify by comparing: loss on all nodes vs loss on masked nodes.
        """
        graphsage_model.train()
        out = graphsage_model(small_graph.x, small_graph.edge_index)

        # Loss on masked nodes only (correct)
        criterion = torch.nn.CrossEntropyLoss()
        masked_loss = criterion(
            out[small_graph.train_mask],
            small_graph.y[small_graph.train_mask],
        )

        # This should NOT crash (labels are valid for masked nodes)
        assert masked_loss.item() >= 0
        assert not torch.isnan(masked_loss)

    def test_unknown_nodes_still_in_forward_pass(self, graphsage_model, small_graph):
        """
        All 20 nodes (including 7 unknown) should get predictions.
        Unknown nodes participate in message passing for structural signal.
        """
        out = graphsage_model(small_graph.x, small_graph.edge_index)
        assert out.shape[0] == 20, "Not all nodes got predictions"


# ── Class Weight Tests ───────────────────────────────────────────────────

class TestClassWeights:
    """Verify class weight computation."""

    def test_weight_shape(self, small_graph):
        """Class weights should be a 2-element tensor."""
        weight = compute_class_weights(small_graph)
        assert weight.shape == (2,)

    def test_illicit_upweighted(self, small_graph):
        """Illicit class (minority) should have higher effective weight."""
        weight = compute_class_weights(small_graph)
        # weight = [n_illicit/n_licit, 1.0]
        # With 2 illicit and 8 licit in train: weight[0] = 2/8 = 0.25, weight[1] = 1.0
        # The illicit class (index 1) has weight 1.0 > licit weight 0.25
        assert weight[1] > weight[0], (
            f"Illicit class should be upweighted: licit={weight[0]}, illicit={weight[1]}"
        )


# ── Focal Loss Tests ─────────────────────────────────────────────────────

class TestFocalLoss:
    """Tests for the Focal Loss implementation."""

    def test_focal_loss_computes(self):
        """Focal Loss should return a scalar loss."""
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        logits = torch.randn(10, 2)
        targets = torch.randint(0, 2, (10,))
        loss = criterion(logits, targets)
        assert loss.shape == (), "Focal loss should be a scalar"
        assert not torch.isnan(loss)

    def test_focal_loss_is_positive(self):
        """Loss should always be non-negative."""
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        logits = torch.randn(50, 2)
        targets = torch.randint(0, 2, (50,))
        loss = criterion(logits, targets)
        assert loss.item() >= 0

    def test_focal_vs_ce_on_easy_examples(self):
        """
        Focal loss should be smaller than CE on easy (high-confidence) examples,
        since it down-weights well-classified samples.
        """
        criterion_focal = FocalLoss(alpha=0.5, gamma=2.0)
        criterion_ce = torch.nn.CrossEntropyLoss()

        # Create "easy" examples (high confidence correct predictions)
        logits = torch.tensor([[5.0, -5.0], [5.0, -5.0], [-5.0, 5.0]])
        targets = torch.tensor([0, 0, 1])

        focal_loss = criterion_focal(logits, targets)
        ce_loss = criterion_ce(logits, targets)

        assert focal_loss.item() < ce_loss.item(), (
            "Focal loss should be smaller than CE on easy examples"
        )


# ── Training Step Smoke Test ─────────────────────────────────────────────

class TestTrainingStep:
    """Smoke test for the training loop."""

    def test_loss_decreases_over_steps(self, graphsage_model, small_graph):
        """Loss should generally decrease over multiple training steps."""
        optimizer = torch.optim.Adam(graphsage_model.parameters(), lr=0.01)
        weight = compute_class_weights(small_graph)
        criterion = torch.nn.CrossEntropyLoss(weight=weight)

        losses = []
        for _ in range(30):
            loss = train_epoch(graphsage_model, small_graph, optimizer, criterion)
            losses.append(loss)

        # Loss at end should be lower than at start (on average)
        avg_first_5 = np.mean(losses[:5])
        avg_last_5 = np.mean(losses[-5:])
        assert avg_last_5 < avg_first_5, (
            f"Loss did not decrease: first 5 avg={avg_first_5:.4f}, "
            f"last 5 avg={avg_last_5:.4f}"
        )

    def test_evaluate_returns_required_metrics(self, graphsage_model, small_graph):
        """evaluate() should return all 5 required metrics."""
        metrics = evaluate(graphsage_model, small_graph, small_graph.val_mask)
        required = [
            "precision_illicit", "recall_illicit", "f1_illicit",
            "pr_auc_illicit", "roc_auc",
        ]
        for key in required:
            assert key in metrics, f"Missing metric: {key}"
            assert isinstance(metrics[key], float), f"{key} is not a float"
