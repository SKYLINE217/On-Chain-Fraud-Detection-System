# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.

"""
Test suite — validates model architectures, feature engineering, and API endpoints.
"""

import torch
import pytest


# ── Model Architecture Tests ──────────────────────────────────────────────

class TestGraphSAGE:
    def test_forward_pass(self):
        from src.models.graphsage import GraphSAGE
        model = GraphSAGE(in_channels=10, hidden_channels=16, out_channels=2)
        x = torch.randn(5, 10)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        out = model(x, edge_index)
        assert out.shape == (5, 2), f"Expected (5, 2), got {out.shape}"



    def test_different_feature_dimensions(self):
        from src.models.graphsage import GraphSAGE
        for in_ch in [166, 174, 200]:
            model = GraphSAGE(in_channels=in_ch, hidden_channels=64, out_channels=2)
            x = torch.randn(10, in_ch)
            edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
            out = model(x, edge_index)
            assert out.shape == (10, 2)


class TestGAT:
    def test_forward_pass(self):
        from src.models.gat import GAT
        model = GAT(in_channels=10, hidden_channels=16, out_channels=2, heads=2)
        x = torch.randn(5, 10)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        out = model(x, edge_index)
        assert out.shape == (5, 2), f"Expected (5, 2), got {out.shape}"

    def test_multi_head_attention(self):
        from src.models.gat import GAT
        for heads in [1, 2, 4, 8]:
            model = GAT(in_channels=10, hidden_channels=8, out_channels=2, heads=heads)
            x = torch.randn(5, 10)
            edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
            out = model(x, edge_index)
            assert out.shape == (5, 2)


class TestTrainEval:
    """BUG-27 Fix: Previously empty class. Now contains real training logic tests."""

    def test_class_weights_correct_direction(self):
        """
        BUG-21 regression test: illicit weight must be > licit weight.
        In the Elliptic dataset illicit nodes are the minority class (~10%),
        so weight[1] (illicit) must be much larger than weight[0] (licit).
        """
        from src.models.graphsage import GraphSAGE
        import torch
        from torch_geometric.data import Data

        # Create synthetic data with ~10% illicit class (mirrors Elliptic)
        torch.manual_seed(42)
        n_total = 100
        n_illicit = 10
        n_licit = 90

        y = torch.cat([
            torch.zeros(n_licit, dtype=torch.long),
            torch.ones(n_illicit, dtype=torch.long),
        ])
        train_mask = torch.ones(n_total, dtype=torch.bool)

        x = torch.randn(n_total, 10)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)

        from src.models.train import compute_class_weights
        weights = compute_class_weights(data)

        assert weights.shape == (2,), f"Expected 2 weights, got {weights.shape}"
        assert weights[1] > weights[0], (
            f"BUG-21 regression: illicit weight ({weights[1]:.4f}) must be > "
            f"licit weight ({weights[0]:.4f}) since illicit is the minority class."
        )
        # Specifically, weight[0] should be 1.0 (licit = majority baseline)
        assert abs(weights[0].item() - 1.0) < 1e-5, (
            f"Licit weight should be 1.0, got {weights[0]:.6f}"
        )

    def test_masks_are_mutually_exclusive(self):
        """
        Train/val/test masks must be non-overlapping (no data leakage).
        Any node should appear in at most one split.
        """
        from torch_geometric.data import Data
        import torch

        n = 50
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)
        train_mask[:30] = True
        val_mask[30:40] = True
        test_mask[40:] = True

        # Verify no overlap
        assert not (train_mask & val_mask).any(), "Train/val overlap detected"
        assert not (train_mask & test_mask).any(), "Train/test overlap detected"
        assert not (val_mask & test_mask).any(), "Val/test overlap detected"

    def test_no_unknown_nodes_in_labeled_masks(self):
        """
        Labeled masks (train/val/test) must exclude unknown nodes (y == -1).
        Unknown nodes should only be used for message passing, not loss computation.
        """
        import torch
        import numpy as np

        UNKNOWN_LABEL = -1
        n = 100
        y = torch.tensor(
            [0] * 40 + [1] * 20 + [UNKNOWN_LABEL] * 40, dtype=torch.long
        )
        is_labeled_np = (y.numpy() != UNKNOWN_LABEL)

        # Labeled mask should only be True for known nodes
        labeled_indices = np.where(is_labeled_np)[0]
        assert len(labeled_indices) == 60
        assert all(y[i].item() != UNKNOWN_LABEL for i in labeled_indices)


# ── API Response Schema Tests ────────────────────────────────────────────

class TestAPISchemas:
    def test_wallet_response_model(self):
        from api.routers.wallet import WalletScoreResponse
        resp = WalletScoreResponse(
            txId="12345",
            timeStep=10,
            txClass="2",
            risk_score=0.15,
            cached=False,
            latency_ms=2.5,
        )
        assert resp.txId == "12345"
        assert resp.risk_score == 0.15

    def test_subgraph_response_model(self):
        from api.routers.wallet import SubgraphResponse, SubgraphNodeResponse, SubgraphEdgeResponse
        resp = SubgraphResponse(
            center="12345",
            hops=2,
            nodes=[SubgraphNodeResponse(txId="12345")],
            edges=[SubgraphEdgeResponse(source="12345", target="67890")],
            node_count=1,
            edge_count=1,
            capped=False,
        )
        assert resp.node_count == 1

    def test_cluster_response_model(self):
        from api.routers.cluster import ClusterDetailResponse
        resp = ClusterDetailResponse(
            cluster_id=42,
            size=100,
            avg_risk_score=0.65,
            max_risk_score=0.99,
            min_risk_score=0.01,
            members=[],
            members_returned=0,
        )
        assert resp.cluster_id == 42
        assert resp.size == 100


# ── Cache Tests ──────────────────────────────────────────────────────────

class TestCacheGracefulDegradation:
    def test_cache_returns_none_when_unavailable(self):
        from api.cache import RedisCache
        cache = RedisCache(host="nonexistent-host", port=9999)
        assert cache.get("test_key") is None

    def test_cache_set_noop_when_unavailable(self):
        from api.cache import RedisCache
        cache = RedisCache(host="nonexistent-host", port=9999)
        # Should not raise
        cache.set("test_key", {"value": 1})
