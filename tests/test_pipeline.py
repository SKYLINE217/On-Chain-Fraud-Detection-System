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
    pass


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
