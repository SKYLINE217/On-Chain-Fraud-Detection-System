# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

"""
Wallet router — GET /wallet/{address} with Redis-cached Neo4j lookups.

Latency budget: <5s (cached hit), <5s (Neo4j index lookup).
Only the hot-path /wallet/{address} gets Redis caching (TTL 1 hour).
"""

import time
import logging
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

from api.cache import RedisCache
from api.neo4j_service import neo4j_service

router = APIRouter(prefix="/wallet", tags=["Wallet"])
cache = RedisCache()
logger = logging.getLogger(__name__)


# ── Response Models ──────────────────────────────────────────────────────

class WalletScoreResponse(BaseModel):
    txId: str
    timeStep: Optional[int] = None
    txClass: Optional[str] = None
    risk_score: Optional[float] = None
    predicted_label: Optional[str] = None
    confidence: Optional[float] = None
    communityId: Optional[int] = None
    pageRank: Optional[float] = None
    clusteringCoeff: Optional[float] = None
    cached: bool = False
    latency_ms: float = 0.0


class SubgraphNodeResponse(BaseModel):
    txId: str
    timeStep: Optional[int] = None
    txClass: Optional[str] = None
    risk_score: Optional[float] = None
    predicted_label: Optional[str] = None
    confidence: Optional[float] = None
    communityId: Optional[int] = None
    pageRank: Optional[float] = None


class SubgraphEdgeResponse(BaseModel):
    source: str
    target: str


class SubgraphResponse(BaseModel):
    center: str
    hops: int
    nodes: list[SubgraphNodeResponse]
    edges: list[SubgraphEdgeResponse]
    node_count: int
    edge_count: int
    capped: bool
    latency_ms: float = 0.0


# ── Endpoints ────────────────────────────────────────────────────────────

@router.get(
    "/{address}",
    response_model=WalletScoreResponse,
    summary="Get wallet risk score",
    description="Fetch risk score and metadata for a transaction address. "
                "Results are cached in Redis (TTL 1h). First hit queries Neo4j; "
                "subsequent hits served from cache.",
)
async def get_wallet_score(address: str):
    """
    GET /wallet/{address}
    Owner: Person A | Latency budget: <5s
    """
    t0 = time.perf_counter()

    # ── Cache hit path ──
    cached = cache.get(f"score:{address}")
    if cached:
        latency = (time.perf_counter() - t0) * 1000
        cached["cached"] = True
        cached["latency_ms"] = round(latency, 2)
        logger.info("Cache HIT for %s (%.1fms)", address, latency)
        return cached

    # ── Neo4j lookup path ──
    wallet = neo4j_service.get_wallet(address)
    if wallet is None:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction {address} not found in the graph.",
        )

    latency = (time.perf_counter() - t0) * 1000
    response = {
        "txId": wallet["txId"],
        "timeStep": wallet.get("timeStep"),
        "txClass": wallet.get("txClass"),
        "risk_score": wallet.get("risk_score"),
        "predicted_label": wallet.get("predicted_label"),
        "confidence": wallet.get("confidence"),
        "communityId": wallet.get("communityId"),
        "pageRank": wallet.get("pageRank"),
        "clusteringCoeff": wallet.get("clusteringCoeff"),
        "cached": False,
        "latency_ms": round(latency, 2),
    }

    # Store in Redis — TTL 1 hour
    cache.set(f"score:{address}", response, ttl=3600)
    logger.info("Cache MISS for %s → stored (%.1fms)", address, latency)
    return response


@router.get(
    "/{address}/subgraph",
    response_model=SubgraphResponse,
    summary="Get k-hop ego-graph",
    description="Fetch the k-hop neighborhood subgraph around a transaction. "
                "HARD CAPS: max 2 hops, max 200 nodes returned. "
                "Returns nodes with risk scores and community IDs, plus edges.",
)
async def get_wallet_subgraph(
    address: str,
    hops: int = Query(default=2, ge=1, le=2, description="Number of hops (max 2)"),
    max_nodes: int = Query(default=200, ge=1, le=200, description="Max nodes to return (max 200)"),
):
    """
    GET /wallet/{address}/subgraph?hops=2&max_nodes=200
    Owner: Person A | Latency budget: <5s (bounded query)
    """
    t0 = time.perf_counter()

    # Try APOC first, fall back to variable-length paths
    try:
        subgraph = neo4j_service.get_subgraph(address, hops=hops, max_nodes=max_nodes)
    except Exception:
        logger.warning("APOC subgraph failed for %s, using fallback", address)
        subgraph = neo4j_service.get_subgraph_no_apoc(address, hops=hops, max_nodes=max_nodes)

    if subgraph["node_count"] == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction {address} not found or has no neighbors.",
        )

    latency = (time.perf_counter() - t0) * 1000
    subgraph["latency_ms"] = round(latency, 2)
    logger.info(
        "Subgraph for %s: %d nodes, %d edges (%.1fms)%s",
        address, subgraph["node_count"], subgraph["edge_count"],
        latency, " [CAPPED]" if subgraph["capped"] else "",
    )
    return subgraph
