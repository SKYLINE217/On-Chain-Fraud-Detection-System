# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

"""
Cluster router — GET /cluster/{cluster_id} and GET /clusters/top.

Latency budget: <5s.
Feeds the Cluster Explorer dashboard tab.
"""

import time
import logging
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

from api.neo4j_service import neo4j_service

router = APIRouter(prefix="/cluster", tags=["Cluster"])
logger = logging.getLogger(__name__)


# ── Response Models ──────────────────────────────────────────────────────

class ClusterMemberResponse(BaseModel):
    txId: str
    timeStep: Optional[int] = None
    txClass: Optional[str] = None
    risk_score: Optional[float] = None
    predicted_label: Optional[str] = None
    confidence: Optional[float] = None
    pageRank: Optional[float] = None


class ClusterDetailResponse(BaseModel):
    cluster_id: int
    size: int
    avg_risk_score: Optional[float] = None
    max_risk_score: Optional[float] = None
    min_risk_score: Optional[float] = None
    members: list[ClusterMemberResponse]
    members_returned: int
    latency_ms: float = 0.0


class ClusterSummaryResponse(BaseModel):
    cluster_id: int
    size: int
    avg_risk: Optional[float] = None


# ── Endpoints ────────────────────────────────────────────────────────────

@router.get(
    "/top",
    response_model=list[ClusterSummaryResponse],
    summary="Get top risky clusters",
    description="Returns top community clusters ranked by average risk score. "
                "Used by the Cluster Explorer dashboard tab.",
)
async def get_top_clusters(
    limit: int = Query(default=100, ge=1, le=500, description="Max clusters to return"),
):
    """
    GET /cluster/top?limit=100
    Owner: Person A | Latency budget: <5s
    """
    t0 = time.perf_counter()
    clusters = neo4j_service.get_top_clusters(limit=limit)
    latency = (time.perf_counter() - t0) * 1000
    logger.info("Top clusters: %d returned (%.1fms)", len(clusters), latency)
    return clusters


@router.get(
    "/{cluster_id}",
    response_model=ClusterDetailResponse,
    summary="Get cluster details",
    description="Fetch all transactions in a community cluster, "
                "sorted by risk score descending.",
)
async def get_cluster_detail(
    cluster_id: int,
    limit: int = Query(default=100, ge=1, le=500, description="Max members to return"),
):
    """
    GET /cluster/{cluster_id}?limit=100
    Owner: Person A | Latency budget: <5s
    """
    t0 = time.perf_counter()
    cluster = neo4j_service.get_cluster(cluster_id, limit=limit)

    if cluster["size"] == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Cluster {cluster_id} not found or is empty.",
        )

    latency = (time.perf_counter() - t0) * 1000
    cluster["latency_ms"] = round(latency, 2)
    logger.info("Cluster %d: size=%d, members_returned=%d (%.1fms)",
                cluster_id, cluster["size"], cluster["members_returned"], latency)
    return cluster
