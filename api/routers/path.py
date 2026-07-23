# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

"""
Path router — GET /path?src=...&dst=...

Shortest path between two transactions. Max depth 10 hops.
Feeds the Transaction Path dashboard tab.
"""

import time
import logging
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from api.neo4j_service import neo4j_service

router = APIRouter(prefix="/path", tags=["Path"])
logger = logging.getLogger(__name__)


class PathResponse(BaseModel):
    source: str
    target: str
    path_found: bool
    path_nodes: list[str]
    path_length: int
    latency_ms: float = 0.0


@router.get(
    "/",
    response_model=PathResponse,
    summary="Find shortest path between two transactions",
    description="Find the shortest path between two transaction addresses. "
                "Capped at 10 hops. Returns 'no path found' if no path exists.",
)
async def find_shortest_path(
    src: str = Query(..., description="Source transaction ID"),
    dst: str = Query(..., description="Destination transaction ID"),
):
    """
    GET /path?src={txId1}&dst={txId2}
    Owner: Person A | Latency budget: <5s
    """
    t0 = time.perf_counter()
    result = neo4j_service.get_shortest_path(src, dst)
    latency = (time.perf_counter() - t0) * 1000
    result["latency_ms"] = round(latency, 2)

    logger.info(
        "Path %s → %s: %s (length=%d, %.1fms)",
        src, dst,
        "FOUND" if result["path_found"] else "NOT FOUND",
        result["path_length"], latency,
    )
    return result
