# api/routers/wallet.py
"""
Wallet lookup, subgraph, and path endpoints.
See person_a_stages.md §3.3 for full reference.

Compliance Disclaimer: This system is a research and portfolio
demonstration only. Not a certified AML/CFT compliance tool.
"""
from fastapi import APIRouter, Depends, HTTPException
from neo4j import AsyncDriver
from redis.asyncio import Redis
import json

from api.deps import get_neo4j_driver, get_redis
from api.middleware.auth import verify_api_key
from api.models.responses import WalletResponse, SubgraphResponse, PathResponse

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.get("/{address}", response_model=WalletResponse)
async def get_wallet(
    address: str,
    driver: AsyncDriver = Depends(get_neo4j_driver),
    redis: Redis = Depends(get_redis),
):
    # 1. Cache check
    cache_key = f"score:{address}"
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    # 2. Neo4j lookup (uses txId_idx — no table scan)
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (t:Transaction {txId: $address})
            RETURN t.txId AS address, t.risk_score AS risk_score,
                   t.predicted_label AS predicted_label, t.confidence AS confidence,
                   t.timeStep AS timeStep, t.communityId AS communityId
            """,
            address=address
        )
        record = await result.single()

    if not record:
        raise HTTPException(status_code=404, detail=f"Address not found: {address}")

    payload = dict(record)

    # Default values for unscored nodes
    if payload.get("risk_score") is None:
        payload["risk_score"] = 0.0
    if payload.get("predicted_label") is None:
        payload["predicted_label"] = "unknown"
    if payload.get("confidence") is None:
        payload["confidence"] = 0.0

    # 3. Cache write (TTL = 1 hour)
    await redis.set(cache_key, json.dumps(payload), ex=3600)
    return payload


@router.get("/{address}/subgraph", response_model=SubgraphResponse)
async def get_subgraph(
    address: str,
    hops: int = 2,
    driver: AsyncDriver = Depends(get_neo4j_driver),
):
    safe_hops = min(hops, 2)  # hard cap — defense in depth

    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (start:Transaction {txId: $address})
            CALL apoc.path.subgraphAll(start, {maxLevel: $max_level, limit: 200})
            YIELD nodes, relationships
            RETURN nodes, relationships
            """,
            address=address,
            max_level=safe_hops,
        )
        record = await result.single()

    if not record:
        raise HTTPException(status_code=404, detail=f"Address not found: {address}")

    nodes = [
        {
            "id": n["txId"],
            "risk_score": n.get("risk_score", 0.0) or 0.0,
            "predicted_label": n.get("predicted_label", "unknown") or "unknown",
            "communityId": n.get("communityId", -1) or -1,
            "timeStep": n.get("timeStep", 0) or 0,
        }
        for n in record["nodes"][:200]  # hard cap in Python too
    ]
    edges = [
        {"src": r.start_node["txId"], "dst": r.end_node["txId"]}
        for r in record["relationships"]
    ]

    return {
        "address": address,
        "nodes": nodes,
        "edges": edges,
        "hops": safe_hops,
        "node_count": len(nodes),
    }


@router.get("/path/find", response_model=PathResponse)
async def get_path(
    src: str,
    dst: str,
    max_hops: int = 10,
    driver: AsyncDriver = Depends(get_neo4j_driver),
):
    """Shortest path between two addresses (Cypher shortestPath)."""
    safe_max = min(max_hops, 10)

    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (src:Transaction {txId: $src}), (dst:Transaction {txId: $dst})
            MATCH path = shortestPath((src)-[:FLOWS_TO*..10]->(dst))
            RETURN [n IN nodes(path) | {
                id: n.txId,
                risk_score: n.risk_score,
                predicted_label: n.predicted_label,
                timeStep: n.timeStep
            }] AS path_nodes,
            length(path) AS hops
            """,
            src=src, dst=dst
        )
        record = await result.single()

    if not record:
        return {"src": src, "dst": dst, "found": False, "path_nodes": [], "hops": None}

    return {
        "src": src,
        "dst": dst,
        "found": True,
        "path_nodes": record["path_nodes"],
        "hops": record["hops"],
    }
