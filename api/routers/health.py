# api/routers/health.py
"""
Health check endpoint — verifies Neo4j and Redis connectivity.

Compliance Disclaimer: This system is a research and portfolio
demonstration only. Not a certified AML/CFT compliance tool.
"""
from fastapi import APIRouter, Depends
from api.deps import get_neo4j_driver, get_redis

from api.middleware.auth import verify_api_key

router = APIRouter()


@router.get("/health")
async def health_check():
    """Public endpoint: minimal info."""
    return {"status": "ok"}


@router.get("/health/detailed", dependencies=[Depends(verify_api_key)])
async def health_check_detailed(
    driver=Depends(get_neo4j_driver),
    redis=Depends(get_redis),
):
    neo4j_status = "unhealthy"
    redis_status = "unhealthy"
    node_count = None

    # Check Neo4j
    try:
        async with driver.session() as session:
            result = await session.run("MATCH (n:Transaction) RETURN count(n) AS cnt")
            record = await result.single()
            node_count = record["cnt"]
            neo4j_status = "healthy"
    except Exception:
        pass

    # Check Redis
    try:
        pong = await redis.ping()
        if pong:
            redis_status = "healthy"
    except Exception:
        pass

    overall = "healthy" if neo4j_status == "healthy" and redis_status == "healthy" else "degraded"

    return {
        "status": overall,
        "neo4j": neo4j_status,
        "redis": redis_status,
        "node_count": node_count,
    }
