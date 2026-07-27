from fastapi import APIRouter, Depends, Path, Query
from typing import Annotated, Literal
from api.deps import get_neo4j_driver
from api.middleware.auth import verify_api_key

router = APIRouter(dependencies=[Depends(verify_api_key)])

@router.get("/list")
async def list_clusters(
    sort: Literal["avg_risk", "max_risk", "size"] = "avg_risk",
    min_risk: float = 0.0,
    min_size: int = 10,
    limit: int = Query(default=100, le=500),
    driver=Depends(get_neo4j_driver),
):
    """Return top clusters sorted by avg_risk (for Cluster Explorer tab)."""
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (t:Transaction)
            WHERE t.communityId IS NOT NULL
            WITH t.communityId AS communityId,
                 count(t) AS size,
                 avg(t.risk_score) AS avg_risk,
                 max(t.risk_score) AS max_risk
            WHERE size >= $min_size AND avg_risk >= $min_risk
            RETURN communityId, size, avg_risk, max_risk
            ORDER BY avg_risk DESC
            LIMIT $limit
            """,
            min_size=min_size, min_risk=min_risk, limit=limit
        )
        return [r.data() async for r in result]

@router.get("/{cluster_id}")
async def get_cluster(
    cluster_id: Annotated[int, Path(ge=0, le=2_147_483_647)],
    driver=Depends(get_neo4j_driver),
):
    """Return top 20 highest-risk wallets in a community."""
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (t:Transaction {communityId: $cid})
            RETURN t.txId AS txId, t.risk_score AS risk_score,
                   t.predicted_label AS predicted_label, t.timeStep AS timeStep
            ORDER BY t.risk_score DESC
            LIMIT 20
            """,
            cid=cluster_id
        )
        wallets = [r.data() async for r in result]

    async with driver.session() as session:
        summary = await session.run(
            """
            MATCH (t:Transaction {communityId: $cid})
            RETURN count(t) AS size, avg(t.risk_score) AS avg_risk,
                   max(t.risk_score) AS max_risk
            """,
            cid=cluster_id
        )
        stats = (await summary.single()).data()

    return {"communityId": cluster_id, **stats, "wallets": wallets}
