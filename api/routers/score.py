"""
POST /score — batch scoring for a list of addresses.
Returns pre-computed scores from Neo4j (no model inference on hot path).
"""
from fastapi import APIRouter, Depends
# Using Any instead of neo4j.AsyncDriver for typing if neo4j is missing.
try:
    from neo4j import AsyncDriver
except ImportError:
    AsyncDriver = type("AsyncDriver", (), {})

try:
    from api.deps import get_neo4j_driver
    from api.middleware.auth import verify_api_key
except ImportError:
    # Mocks for unit testing if Person A/C haven't built these yet
    async def get_neo4j_driver(): yield None
    async def verify_api_key(): pass

from api.models.requests import BatchScoreRequest

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.post("")
async def batch_score(
    body: BatchScoreRequest,
    driver: AsyncDriver = Depends(get_neo4j_driver),
):
    """
    Look up pre-computed scores for multiple addresses.
    Scores come from Neo4j (written by nightly batch job) — no GNN inference.
    Max 1000 addresses per request.
    """
    addresses = body.addresses[:1000]  # enforce cap (also in Pydantic validator)

    records = []
    if driver is not None:
        async with driver.session() as session:
            result = await session.run(
                """
                UNWIND $addresses AS addr
                MATCH (t:Transaction {txId: addr})
                RETURN t.txId AS address,
                       t.risk_score AS risk_score,
                       t.predicted_label AS predicted_label,
                       t.confidence AS confidence
                """,
                addresses=addresses
            )
            records = [r.data() async for r in result]

    # Fill in nulls for addresses not found
    found = {r.get("address") for r in records}
    for addr in addresses:
        if addr not in found:
            records.append({
                "address": addr,
                "risk_score": None,
                "predicted_label": "not_found",
                "confidence": None,
            })

    return records
