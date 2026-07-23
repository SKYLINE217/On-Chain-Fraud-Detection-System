# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/explain", tags=["Explainability"])

class ExplainRequest(BaseModel):
    tx_id: str

class ExplainResponse(BaseModel):
    tx_id: str
    top_features: list

@router.post("/tx", response_model=ExplainResponse)
async def explain_transaction(request: ExplainRequest):
    return {
        "tx_id": request.tx_id,
        "top_features": ["in_degree", "out_degree", "tx_amount_std"]
    }
