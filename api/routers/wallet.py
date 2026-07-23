# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.cache import RedisCache

router = APIRouter(prefix="/wallet", tags=["Wallet"])
cache = RedisCache()

class WalletRequest(BaseModel):
    address: str

class WalletResponse(BaseModel):
    address: str
    risk_score: float
    is_illicit: bool

@router.post("/score", response_model=WalletResponse)
async def get_wallet_score(request: WalletRequest):
    cached_res = cache.get(f"wallet:{request.address}")
    if cached_res:
        return cached_res
        
    # Placeholder score evaluation logic
    score = 0.85 if request.address.startswith("0xbad") else 0.05
    res = {
        "address": request.address,
        "risk_score": score,
        "is_illicit": score > 0.5
    }
    
    cache.set(f"wallet:{request.address}", res)
    return res
