# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

"""
FastAPI application entry point for the On-Chain Fraud Detection System.
Registers all routers and configures middleware.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import wallet, explain

app = FastAPI(
    title="On-Chain Fraud Detection API",
    description="GNN-based fraud detection and risk scoring for blockchain transactions.",
    version="1.0.0",
)

# CORS — allow dashboard and dev origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(wallet.router)
app.include_router(explain.router)


@app.get("/health", tags=["Health"])
async def health_check():
    """Liveness probe for container orchestration."""
    return {"status": "healthy", "service": "onchain-fraud-detection-api"}


@app.get("/", tags=["Root"])
async def root():
    return {
        "project": "On-Chain Fraud Detection System",
        "version": "1.0.0",
        "endpoints": ["/wallet/score", "/explain/tx", "/health"],
    }
