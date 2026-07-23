# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

"""
FastAPI application entry point for the On-Chain Fraud Detection System.
Registers all routers, configures middleware and lifecycle hooks.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import wallet, explain, cluster, path
from api.neo4j_service import neo4j_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle — initialize and cleanup resources."""
    logger.info("Starting On-Chain Fraud Detection API...")
    yield
    logger.info("Shutting down — closing Neo4j driver...")
    neo4j_service.close()


app = FastAPI(
    title="On-Chain Fraud Detection API",
    description=(
        "GNN-based fraud detection and risk scoring for blockchain transactions. "
        "Provides wallet risk scoring, subgraph exploration, cluster analysis, "
        "and transaction path tracing."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow dashboard and dev origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register all routers ─────────────────────────────────────────────────
app.include_router(wallet.router)
app.include_router(explain.router)
app.include_router(cluster.router)
app.include_router(path.router)


# ── Root & Health ────────────────────────────────────────────────────────

from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

# Mount static files for the dashboard
app.mount("/static", StaticFiles(directory="api/static"), name="static")

@app.get("/health", tags=["Health"])
async def health_check():
    """Liveness probe for container orchestration."""
    return {"status": "healthy", "service": "onchain-fraud-detection-api"}

@app.get("/", tags=["Root"])
async def root():
    # Redirect to the Explainability Dashboard
    return RedirectResponse(url="/static/index.html")

@app.get("/api-docs", tags=["Root"])
async def api_docs():
    return {
        "project": "On-Chain Fraud Detection System",
        "version": "1.0.0",
        "endpoints": {
            "wallet_score": "GET /wallet/{address}",
            "wallet_subgraph": "GET /wallet/{address}/subgraph?hops=2",
            "cluster_top": "GET /cluster/top",
            "cluster_detail": "GET /cluster/{cluster_id}",
            "path": "GET /path?src=...&dst=...",
            "explain": "POST /explain/tx",
            "health": "GET /health",
        },
    }
