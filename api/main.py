# Copyright (c) 2025 On-Chain Fraud Detection System Team. All rights reserved.
# Licensed under LICENSE.md. Unauthorized copying or distribution is prohibited.

"""
FastAPI application entry point for the On-Chain Fraud Detection System.
Registers all routers, configures middleware and lifecycle hooks.

Fixes applied:
  BUG-01: CORS wildcard replaced with env-controlled explicit origin list
  BUG-12: Health check now verifies Neo4j, Redis, and model checkpoint
  BUG-13: slowapi rate limiting added on /explain
  BUG-34: All routers mounted under /api prefix for production compatibility
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.routers import wallet, explain, cluster, path
from api.neo4j_service import neo4j_service
from api import cache as cache_module

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Rate Limiting (BUG-13) ────────────────────────────────────────────────
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address)
    _rate_limiting_available = True
    logger.info("slowapi rate limiting enabled")
except ImportError:
    logger.warning("slowapi not installed — rate limiting disabled. Install with: pip install slowapi")
    limiter = None
    _rate_limiting_available = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle — initialize and cleanup resources."""
    logger.info("Starting On-Chain Fraud Detection API...")

    # BUG-02 / BUG-12: Verify critical environment variables at startup
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    if not neo4j_password:
        logger.warning(
            "NEO4J_PASSWORD environment variable is not set. "
            "API will start but Neo4j connections will fail. "
            "Copy .env.example to .env and fill in real credentials."
        )

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

# ── Rate Limiting State (BUG-13) ─────────────────────────────────────────
if _rate_limiting_available:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS — BUG-01 Fix ────────────────────────────────────────────────────
# Explicit origin list only. No wildcard with credentials.
# Set ALLOWED_ORIGINS env var as comma-separated list in production.
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,      # BUG-01: explicit list, not wildcard
    allow_credentials=False,            # BUG-01: no credentials unless needed
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# ── Register all routers under /api prefix (BUG-34) ──────────────────────
# /api prefix ensures frontend → backend routing works in both dev and prod.
app.include_router(wallet.router, prefix="/api")
app.include_router(explain.router, prefix="/api")
app.include_router(cluster.router, prefix="/api")
app.include_router(path.router, prefix="/api")


# ── Root & Health ─────────────────────────────────────────────────────────

# Mount static files for the dashboard
try:
    app.mount("/static", StaticFiles(directory="api/static"), name="static")
except Exception:
    logger.warning("Static files directory not found — skipping static mount")


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Liveness/readiness probe for container orchestration.
    BUG-12 Fix: Returns 503 if Neo4j is unavailable (not always 200).
    """
    status = {"neo4j": "unknown", "redis": "unknown", "model": "unknown"}

    # Check Neo4j connectivity
    try:
        neo4j_service.driver.verify_connectivity()
        status["neo4j"] = "ok"
    except Exception as e:
        status["neo4j"] = f"error: {str(e)[:100]}"

    # Check Redis
    _cache = cache_module.cache
    if _cache.is_available:
        status["redis"] = "ok"
    else:
        status["redis"] = "unavailable"

    # Check model checkpoint
    model_path = Path("checkpoints/best_model.pt")
    status["model"] = "ok" if model_path.exists() else "missing"

    # Neo4j is required; Redis and model are optional
    all_ok = status["neo4j"] == "ok"
    return JSONResponse(
        content={
            "status": "healthy" if all_ok else "degraded",
            "service": "onchain-fraud-detection-api",
            **status,
        },
        status_code=200 if all_ok else 503,
    )


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
            "wallet_score": "GET /api/wallet/{address}",
            "wallet_subgraph": "GET /api/wallet/{address}/subgraph?hops=2",
            "cluster_top": "GET /api/cluster/top",
            "cluster_detail": "GET /api/cluster/{cluster_id}",
            "path": "GET /api/path?src=...&dst=...",
            "explain": "POST /api/explain/{address}",
            "health": "GET /health",
        },
    }
