# api/main.py
"""
FastAPI application factory for onchain-fraud-gnn.
See person_a_stages.md §3.1, security.md §3.2, §5, §6 for full reference.

Compliance Disclaimer: This system is a research and portfolio
demonstration only. Not a certified AML/CFT compliance tool.
"""
from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import logging

from api.routers import wallet, health, cluster
from api.middleware.auth import verify_api_key

logger = logging.getLogger(__name__)

app = FastAPI(
    title="onchain-fraud-gnn API",
    description="On-chain fraud/AML detection using Graph Neural Networks. Research/portfolio only.",
    version="2.0.0",
    docs_url="/docs",  # OpenAPI docs (dev only in production)
)

# ── CORS (locked to BFF only) ──────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://bff:3000", "http://localhost:3000"],  # Docker internal + dev
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
    allow_credentials=False,
)

# ── Rate Limiting (secondary layer — BFF is primary) ───
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded. Try again later."}
    )


# ── Exception Handlers (no info leakage — security.md §10) ─
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    # Never return exc details to client in production
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Pydantic errors are safe to return (field-level only, no internals)
    return JSONResponse(
        status_code=422,
        content={"error": "Validation failed", "details": exc.errors()}
    )


# ── Register Routers ───────────────────────────────────
app.include_router(
    wallet.router,
    prefix="/wallet",
    tags=["wallet"],
)

app.include_router(
    health.router,
    tags=["health"],
)

app.include_router(
    cluster.router,
    prefix="/cluster",
    tags=["cluster"],
)
