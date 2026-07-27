
"""
API key verification middleware for FastAPI.
All requests must include a valid X-API-Key header.
See security.md §2.1 for full reference.

Compliance Disclaimer: This system is a research and portfolio
demonstration only. Not a certified AML/CFT compliance tool.
"""
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
import hmac
import os

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key() -> str:
    """Re-read from env on every call (allows runtime rotation via env update)."""
    if "API_KEY" not in os.environ:
        raise ValueError("API_KEY environment variable is not set")
    return os.environ["API_KEY"]

async def verify_api_key(api_key: str = Security(api_key_header)):
    valid = get_api_key()
    if api_key is None or not hmac.compare_digest(api_key, valid):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "APIKey"},
        )
    return api_key
