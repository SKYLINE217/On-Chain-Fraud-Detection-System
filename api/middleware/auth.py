# api/middleware/auth.py
"""
API key verification middleware for FastAPI.
All requests must include a valid X-API-Key header.
See security.md §2.1 for full reference.

Compliance Disclaimer: This system is a research and portfolio
demonstration only. Not a certified AML/CFT compliance tool.
"""
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
import os

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
VALID_API_KEY = os.environ.get("API_KEY", "dev_key_change_me")


async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key is None or api_key != VALID_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key"
        )
    return api_key
