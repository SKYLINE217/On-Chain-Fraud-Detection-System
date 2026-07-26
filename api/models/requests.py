# api/models/requests.py
"""
Pydantic request models for FastAPI input validation.
See security.md §4.2 for full reference.

Compliance Disclaimer: This system is a research and portfolio
demonstration only. Not a certified AML/CFT compliance tool.
"""
from pydantic import BaseModel, field_validator, model_validator
import re

ADDRESS_PATTERN = re.compile(r'^[a-zA-Z0-9_\-]+$')


class WalletRequest(BaseModel):
    address: str

    @field_validator("address")
    @classmethod
    def validate_address(cls, v: str) -> str:
        if not v or len(v) > 100:
            raise ValueError("Address must be 1-100 characters")
        if not ADDRESS_PATTERN.match(v):
            raise ValueError("Address contains invalid characters")
        return v


class SubgraphRequest(BaseModel):
    address: str
    hops: int = 2

    @field_validator("address")
    @classmethod
    def validate_address(cls, v: str) -> str:
        if not ADDRESS_PATTERN.match(v) or len(v) > 100:
            raise ValueError("Invalid address")
        return v

    @field_validator("hops")
    @classmethod
    def validate_hops(cls, v: int) -> int:
        if v < 1 or v > 2:
            raise ValueError("hops must be 1 or 2")
        return v


class PathRequest(BaseModel):
    src: str
    dst: str
    max_hops: int = 10

    @field_validator("src", "dst")
    @classmethod
    def validate_address(cls, v: str) -> str:
        if not ADDRESS_PATTERN.match(v) or len(v) > 100:
            raise ValueError("Invalid address")
        return v

    @field_validator("max_hops")
    @classmethod
    def validate_max_hops(cls, v: int) -> int:
        if v < 1 or v > 10:
            raise ValueError("max_hops must be 1-10")
        return v

    @model_validator(mode="after")
    def src_dst_different(self):
        if self.src == self.dst:
            raise ValueError("src and dst must be different addresses")
        return self


class BatchScoreRequest(BaseModel):
    addresses: list[str]

    @field_validator("addresses")
    @classmethod
    def validate_addresses(cls, v: list[str]) -> list[str]:
        if len(v) == 0 or len(v) > 1000:
            raise ValueError("Must provide 1-1000 addresses")
        for addr in v:
            if not ADDRESS_PATTERN.match(addr) or len(addr) > 100:
                raise ValueError(f"Invalid address: {addr}")
        return v
