# api/models/responses.py
"""
Pydantic response models for FastAPI output serialization.
Matches the frontend.md §7 type definitions exactly.

Compliance Disclaimer: This system is a research and portfolio
demonstration only. Not a certified AML/CFT compliance tool.
"""
from pydantic import BaseModel
from typing import Optional


class WalletResponse(BaseModel):
    address: str
    risk_score: float
    predicted_label: str
    confidence: float
    timeStep: int
    communityId: Optional[int] = None


class SubgraphNode(BaseModel):
    id: str
    risk_score: float = 0.0
    predicted_label: str = "unknown"
    communityId: int = -1
    timeStep: int = 0


class SubgraphEdge(BaseModel):
    src: str
    dst: str


class SubgraphResponse(BaseModel):
    address: str
    nodes: list[SubgraphNode]
    edges: list[SubgraphEdge]
    hops: int
    node_count: int


class PathNode(BaseModel):
    id: str
    risk_score: Optional[float] = None
    predicted_label: Optional[str] = None
    timeStep: Optional[int] = None


class PathResponse(BaseModel):
    src: str
    dst: str
    found: bool
    path_nodes: list[PathNode]
    hops: Optional[int] = None


class ClusterSummary(BaseModel):
    communityId: int
    size: int
    avg_risk: float
    max_risk: float


class ClusterWallet(BaseModel):
    txId: str
    risk_score: Optional[float] = None
    predicted_label: Optional[str] = None
    timeStep: Optional[int] = None


class ClusterDetail(ClusterSummary):
    wallets: list[ClusterWallet]


class HealthResponse(BaseModel):
    status: str
    neo4j: str
    redis: str
    node_count: Optional[int] = None
