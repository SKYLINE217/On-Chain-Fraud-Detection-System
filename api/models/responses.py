# api/models/responses.py
"""
Pydantic response models for FastAPI output serialization.
Matches the frontend.md §7 type definitions exactly.

Compliance Disclaimer: This system is a research and portfolio
demonstration only. Not a certified AML/CFT compliance tool.
"""
from pydantic import BaseModel, field_validator
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

class ShapFeature(BaseModel):
    feature_name: str
    feature_value: float
    shap_value: float

class ImportantNode(BaseModel):
    node_id: str
    importance_score: float

class ImportantEdge(BaseModel):
    src: str
    dst: str
    importance_score: float

class SubgraphExplanation(BaseModel):
    important_nodes: list[ImportantNode]
    important_edges: list[ImportantEdge]

class ExplainResponse(BaseModel):
    address: str
    shap_top_features: list[ShapFeature]
    subgraph_explanation: SubgraphExplanation
    rationale: str
    explanation_model: str
    latency_warning: str

    @field_validator("rationale")
    @classmethod
    def rationale_nonempty(cls, v: str) -> str:
        if not v or len(v) < 5:
            raise ValueError("Rationale must be a non-empty string")
        return v

    @field_validator("shap_top_features")
    @classmethod
    def shap_nonempty(cls, v):
        if len(v) == 0:
            raise ValueError("shap_top_features cannot be empty")
        return v
