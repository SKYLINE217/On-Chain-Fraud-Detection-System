from pydantic import BaseModel, field_validator

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
