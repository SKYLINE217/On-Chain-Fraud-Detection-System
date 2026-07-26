from pydantic import BaseModel, Field

class BatchScoreRequest(BaseModel):
    addresses: list[str] = Field(..., max_length=1000)
