"""Request/response schemas for the API (v1)."""

from pydantic import BaseModel


class PredictRequest(BaseModel):
    user_id: int
    item_id: int


class PredictResponse(BaseModel):
    user_id: int
    item_id: int
    score: float
