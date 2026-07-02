"""Request/response schemas for the API (v1, v2)."""

from pydantic import BaseModel


class PredictRequest(BaseModel):
    user_id: int
    item_id: int


class PredictResponse(BaseModel):
    user_id: int
    item_id: int
    score: float


class RecommendRequest(BaseModel):
    user_id: int
    top_k: int = 10


class RecommendedItem(BaseModel):
    item_id: int
    score: float


class RecommendResponse(BaseModel):
    user_id: int
    items: list[RecommendedItem]
