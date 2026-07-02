"""FastAPI backend serving the recommender (v1: /predict, v2: /recommend).

The model is loaded once at startup (not per request) and reused for every call.

Run from the project root:  uvicorn src.api:app --reload
"""

from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException

from . import config
from .model import MFModel
from .schemas import (
    PredictRequest,
    PredictResponse,
    RecommendedItem,
    RecommendRequest,
    RecommendResponse,
)

# Holds the loaded model and id mappings for the lifetime of the process.
state: dict = {}


def load_model() -> None:
    if not config.MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {config.MODEL_PATH}. Train it first: python -m src.train"
        )
    ckpt = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=False)
    model = MFModel(ckpt["num_users"], ckpt["num_items"], ckpt["embedding_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    state["model"] = model
    state["user2idx"] = ckpt["user2idx"]
    state["item2idx"] = ckpt["item2idx"]
    # Reverse map (index -> item_id) to translate model outputs back to real ids.
    state["idx2item"] = {idx: item_id for item_id, idx in ckpt["item2idx"].items()}
    # Candidate pool for v2: every known item. A recall layer replaces this in v3.
    state["all_item_idx"] = torch.arange(ckpt["num_items"], dtype=torch.long)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()  # runs once when the server starts
    yield
    state.clear()


app = FastAPI(title="rec-system", version="2.0", lifespan=lifespan)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    user2idx = state["user2idx"]
    item2idx = state["item2idx"]

    if req.user_id not in user2idx:
        raise HTTPException(status_code=404, detail=f"Unknown user_id: {req.user_id}")
    if req.item_id not in item2idx:
        raise HTTPException(status_code=404, detail=f"Unknown item_id: {req.item_id}")

    u = torch.tensor([user2idx[req.user_id]], dtype=torch.long)
    i = torch.tensor([item2idx[req.item_id]], dtype=torch.long)
    with torch.no_grad():
        score = state["model"].predict(u, i).item()

    return PredictResponse(user_id=req.user_id, item_id=req.item_id, score=score)


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    user2idx = state["user2idx"]
    if req.user_id not in user2idx:
        raise HTTPException(status_code=404, detail=f"Unknown user_id: {req.user_id}")
    if req.top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be >= 1")

    # v2 candidate list = all items. Score them, keep the top_k highest.
    top_idx, top_scores = state["model"].rank_items(
        user_idx=user2idx[req.user_id],
        candidate_item_idx=state["all_item_idx"],
        k=req.top_k,
    )

    idx2item = state["idx2item"]
    items = [
        RecommendedItem(item_id=idx2item[int(idx)], score=float(score))
        for idx, score in zip(top_idx.tolist(), top_scores.tolist())
    ]
    return RecommendResponse(user_id=req.user_id, items=items)
