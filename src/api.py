"""FastAPI backend serving the v1 recommender.

The model is loaded once at startup (not per request) and reused for every call.

Run from the project root:  uvicorn src.api:app --reload
"""

from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException

from . import config
from .model import MFModel
from .schemas import PredictRequest, PredictResponse

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()  # runs once when the server starts
    yield
    state.clear()


app = FastAPI(title="rec-system", version="1.0", lifespan=lifespan)


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
