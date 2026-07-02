"""Smoke tests that run without downloaded data or a trained model.

They check the model builds, a forward pass produces valid probabilities, and
the API request/response schemas behave as expected.
"""

import torch

from src.model import MFModel
from src.schemas import PredictRequest, PredictResponse


def test_model_forward_shape():
    model = MFModel(num_users=10, num_items=20, embedding_dim=8)
    u = torch.tensor([0, 1, 2])
    i = torch.tensor([5, 6, 7])
    logits = model(u, i)
    assert logits.shape == (3,)


def test_predict_returns_probability():
    model = MFModel(num_users=10, num_items=20, embedding_dim=8)
    scores = model.predict(torch.tensor([0]), torch.tensor([1]))
    assert scores.shape == (1,)
    assert 0.0 <= scores.item() <= 1.0


def test_schemas_roundtrip():
    req = PredictRequest(user_id=1, item_id=2)
    assert req.user_id == 1
    resp = PredictResponse(user_id=1, item_id=2, score=0.5)
    assert resp.score == 0.5
