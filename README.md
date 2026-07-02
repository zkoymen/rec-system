# rec-system

An end-to-end recommender system built step by step, focused on practical AI
engineering — not just model training.

The project grows through small, always-working versions:

1. **v1** — train a simple model and serve a `/predict` endpoint (FastAPI).
2. **v2** — add a `/recommend` endpoint that returns top items for a user.
3. **v3** — add a recall layer to fetch candidates.
4. **v4** — add a ranking layer to score and sort candidates.
5. **v5** — improve serving structure, logging, and health checks.

## Quickstart (v1)

```bash
pip install -r requirements.txt
python -m src.train                       # downloads data, trains, saves models/model.pt
uvicorn src.api:app --reload              # serve the API
```

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "item_id": 1}'
# -> {"user_id": 1, "item_id": 1, "score": 0.92}
```

## Tech Stack

Python · PyTorch · FastAPI · Uvicorn · Pandas · NumPy

## Status

Early setup. More coming as each version lands.
