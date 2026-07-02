"""Central configuration for the recommender system (v1)."""

from pathlib import Path

# Project layout
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

RATINGS_CSV = DATA_DIR / "ratings.csv"
MODEL_PATH = MODELS_DIR / "model.pt"

# Data
# A rating >= this threshold is treated as a positive (liked) example.
POSITIVE_THRESHOLD = 4.0

# Model
EMBEDDING_DIM = 32

# Training
EPOCHS = 5
BATCH_SIZE = 1024
LEARNING_RATE = 0.01
