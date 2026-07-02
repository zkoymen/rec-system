"""Dataset loading and label preparation (v1).

Downloads the small MovieLens dataset on first use, converts ratings into binary
"liked" labels, and builds contiguous integer indices for users and items so
they can be used directly as embedding lookups.
"""

import io
import urllib.request
import zipfile

import pandas as pd

from . import config

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def ensure_ratings_csv() -> None:
    """Download MovieLens ratings into data/ratings.csv if not present."""
    if config.RATINGS_CSV.exists():
        return

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MovieLens dataset from {MOVIELENS_URL} ...")
    with urllib.request.urlopen(MOVIELENS_URL) as resp:
        raw = resp.read()

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        with zf.open("ml-latest-small/ratings.csv") as f:
            df = pd.read_csv(f)

    df.to_csv(config.RATINGS_CSV, index=False)
    print(f"Saved ratings to {config.RATINGS_CSV} ({len(df)} rows).")


def load_dataset():
    """Return (dataframe, user2idx, item2idx).

    The dataframe gains three columns: user_idx, item_idx, label.
    """
    ensure_ratings_csv()
    df = pd.read_csv(config.RATINGS_CSV)

    # MovieLens columns: userId, movieId, rating, timestamp
    user2idx = {uid: i for i, uid in enumerate(df["userId"].unique())}
    item2idx = {iid: i for i, iid in enumerate(df["movieId"].unique())}

    df["user_idx"] = df["userId"].map(user2idx)
    df["item_idx"] = df["movieId"].map(item2idx)
    df["label"] = (df["rating"] >= config.POSITIVE_THRESHOLD).astype("float32")

    return df, user2idx, item2idx
