"""Train the v1 recommender and save it as models/model.pt.

Run from the project root:  python -m src.train
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .data import load_dataset
from .model import MFModel


def train() -> None:
    df, user2idx, item2idx = load_dataset()
    num_users, num_items = len(user2idx), len(item2idx)
    print(f"Users: {num_users}, Items: {num_items}, Interactions: {len(df)}")

    users = torch.tensor(df["user_idx"].values, dtype=torch.long)
    items = torch.tensor(df["item_idx"].values, dtype=torch.long)
    labels = torch.tensor(df["label"].values, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(users, items, labels),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )

    model = MFModel(num_users, num_items, config.EMBEDDING_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(1, config.EPOCHS + 1):
        total_loss = 0.0
        for u, i, y in loader:
            optimizer.zero_grad()
            logits = model(u, i)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        print(f"Epoch {epoch}/{config.EPOCHS} - loss: {total_loss / len(df):.4f}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "user2idx": user2idx,
        "item2idx": item2idx,
        "embedding_dim": config.EMBEDDING_DIM,
        "num_users": num_users,
        "num_items": num_items,
    }
    torch.save(checkpoint, config.MODEL_PATH)
    print(f"Saved model to {config.MODEL_PATH}")


if __name__ == "__main__":
    train()
