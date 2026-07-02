"""Simple matrix-factorization recommender model (v1).

Each user and item gets an embedding vector. The predicted "like" score is the
dot product of the two vectors (plus biases) passed through a sigmoid, so the
output is a probability in [0, 1].
"""

import torch
import torch.nn as nn


class MFModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        # Small init keeps early scores near 0.5.
        nn.init.normal_(self.user_emb.weight, std=0.05)
        nn.init.normal_(self.item_emb.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """Return raw logits (pre-sigmoid) for numerical stability in training."""
        dot = (self.user_emb(user_idx) * self.item_emb(item_idx)).sum(dim=-1)
        logit = dot + self.user_bias(user_idx).squeeze(-1) + self.item_bias(item_idx).squeeze(-1)
        return logit

    def predict(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """Return like-probability in [0, 1]."""
        return torch.sigmoid(self.forward(user_idx, item_idx))

    @torch.no_grad()
    def rank_items(self, user_idx: int, candidate_item_idx: torch.Tensor, k: int):
        """Score candidate items for one user and return the top-k.

        This is the "ranking" step: given a user and a set of candidate items,
        score them all in one batch and keep the highest. In later versions the
        candidate set will come from a separate recall layer instead of being
        every item.

        Returns (top_item_idx, top_scores), both 1-D tensors of length <= k.
        """
        users = torch.full_like(candidate_item_idx, user_idx)
        scores = self.predict(users, candidate_item_idx)
        k = min(k, candidate_item_idx.numel())
        top_scores, top_pos = torch.topk(scores, k)
        return candidate_item_idx[top_pos], top_scores
