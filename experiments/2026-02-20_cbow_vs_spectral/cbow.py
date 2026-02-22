"""CBOW model with negative sampling.

Standard dense embedding baseline for comparison with spectral model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Separate input/output embeddings (like Word2Vec)
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

        # Word2Vec-style init
        initrange = 0.5 / embedding_dim
        self.in_embed.weight.data.uniform_(-initrange, initrange)
        self.out_embed.weight.data.zero_()

    def forward(self, context_ids, target_ids, negative_ids):
        """
        context_ids:  (B, 2*W) — context word indices
        target_ids:   (B,)     — target word index
        negative_ids: (B, K)   — negative sample indices

        Returns: scalar loss
        """
        # Context vector = mean of context embeddings
        ctx = self.in_embed(context_ids).mean(dim=1)  # (B, D)

        # Positive: dot product with target
        pos = self.out_embed(target_ids)  # (B, D)
        pos_score = (ctx * pos).sum(dim=1)  # (B,)

        # Negative: dot products with negative samples
        neg = self.out_embed(negative_ids)  # (B, K, D)
        neg_score = torch.bmm(neg, ctx.unsqueeze(2)).squeeze(2)  # (B, K)

        # Negative sampling loss (sum over negatives, mean over batch)
        pos_loss = -F.logsigmoid(pos_score)           # (B,)
        neg_loss = -F.logsigmoid(-neg_score).sum(1)    # (B,)

        return (pos_loss + neg_loss).mean()

    def get_embeddings(self):
        """Return input embeddings as numpy array (V, D)."""
        return self.in_embed.weight.detach().cpu().numpy()
