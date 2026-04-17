# losses/nt_xent.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
    From: "A Simple Framework for Contrastive Learning" (Chen et al., 2020).

    Given a batch of N images, SimCLR produces 2N embeddings (two views each).
    For each embedding, its positive pair is the other view of the same image.
    All other 2N-2 embeddings in the batch are negatives.

    Args:
        temperature : τ — controls sharpness of the distribution.
                      Lower τ = harder negatives, more confident predictions.
                      Chen et al. use τ = 0.07 for ImageNet, 0.5 is safer to start.
        device      : 'cuda' or 'cpu'
    """

    def __init__(self, temperature: float = 0.5, device: str = "cuda"):
        super().__init__()
        self.temperature = temperature
        self.device      = device

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i : Projection embeddings from view 1. Shape: [N, D]
            z_j : Projection embeddings from view 2. Shape: [N, D]

        Returns:
            Scalar loss value.
        """
        N = z_i.shape[0]  # batch size

        # ── Step 1: L2-normalize embeddings onto the unit hypersphere ──────────
        # This ensures similarity is measured by cosine distance, not magnitude.
        z_i = F.normalize(z_i, dim=1)  # [N, D]
        z_j = F.normalize(z_j, dim=1)  # [N, D]

        # ── Step 2: Concatenate into one big 2N embedding matrix ──────────────
        # Layout: [z_i_0, z_i_1, ..., z_i_N, z_j_0, z_j_1, ..., z_j_N]
        z = torch.cat([z_i, z_j], dim=0)  # [2N, D]

        # ── Step 3: Full (2N x 2N) cosine similarity matrix ───────────────────
        # sim[a, b] = cosine_similarity(z[a], z[b]) / τ
        sim = torch.mm(z, z.T) / self.temperature  # [2N, 2N]

        # ── Step 4: Build positive pair indices ───────────────────────────────
        # For index i in [0, N):    positive is at index i + N  (its j-view)
        # For index i in [N, 2N):   positive is at index i - N  (its i-view)
        pos_idx = torch.cat([
            torch.arange(N, 2 * N),   # i-views → j-views
            torch.arange(0, N),        # j-views → i-views
        ]).to(self.device)             # [2N]

        # ── Step 5: Mask out self-similarity (diagonal) ───────────────────────
        # An embedding must not be its own positive/negative.
        mask = torch.eye(2 * N, dtype=torch.bool).to(self.device)
        sim = sim.masked_fill(mask, float("-inf"))  # [2N, 2N]

        # ── Step 6: Cross-entropy over the 2N classification problems ─────────
        # Each row is a classification: "which of the 2N-1 others is my positive?"
        # F.cross_entropy expects logits [2N, 2N] and targets [2N]
        loss = F.cross_entropy(sim, pos_idx)

        return loss