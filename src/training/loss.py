from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (SimCLR).

    Given two augmented views of the same graph, pulls together
    embeddings of the same node and pushes apart embeddings of
    different nodes within the batch.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        N  = z1.size(0)

        z  = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.T) / self.temperature

        # mask out self-similarity
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float("-inf"))

        # positive pairs: (i, i+N) and (i+N, i)
        labels = torch.cat([torch.arange(N, 2*N), torch.arange(N)]).to(z.device)
        loss   = F.cross_entropy(sim, labels)
        return loss
