from __future__ import annotations
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Baseline MLP — operates on node features only, ignores graph structure.

    Architecture: Linear -> LayerNorm -> ReLU -> Dropout, repeated n_layers times.
    Final layer projects to output_dim (the latent representation).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, edge_index=None, edge_attr=None) -> torch.Tensor:
        return self.net(x)
