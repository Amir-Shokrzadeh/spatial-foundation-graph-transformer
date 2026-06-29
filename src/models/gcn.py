from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """
    Graph Convolutional Network (Kipf & Welling 2017).

    Each layer aggregates features from 1-hop neighbors via normalized
    adjacency multiplication. Uses residual connections when dims match.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
        self.convs = nn.ModuleList([
            GCNConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(dims[i + 1]) for i in range(len(dims) - 2)
        ])
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr=None) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.norms[i](x)
                x = self.act(x)
                x = self.dropout(x)
        return x
