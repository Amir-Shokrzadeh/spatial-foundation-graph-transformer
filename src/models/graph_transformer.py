from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv


class GraphTransformer(nn.Module):
    """
    Graph Transformer using TransformerConv (Shi et al. 2021).

    Each layer computes multi-head attention over the local neighborhood.
    Attention weights are stored in self.attention_weights after each
    forward pass for visualization.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_layers: int = 4, n_heads: int = 8,
                 dropout: float = 0.1, edge_dim: int = None) -> None:
        super().__init__()

        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            is_last = (i == n_layers - 1)
            in_ch   = input_dim if i == 0 else hidden_dim
            # last layer: concat=False so out_channels = output_dim directly
            # middle layers: concat=True so total output = out_channels * n_heads = hidden_dim
            out_ch  = output_dim if is_last else hidden_dim // n_heads
            self.convs.append(TransformerConv(
                in_channels=in_ch,
                out_channels=out_ch,
                heads=n_heads,
                dropout=dropout,
                edge_dim=edge_dim,
                concat=not is_last,
                beta=True,
            ))
            if not is_last:
                self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.attention_weights = []

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor = None) -> torch.Tensor:
        self.attention_weights = []
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, edge_index, edge_attr=edge_attr,
                           return_attention_weights=True)
            self.attention_weights.append(attn)
            if i < len(self.convs) - 1:
                x = self.norms[i](x)
                x = self.act(x)
                x = self.dropout(x)
        return x
