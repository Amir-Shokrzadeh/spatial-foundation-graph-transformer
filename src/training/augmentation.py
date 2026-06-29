from __future__ import annotations
import torch
from torch_geometric.data import Data


def augment(data: Data, drop_feat_prob: float = 0.1,
            drop_edge_prob: float = 0.1) -> Data:
    """
    Apply stochastic augmentations to a PyG Data object.

    Augmentations:
        - Feature masking: randomly zero out node features with probability drop_feat_prob
        - Edge dropout:    randomly remove edges with probability drop_edge_prob

    Returns a new Data object; the original is not modified.
    """
    x          = data.x.clone()
    edge_index = data.edge_index.clone()

    # feature masking
    if drop_feat_prob > 0:
        mask = torch.bernoulli(
            torch.full(x.shape, 1 - drop_feat_prob)
        ).bool()
        x = x * mask

    # edge dropout
    if drop_edge_prob > 0:
        n_edges  = edge_index.size(1)
        keep     = torch.bernoulli(
            torch.full((n_edges,), 1 - drop_edge_prob)
        ).bool()
        edge_index = edge_index[:, keep]

    return Data(x=x, edge_index=edge_index)
