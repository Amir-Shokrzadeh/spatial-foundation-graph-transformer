from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from torch_geometric.data import Data
from training.augmentation import augment
from training.loss import NTXentLoss


class Trainer:
    """
    Training loop for unsupervised contrastive learning on a spatial graph.

    Supports optional edge_attr — when the graph has edge features,
    they are sliced to match the surviving edges after augmentation dropout.

    For each epoch:
        1. Apply two random augmentations (drop features, drop edges)
        2. Run both views through the model
        3. Compute NT-Xent loss between the two embedding sets
        4. Backpropagate and update weights
        5. Apply early stopping on loss plateau
    """

    def __init__(self, cfg: dict[str, Any], graph_data: Data,
                 model: nn.Module) -> None:
        self.cfg        = cfg
        self.graph_data = graph_data
        self.model      = model
        self.loss_fn    = NTXentLoss(
            temperature=cfg["training"]["contrastive"]["temperature"]
        )
        self.optimizer  = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
        )
        self.epochs    = cfg["training"]["epochs"]
        self.patience  = cfg["training"]["patience"]
        self.log_every = cfg["logging"]["log_every_n"]
        self.history   = []

    def _get_edge_attr(self, n_edges: int):
        """Slice edge_attr to match n_edges after augmentation dropout."""
        if self.graph_data.edge_attr is None:
            return None
        return self.graph_data.edge_attr[:n_edges]

    def train(self) -> list[float]:
        aug_cfg   = self.cfg["training"]["contrastive"]
        best_loss = float("inf")
        patience  = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()

            v1 = augment(self.graph_data, aug_cfg["augment_drop_feat"],
                         aug_cfg["augment_drop_edge"])
            v2 = augment(self.graph_data, aug_cfg["augment_drop_feat"],
                         aug_cfg["augment_drop_edge"])

            ea1 = self._get_edge_attr(v1.edge_index.shape[1])
            ea2 = self._get_edge_attr(v2.edge_index.shape[1])

            z1 = self.model(v1.x, v1.edge_index, edge_attr=ea1)
            z2 = self.model(v2.x, v2.edge_index, edge_attr=ea2)

            loss = self.loss_fn(z1, z2)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.history.append(loss.item())

            if epoch % self.log_every == 0:
                print(f"Epoch {epoch:>4d}/{self.epochs}  loss={loss.item():.4f}")

            if loss.item() < best_loss - 1e-4:
                best_loss = loss.item()
                patience  = 0
            else:
                patience += 1
                if patience >= self.patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        return self.history
