from __future__ import annotations
from typing import Any
import torch.nn as nn
from models.mlp import MLP
from models.gcn import GCN
from models.graphsage import GraphSAGE
from models.graph_transformer import GraphTransformer


def get_model(cfg: dict[str, Any]) -> nn.Module:
    """
    Instantiate the model specified in cfg["training"]["model"].

    Supported keys: "mlp", "gcn", "graphsage", "graph_transformer"
    """
    key        = cfg["training"].get("model", "graph_transformer")
    input_dim  = cfg["models"]["input_dim"]
    hidden_dim = cfg["models"]["hidden_dim"]
    output_dim = cfg["models"]["output_dim"]
    dropout    = cfg["models"]["dropout"]

    if key == "mlp":
        return MLP(input_dim, hidden_dim, output_dim,
                   n_layers=cfg["models"]["mlp"]["n_layers"],
                   dropout=dropout)

    if key == "gcn":
        return GCN(input_dim, hidden_dim, output_dim,
                   n_layers=cfg["models"]["gcn"]["n_layers"],
                   dropout=dropout)

    if key == "graphsage":
        return GraphSAGE(input_dim, hidden_dim, output_dim,
                         n_layers=cfg["models"]["graphsage"]["n_layers"],
                         dropout=dropout,
                         aggr=cfg["models"]["graphsage"]["aggr"])

    if key == "graph_transformer":
        return GraphTransformer(input_dim, hidden_dim, output_dim,
                                n_layers=cfg["models"]["graph_transformer"]["n_layers"],
                                n_heads=cfg["models"]["graph_transformer"]["n_heads"],
                                dropout=dropout)

    raise NotImplementedError(f"Model '{key}' is not registered in the factory.")
