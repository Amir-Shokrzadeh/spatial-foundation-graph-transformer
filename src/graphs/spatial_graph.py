from __future__ import annotations
from typing import Any
import numpy as np
import scipy.sparse as sp
import anndata as ad
from sklearn.neighbors import NearestNeighbors


def build_spatial_graph(adata: ad.AnnData, cfg: dict[str, Any]) -> ad.AnnData:
    """
    Build a k-nearest neighbor spatial graph from Visium spot coordinates.

    Stores results in:
        adata.obsp["spatial_connectivities"] - sparse weighted adjacency matrix
        adata.uns["spatial_graph"]           - edge_index, weights, and metadata

    Parameters
    ----------
    adata : AnnData  Embedded AnnData with obsm["spatial"] coordinates.
    cfg   : dict     Master config (uses cfg["graph"]).

    Returns
    -------
    AnnData  Same object with graph stored in obsp and uns.
    """
    coords = adata.obsm["spatial"].astype(np.float32)
    k      = cfg["graph"]["k_neighbors"]
    n_spots = coords.shape[0]

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    distances = distances[:, 1:]
    indices   = indices[:, 1:]

    src     = np.repeat(np.arange(n_spots), k)
    dst     = indices.flatten()
    weights = 1.0 / (distances.flatten() + 1e-8)

    edge_index = np.stack([src, dst], axis=0)

    adj = sp.csr_matrix((weights, (src, dst)), shape=(n_spots, n_spots))

    adata.obsp["spatial_connectivities"] = adj
    adata.uns["spatial_graph"] = {
        "edge_index"  : edge_index,
        "edge_weights": weights,
        "k_neighbors" : k,
        "method"      : cfg["graph"]["method"],
        "n_edges"     : edge_index.shape[1],
        "n_nodes"     : n_spots,
    }

    return adata
