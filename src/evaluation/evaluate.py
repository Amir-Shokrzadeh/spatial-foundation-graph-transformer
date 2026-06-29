from __future__ import annotations
from typing import Any
import json
import numpy as np
import scanpy as sc
import anndata as ad
from pathlib import Path
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def compute_silhouette(Z: np.ndarray, labels: np.ndarray,
                       sample_size: int = 2000, seed: int = 42) -> float:
    """Silhouette score on embedding Z with cluster labels."""
    return float(silhouette_score(Z, labels, sample_size=sample_size,
                                  random_state=seed))


def compute_neighborhood_preservation(Z: np.ndarray, coords: np.ndarray,
                                       k: int = 15) -> float:
    """
    Fraction of spatial k-nearest neighbors preserved in embedding space.
    1.0 = perfect preservation, 0.0 = no preservation.
    """
    knn_spatial = NearestNeighbors(n_neighbors=k).fit(coords)
    knn_embed   = NearestNeighbors(n_neighbors=k).fit(Z)
    nbrs_spatial = knn_spatial.kneighbors(coords, return_distance=False)
    nbrs_embed   = knn_embed.kneighbors(Z,      return_distance=False)
    overlap = np.mean([
        len(set(nbrs_spatial[i]) & set(nbrs_embed[i])) / k
        for i in range(len(Z))
    ])
    return float(overlap)


def cluster_leiden(adata: ad.AnnData, use_rep: str,
                   resolution: float = 0.5, seed: int = 42,
                   key_added: str = "leiden") -> ad.AnnData:
    """Run Leiden clustering on adata using the given embedding."""
    sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=15,
                    random_state=seed, key_added=f"{key_added}_neighbors")
    sc.tl.leiden(adata, resolution=resolution, random_state=seed,
                 neighbors_key=f"{key_added}_neighbors",
                 flavor="igraph", n_iterations=2, key_added=key_added)
    return adata


def evaluate(adata: ad.AnnData, cfg: dict[str, Any],
             embedding_key: str = "X_graph_transformer",
             cluster_key: str = "leiden") -> dict[str, Any]:
    """
    Run full evaluation suite on a trained embedding.

    Parameters
    ----------
    adata         : AnnData with obsm[embedding_key] and obs[cluster_key]
    cfg           : Master config dict
    embedding_key : Key in adata.obsm to evaluate
    cluster_key   : Key in adata.obs with cluster labels

    Returns
    -------
    dict with all computed metrics
    """
    eval_cfg = cfg["evaluation"]
    Z        = adata.obsm[embedding_key]
    Z_scaled = StandardScaler().fit_transform(Z)
    labels   = adata.obs[cluster_key].astype(int).values
    coords   = adata.obsm["spatial"]

    metrics = {"embedding_key": embedding_key, "cluster_key": cluster_key}

    if "silhouette" in eval_cfg["metrics"]:
        metrics["silhouette_score"] = round(
            compute_silhouette(Z_scaled, labels, seed=cfg["seed"]), 4
        )

    if "neighborhood_preservation" in eval_cfg["metrics"]:
        metrics["neighborhood_preservation"] = round(
            compute_neighborhood_preservation(
                Z_scaled, coords,
                k=eval_cfg["n_neighbors_preservation"]
            ), 4
        )

    metrics["n_clusters"] = int(adata.obs[cluster_key].nunique())
    return metrics


def save_metrics(metrics: dict[str, Any], path: str = "outputs/metrics/evaluation.json") -> None:
    """Save metrics dict to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {path}")
