from __future__ import annotations
from typing import Any
import scanpy as sc
import anndata as ad


def normalize(adata: ad.AnnData, cfg: dict[str, Any]) -> ad.AnnData:
    """
    Normalize counts, log1p transform, select HVGs, and run PCA.

    Parameters
    ----------
    adata : AnnData  QC-filtered count matrix.
    cfg   : dict     Master config (uses cfg["preprocessing"]).

    Returns
    -------
    AnnData  Normalized object with HVGs flagged and X_pca in obsm.
    """
    pp = cfg["preprocessing"]

    sc.pp.normalize_total(adata, target_sum=pp["normalize_total"])
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=pp["n_hvgs"],
        flavor="seurat_v3",
    )
    sc.pp.pca(adata, n_comps=pp["n_pca_components"], use_highly_variable=True)

    return adata
