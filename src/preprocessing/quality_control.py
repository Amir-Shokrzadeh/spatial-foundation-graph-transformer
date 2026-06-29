from __future__ import annotations
from  typing import Any
import scanpy as sc 
import anndata as ad

def run_qc(adata: ad.AnnData, cfg: dict[str, Any]) -> ad.AnnData:
    """
    Flag mitochondrial genes, compute QC metrics, filter spots and genes.

    Parameters
    ----------
    adata : AnnData  Raw count matrix.
    cfg   : dict     Master config (uses cfg["qc"] thresholds).

    Returns
    -------
    AnnData  Filtered copy with QC metrics stored in .obs.
    """
    qc = cfg["qc"]

    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, log1p=False)

    mask = (
        (adata.obs["total_counts"]      >= qc["min_counts"]) &
        (adata.obs["total_counts"]      <= qc["max_counts"]) &
        (adata.obs["n_genes_by_counts"] >= qc["min_genes"]) &
        (adata.obs["pct_counts_mt"]     <= qc["max_pct_mt"])
    )
    adata = adata[mask].copy()
    sc.pp.filter_genes(adata, min_cells=qc["min_cells"])

    return adata
