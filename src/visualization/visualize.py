from __future__ import annotations
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import anndata as ad
from pathlib import Path


def _savefig(fig: plt.Figure, path: str, dpi: int = 200) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")


def plot_qc_distributions(adata: ad.AnnData, cfg: dict[str, Any],
                           save_path: str = "figures/evaluation/qc_distributions.png") -> None:
    """Plot total counts, genes detected, and MT% distributions with QC thresholds."""
    qc  = cfg["qc"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].hist(adata.obs["total_counts"], bins=50, color="steelblue", edgecolor="white")
    axes[0].axvline(qc["min_counts"], color="red",    linestyle="--", label=f'min={qc["min_counts"]}')
    axes[0].axvline(qc["max_counts"], color="orange", linestyle="--", label=f'max={qc["max_counts"]}')
    axes[0].set_xlabel("Total counts"); axes[0].set_ylabel("Spots")
    axes[0].set_title("Total counts per spot"); axes[0].legend()

    axes[1].hist(adata.obs["n_genes_by_counts"], bins=50, color="seagreen", edgecolor="white")
    axes[1].axvline(qc["min_genes"], color="red", linestyle="--", label=f'min={qc["min_genes"]}')
    axes[1].set_xlabel("Genes detected"); axes[1].set_ylabel("Spots")
    axes[1].set_title("Genes detected per spot"); axes[1].legend()

    axes[2].hist(adata.obs["pct_counts_mt"], bins=50, color="salmon", edgecolor="white")
    axes[2].axvline(qc["max_pct_mt"], color="red", linestyle="--", label=f'max={qc["max_pct_mt"]}%')
    axes[2].set_xlabel("MT %"); axes[2].set_ylabel("Spots")
    axes[2].set_title("Mitochondrial %"); axes[2].legend()

    plt.suptitle("Visium Breast Cancer — QC Metrics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, save_path)
    plt.show()


def plot_tissue_overview(adata: ad.AnnData, cfg: dict[str, Any],
                          save_path: str = "figures/spatial/tissue_overview.png") -> None:
    """2x2 overview: tissue image, spatial clusters, UMAP, total counts heatmap."""
    coords = adata.obsm["spatial"]
    Z_umap = adata.obsm["X_umap_gt"]
    labels = adata.obs["leiden"].astype(int).values
    cmap   = plt.cm.tab10

    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    img_key = list(adata.uns["spatial"].keys())[0]
    img     = adata.uns["spatial"][img_key]["images"]["hires"]
    sf      = adata.uns["spatial"][img_key]["scalefactors"]["tissue_hires_scalef"]

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img, origin="upper")
    ax0.scatter(coords[:, 0] * sf, coords[:, 1] * sf, s=2, c="white", alpha=0.25)
    ax0.set_title("H&E Tissue Image + Spot Positions", fontweight="bold")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    sc1 = ax1.scatter(coords[:, 0], -coords[:, 1], c=labels, cmap=cmap, s=5, alpha=0.9)
    ax1.set_title("Spatial Map — Leiden Clusters", fontweight="bold")
    ax1.set_xlabel("Spatial X"); ax1.set_ylabel("Spatial Y")
    plt.colorbar(sc1, ax=ax1, label="Cluster")

    ax2 = fig.add_subplot(gs[1, 0])
    sc2 = ax2.scatter(Z_umap[:, 0], Z_umap[:, 1], c=labels, cmap=cmap, s=4, alpha=0.85)
    ax2.set_title("UMAP — Graph Transformer Embeddings", fontweight="bold")
    ax2.set_xlabel("UMAP 1"); ax2.set_ylabel("UMAP 2")
    plt.colorbar(sc2, ax=ax2, label="Cluster")

    ax3 = fig.add_subplot(gs[1, 1])
    sc3 = ax3.scatter(coords[:, 0], -coords[:, 1],
                      c=adata.obs["total_counts"], cmap="viridis", s=5, alpha=0.9)
    ax3.set_title("Spatial Heatmap — Total Counts per Spot", fontweight="bold")
    ax3.set_xlabel("Spatial X"); ax3.set_ylabel("Spatial Y")
    plt.colorbar(sc3, ax=ax3, label="Total counts")

    fig.suptitle("Visium Human Breast Cancer — Spatial Foundation Graph Transformer",
                 fontsize=15, fontweight="bold", y=1.01)
    _savefig(fig, save_path)
    plt.show()


def plot_umap_comparison(adata: ad.AnnData, Z_umap_pca: np.ndarray,
                          save_path: str = "figures/umap/pca_vs_graph_transformer.png") -> None:
    """Side-by-side UMAP and spatial maps for PCA vs Graph Transformer."""
    coords     = adata.obsm["spatial"]
    Z_umap_gt  = adata.obsm["X_umap_gt"]
    labels_gt  = adata.obs["leiden"].astype(int).values
    labels_pca = adata.obs["leiden_pca"].astype(int).values

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    axes[0, 0].scatter(Z_umap_pca[:, 0], Z_umap_pca[:, 1],
                       c=labels_pca, cmap="tab10", s=4, alpha=0.85)
    axes[0, 0].set_title("UMAP — PCA Baseline", fontweight="bold")
    axes[0, 0].set_xlabel("UMAP 1"); axes[0, 0].set_ylabel("UMAP 2")

    axes[0, 1].scatter(Z_umap_gt[:, 0], Z_umap_gt[:, 1],
                       c=labels_gt, cmap="tab10", s=4, alpha=0.85)
    axes[0, 1].set_title("UMAP — Graph Transformer", fontweight="bold")
    axes[0, 1].set_xlabel("UMAP 1"); axes[0, 1].set_ylabel("UMAP 2")

    axes[1, 0].scatter(coords[:, 0], -coords[:, 1],
                       c=labels_pca, cmap="tab10", s=5, alpha=0.9)
    axes[1, 0].set_title("Spatial Map — PCA Leiden Clusters", fontweight="bold")
    axes[1, 0].set_xlabel("Spatial X"); axes[1, 0].set_ylabel("Spatial Y")

    axes[1, 1].scatter(coords[:, 0], -coords[:, 1],
                       c=labels_gt, cmap="tab10", s=5, alpha=0.9)
    axes[1, 1].set_title("Spatial Map — Graph Transformer Leiden Clusters", fontweight="bold")
    axes[1, 1].set_xlabel("Spatial X"); axes[1, 1].set_ylabel("Spatial Y")

    plt.suptitle("PCA Baseline vs Graph Transformer — Embedding Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, save_path)
    plt.show()


def plot_spatial_attention(coords: np.ndarray, attn_score: np.ndarray,
                            attn_std: np.ndarray,
                            save_path: str = "figures/attention/spatial_attention_edge.png") -> None:
    """Plot mean attention weight and selectivity (std) across the tissue."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sc0 = axes[0].scatter(coords[:, 0], -coords[:, 1], c=attn_score, cmap="hot",
                          s=5, alpha=0.9,
                          vmin=np.percentile(attn_score, 5),
                          vmax=np.percentile(attn_score, 95))
    axes[0].set_title("Mean Incoming Attention Weight per Spot", fontweight="bold")
    axes[0].set_xlabel("Spatial X"); axes[0].set_ylabel("Spatial Y")
    plt.colorbar(sc0, ax=axes[0], label="Mean attention weight")

    sc1 = axes[1].scatter(coords[:, 0], -coords[:, 1], c=attn_std, cmap="plasma",
                          s=5, alpha=0.9,
                          vmin=np.percentile(attn_std, 5),
                          vmax=np.percentile(attn_std, 95))
    axes[1].set_title("Attention Selectivity (Std across neighbors)", fontweight="bold")
    axes[1].set_xlabel("Spatial X"); axes[1].set_ylabel("Spatial Y")
    plt.colorbar(sc1, ax=axes[1], label="Attention std")

    plt.suptitle("Graph Transformer + Edge Features — Spatial Attention (Last Layer)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, save_path)
    plt.show()
