from __future__ import annotations
from typing import Any
import numpy as np
import anndata as ad
from embeddings.base import EmbeddingProvider


class PCAEmbeddingProvider(EmbeddingProvider):
    """Uses the precomputed X_pca as node embeddings."""

    def embed(self, adata: ad.AnnData) -> ad.AnnData:
        if "X_pca" not in adata.obsm:
            raise ValueError("X_pca not found in adata.obsm. Run normalization first.")
        adata.obsm["X_embedding"] = adata.obsm["X_pca"].copy()
        np.save("data/embeddings/X_pca.npy", adata.obsm["X_embedding"])
        print(f"PCA embedding shape: {adata.obsm['X_embedding'].shape}")
        return adata
