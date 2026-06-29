from __future__ import annotations
from typing import Any
import anndata as ad
from embeddings.base import EmbeddingProvider
from embeddings.pca_provider import PCAEmbeddingProvider


def get_embedding_provider(cfg: dict[str, Any]) -> EmbeddingProvider:
    """
    Return the correct EmbeddingProvider based on cfg["embeddings"]["provider"].

    Supported providers:
        "pca"      - PCA on precomputed X_pca (default, no extra dependencies)
        "scgpt"    - scGPT foundation model (requires GPU + Linux recommended)
        "geneformer" - Geneformer foundation model (not yet implemented)
        "uce"      - UCE foundation model (not yet implemented)
    """
    provider = cfg["embeddings"]["provider"]

    if provider == "pca":
        return PCAEmbeddingProvider(cfg)

    if provider == "scgpt":
        from embeddings.scgpt_provider import scGPTEmbeddingProvider
        return scGPTEmbeddingProvider(cfg)

    if provider in ("geneformer", "uce"):
        raise NotImplementedError(
            f"Embedding provider '{provider}' is not yet implemented.\n"
            f"To add it: create src/embeddings/{provider}_provider.py "
            f"subclassing EmbeddingProvider, then register it here."
        )

    raise ValueError(
        f"Unknown embedding provider: '{provider}'.\n"
        f"Supported: 'pca', 'scgpt', 'geneformer', 'uce'."
    )
