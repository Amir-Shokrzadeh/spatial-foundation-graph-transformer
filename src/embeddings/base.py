from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import anndata as ad


class EmbeddingProvider(ABC):
    """
    Abstract base class for all embedding providers.
    Subclasses implement embed() to populate adata.obsm["X_embedding"].
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    @abstractmethod
    def embed(self, adata: ad.AnnData) -> ad.AnnData:
        """
        Compute embeddings and store in adata.obsm["X_embedding"].

        Parameters
        ----------
        adata : AnnData  Normalized AnnData object.

        Returns
        -------
        AnnData  Same object with adata.obsm["X_embedding"] populated.
        """
        ...
