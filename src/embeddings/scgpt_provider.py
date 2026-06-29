from __future__ import annotations
import logging
from pathlib import Path
from typing import Any

import numpy as np
import anndata as ad

from embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class scGPTEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider using pretrained scGPT (Chen et al. 2023).

    scGPT is a foundation model pretrained on 33 million human single cells.
    It encodes each cell/spot into a biologically meaningful latent space
    by treating genes as tokens and learning gene-gene relationships via
    a transformer architecture.

    Reference:
        Chen et al. (2023). scGPT: toward building a foundation model for
        single-cell multi-omics using generative AI. Nature Methods.
        https://doi.org/10.1038/s41592-024-02201-0

    Requirements:
        - pip install scgpt
        - Pretrained weights downloaded to cfg["embeddings"]["scgpt"]["model_dir"]
        - GPU strongly recommended (CPU inference is very slow for 3000+ spots)
        - Linux recommended (Windows has torchtext binary compatibility issues)

    Weights download:
        import scgpt
        scgpt.download_pretrained_model("human", save_dir="data/raw/scgpt_weights")

    Usage:
        Set embeddings.provider: "scgpt" in configs/default.yaml
        Then run: provider = get_embedding_provider(cfg); adata = provider.embed(adata)
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        self.model_dir  = Path(cfg["embeddings"]["scgpt"]["model_dir"])
        self.gene_col   = cfg["embeddings"]["scgpt"].get("gene_col", "gene_name")
        self.batch_size = cfg["embeddings"]["scgpt"].get("batch_size", 32)
        self.device     = cfg["embeddings"]["scgpt"].get("device", "cuda")

    def embed(self, adata: ad.AnnData) -> ad.AnnData:
        """
        Generate scGPT embeddings for each spot in adata.

        Steps:
            1. Subset to highly variable genes
            2. Tokenize gene expression with scGPT gene vocabulary
            3. Run transformer encoder in batches
            4. Store embeddings in adata.obsm["X_embedding"]

        Parameters
        ----------
        adata : AnnData
            Normalized AnnData object (log1p counts, HVGs flagged).

        Returns
        -------
        AnnData
            Same object with adata.obsm["X_embedding"] populated (shape: n_spots x 512).
        """
        try:
            import scgpt
            from scgpt.tasks import embed_data
        except ImportError as e:
            raise ImportError(
                "scgpt is not installed or failed to load.\n"
                "Install with: pip install scgpt\n"
                "Note: requires Linux + GPU for full inference.\n"
                f"Original error: {e}"
            )

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"scGPT model weights not found at {self.model_dir}.\n"
                "Download with:\n"
                "  import scgpt\n"
                "  scgpt.download_pretrained_model(\'human\', "
                "save_dir=\'data/raw/scgpt_weights\')"
            )

        logger.info("Running scGPT embedding on %d spots...", adata.n_obs)
        logger.info("Model dir : %s", self.model_dir)
        logger.info("Device    : %s", self.device)
        logger.info("Batch size: %d", self.batch_size)

        # subset to HVGs for efficiency
        adata_hvg = adata[:, adata.var["highly_variable"]].copy()
        logger.info("Using %d HVGs as input tokens", adata_hvg.n_vars)

        # scGPT embed_data expects:
        #   adata with raw-ish counts in .X
        #   gene names as var_names (HGNC symbols)
        #   gene_col key in adata.var pointing to gene name column
        embeddings = embed_data(
            adata_hvg,
            model_dir=str(self.model_dir),
            gene_col="index",          # var_names are already HGNC symbols
            batch_size=self.batch_size,
            device=self.device,
            use_fast_transformer=False, # flash_attn not required
            return_new_adata=False,
        )

        # embed_data stores result in adata_hvg.obsm["X_scGPT"]
        X_scgpt = adata_hvg.obsm["X_scGPT"]
        adata.obsm["X_scGPT"]    = X_scgpt
        adata.obsm["X_embedding"] = X_scgpt.copy()

        np.save("data/embeddings/X_scgpt.npy", X_scgpt)
        logger.info("scGPT embedding shape: %s", X_scgpt.shape)
        print(f"scGPT embedding shape: {X_scgpt.shape}")

        return adata
