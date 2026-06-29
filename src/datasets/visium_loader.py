from __future__ import annotations
from pathlib import Path
from typing import Any
import squidpy as sq
import anndata as ad


class VisiumDataset:
    """Loads 10x Visium data from local files into AnnData."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.data_dir = Path(cfg["paths"]["data_raw"]) / "visium_breast_cancer"
        self.counts_file = "V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5"
        self.processed_path = Path(cfg["paths"]["data_processed"]) / "adata_raw.h5ad"

    def load(self, force_reload: bool = False) -> ad.AnnData:
        """Return cached h5ad if available, otherwise read from raw files."""
        if self.processed_path.exists() and not force_reload:
            print(f"Loading cached AnnData from {self.processed_path}")
            return ad.read_h5ad(self.processed_path)
        print(f"Reading Visium data from {self.data_dir}")
        adata = sq.read.visium(path=self.data_dir, counts_file=self.counts_file)
        adata.var_names_make_unique()
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(self.processed_path)
        print(f"Saved to {self.processed_path}")
        return adata


def load_visium(cfg: dict[str, Any], force_reload: bool = False) -> ad.AnnData:
    """Convenience wrapper around VisiumDataset.load()."""
    return VisiumDataset(cfg).load(force_reload=force_reload)
