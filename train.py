"""
train.py – Top-level training entry point
=========================================
Usage:
    python train.py                          # uses configs/default.yaml
    python train.py --config configs/my.yaml
    python train.py --config configs/default.yaml --embedding_provider scgpt
    python train.py --config configs/default.yaml --model graph_transformer

Each flag overrides the corresponding key in the YAML config.
"""
import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Allow running from project root without `pip install -e .`
# (editable install is the preferred approach; this is a fallback)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.config import load_config, override_config  # noqa: E402
from utils.seed import set_global_seed               # noqa: E402
from utils.logging import setup_logging              # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spatial Foundation Graph Transformer – training pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--embedding_provider",
        type=str,
        default=None,
        choices=["pca", "scgpt", "geneformer", "uce"],
        help="Override embeddings.provider from config.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["mlp", "gcn", "graphsage", "graph_transformer"],
        help="Override model architecture.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs from config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override global random seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load & patch config ──────────────────────────────────────────────
    cfg = load_config(args.config)

    overrides = {}
    if args.embedding_provider is not None:
        overrides["embeddings.provider"] = args.embedding_provider
    if args.model is not None:
        overrides["training.model"] = args.model
    if args.epochs is not None:
        overrides["training.epochs"] = args.epochs
    if args.seed is not None:
        overrides["seed"] = args.seed

    if overrides:
        cfg = override_config(cfg, overrides)

    # ── Reproducibility ──────────────────────────────────────────────────
    set_global_seed(cfg["seed"])

    # ── Logging ──────────────────────────────────────────────────────────
    setup_logging(cfg)
    logger = logging.getLogger(__name__)
    logger.info("Configuration loaded from: %s", args.config)
    logger.info("Embedding provider : %s", cfg["embeddings"]["provider"])
    logger.info("Random seed        : %d", cfg["seed"])

    # ── Pipeline stages (imported lazily so each is independently testable)
    logger.info("=" * 60)
    logger.info("Stage 1 / 6 : Loading dataset")
    logger.info("=" * 60)
    from datasets.visium_loader import VisiumDataset  # noqa: E402
    dataset = VisiumDataset(cfg)
    adata = dataset.load()

    logger.info("=" * 60)
    logger.info("Stage 2 / 6 : Quality control")
    logger.info("=" * 60)
    from preprocessing.quality_control import run_qc  # noqa: E402
    adata = run_qc(adata, cfg)

    logger.info("=" * 60)
    logger.info("Stage 3 / 6 : Normalisation & HVG selection")
    logger.info("=" * 60)
    from preprocessing.normalization import normalize  # noqa: E402
    adata = normalize(adata, cfg)

    logger.info("=" * 60)
    logger.info("Stage 4 / 6 : Generating node embeddings")
    logger.info("=" * 60)
    from embeddings.factory import get_embedding_provider  # noqa: E402
    provider = get_embedding_provider(cfg)
    adata = provider.embed(adata)

    logger.info("=" * 60)
    logger.info("Stage 5 / 6 : Constructing spatial graph")
    logger.info("=" * 60)
    from graphs.spatial_graph import build_spatial_graph  # noqa: E402
    graph_data = build_spatial_graph(adata, cfg)

    logger.info("=" * 60)
    logger.info("Stage 6 / 6 : Training model")
    logger.info("=" * 60)
    from training.trainer import Trainer  # noqa: E402
    trainer = Trainer(cfg, graph_data)
    trainer.train()

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
