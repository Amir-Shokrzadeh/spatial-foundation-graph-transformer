"""
src/utils/logging.py
====================
Configure Python's standard logging with file + console handlers.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any


def setup_logging(cfg: dict[str, Any]) -> None:
    """
    Initialise root logger from the ``logging`` section of the config dict.

    Handlers:
        - StreamHandler  → stdout (respects log level)
        - FileHandler    → outputs/logs/run.log
    """
    log_cfg = cfg.get("logging", {})
    level_name: str = log_cfg.get("level", "INFO").upper()
    level: int = getattr(logging, level_name, logging.INFO)

    log_dir = Path(cfg.get("paths", {}).get("logs", "outputs/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="a", encoding="utf-8"),
    ]

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
    )

    # Suppress noisy third-party loggers
    for noisy in ("PIL", "matplotlib", "numba", "anndata", "scanpy"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
