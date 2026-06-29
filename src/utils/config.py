"""
src/utils/config.py
===================
Load, merge, and override YAML configuration files.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and return it as a nested dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def override_config(
    cfg: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """
    Apply dot-path overrides to a nested config dict.

    Example:
        override_config(cfg, {"embeddings.provider": "scgpt", "seed": 0})
    """
    cfg = copy.deepcopy(cfg)
    for dot_path, value in overrides.items():
        keys = dot_path.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    """Serialise a config dict back to YAML (useful for run logging)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)
