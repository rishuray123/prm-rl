"""Tiny config helpers.

We keep configs as plain YAML loaded through OmegaConf so any script can be
called with `--config path/to.yaml [--override key=value ...]`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    cfg = OmegaConf.load(str(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    if not isinstance(cfg, DictConfig):  # pragma: no cover
        raise TypeError(f"Expected mapping at {path}, got {type(cfg).__name__}")
    return cfg


def parse_cli() -> tuple[DictConfig, argparse.Namespace]:
    """Standard CLI parser used by every `scripts/*.py` entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="dotlist overrides, e.g. training.learning_rate=1e-5",
    )
    p.add_argument("--output_dir", default=None, help="Override cfg.output_dir")
    args = p.parse_args()
    cfg = load_config(args.config, overrides=args.override)
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir  # type: ignore[assignment]
    return cfg, args


def to_container(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
