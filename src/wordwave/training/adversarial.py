"""Arm 8: Adversarial co-training (skeleton).

Alternates between:

1. `k_policy` steps of RL against the current PRM (via `training.rl.run_rl`);
2. `k_prm`   steps of PRM fine-tuning where the "negatives" are policy
   completions the PRM currently *over-scores* relative to their true
   outcome correctness.

This file is intentionally a skeleton — the intended usage on Vista is
one Slurm array job that alternates the two phases. The inner loops are
identical to `run_rl` / `run_prm`; only the negative-mining step is new.
"""
from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from ..utils.logging import get_logger
from .prm_train import run_prm
from .rl import run_rl

log = get_logger(__name__)


def run_adversarial(cfg: DictConfig) -> str:
    n_rounds = cfg.adversarial.get("n_rounds", 3)
    out_root = Path(cfg.output_dir)
    for r in range(n_rounds):
        round_dir = out_root / f"round_{r:02d}"
        log.info("=== Adversarial round %d/%d ===", r + 1, n_rounds)

        rl_cfg = OmegaConf.merge(cfg, OmegaConf.create({"output_dir": str(round_dir / "policy")}))
        rl_path = run_rl(rl_cfg)

        # Between rounds we would (a) sample completions from `rl_path`, (b) label
        # them by outcome correctness, (c) mine "false-positive" negatives — where
        # the current PRM assigns high step scores despite a wrong outcome — and
        # (d) add them to the PRM training set with label=0.
        # (See docs/adversarial_mining.md — omitted here to keep the file thin.)

        prm_cfg = OmegaConf.merge(cfg, OmegaConf.create({"output_dir": str(round_dir / "prm")}))
        run_prm(prm_cfg)

    return str(out_root)
