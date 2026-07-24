"""Expand a golden dataset into step-level PRM training rows.

With `--inject_negatives_prob > 0` the builder also emits synthetic
negatives per step (arithmetic mutations, operator swaps, fabricated
conclusions, duplicated previous steps) so the PRM sees real gradient
signal. See docs/knowledge-base.md §6.1 for the motivation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..data.golden import load_golden
from ..data.prm_data import DEFAULT_NEGATIVE_KINDS, build_prm_dataset, summarize_prm_dataset
from ..utils.logging import get_logger, setup_logging

log = get_logger(__name__)


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--golden", required=True, help="Path to a saved golden dataset")
    p.add_argument("--out", required=True, help="Output directory for the PRM dataset")
    p.add_argument(
        "--inject_negatives_prob",
        type=float,
        default=0.0,
        help="Per-step probability of also emitting a synthetic-negative row.",
    )
    p.add_argument(
        "--negative_kinds",
        nargs="*",
        default=list(DEFAULT_NEGATIVE_KINDS),
        help="Which synthetic-negative kinds to sample from.",
    )
    p.add_argument(
        "--max_negatives_per_example",
        type=int,
        default=2,
        help="Cap on synthetic negatives added per golden example.",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    golden = load_golden(args.golden)
    prm_ds = build_prm_dataset(
        golden,
        inject_negatives_prob=args.inject_negatives_prob,
        negative_kinds=args.negative_kinds,
        max_negatives_per_example=args.max_negatives_per_example,
        seed=args.seed,
    )
    stats = summarize_prm_dataset(prm_ds)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prm_ds.save_to_disk(str(out))
    log.info(
        "Wrote %d PRM rows to %s (positives=%d, negatives=%d, pos_frac=%.3f)",
        stats["n"],
        out,
        stats["n_pos"],
        stats["n_neg"],
        stats["pos_frac"],
    )


if __name__ == "__main__":
    main()
