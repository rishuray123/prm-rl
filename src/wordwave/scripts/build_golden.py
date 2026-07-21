"""Build a golden reasoning dataset from GSM8K.

For the quickstart we use the `gsm8k_native` strategy — the dataset's
built-in solutions are already high quality step-by-step derivations.
Swap to `teacher` / `verifier` when running on Vista with a strong
teacher available.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_from_disk

from ..data.golden import build_golden_dataset
from ..data.gsm8k import load_gsm8k
from ..utils.logging import get_logger, setup_logging

log = get_logger(__name__)


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--strategy", default="gsm8k_native", choices=["gsm8k_native", "teacher", "verifier"])
    p.add_argument("--from_disk", default=None, help="Load projected GSM8K from a saved path")
    args = p.parse_args()

    if args.from_disk:
        base = load_from_disk(args.from_disk)[args.split]
    else:
        base = load_gsm8k(split=args.split, n=args.n)

    golden = build_golden_dataset(base, strategy=args.strategy, max_examples=args.n)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    golden.save_to_disk(str(out))
    log.info("Wrote %d golden examples to %s", len(golden), out)


if __name__ == "__main__":
    main()
