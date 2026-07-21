"""Fetch GSM8K and dump to disk as an Arrow DatasetDict."""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import DatasetDict

from ..data.gsm8k import load_gsm8k
from ..utils.logging import get_logger, setup_logging

log = get_logger(__name__)


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--subset", default="main")
    p.add_argument("--n_train", type=int, default=None)
    p.add_argument("--n_test", type=int, default=None)
    args = p.parse_args()

    train = load_gsm8k(split="train", subset=args.subset, n=args.n_train)
    test = load_gsm8k(split="test", subset=args.subset, n=args.n_test)
    log.info("Loaded gsm8k: %d train / %d test", len(train), len(test))
    dsd = DatasetDict({"train": train, "test": test})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dsd.save_to_disk(str(out))
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
