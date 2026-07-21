"""Expand a golden dataset into step-level PRM training rows."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..data.golden import load_golden
from ..data.prm_data import build_prm_dataset
from ..utils.logging import get_logger, setup_logging

log = get_logger(__name__)


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--golden", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    golden = load_golden(args.golden)
    prm_ds = build_prm_dataset(golden)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prm_ds.save_to_disk(str(out))
    log.info("Wrote %d PRM rows to %s", len(prm_ds), out)


if __name__ == "__main__":
    main()
