from __future__ import annotations

import logging
import os
import sys


def setup_logging(level: str | int = "INFO") -> None:
    fmt = "%(asctime)s | %(levelname).1s | %(name)s | %(message)s"
    logging.basicConfig(
        level=level if not isinstance(level, str) else level.upper(),
        format=fmt,
        stream=sys.stdout,
        force=True,
    )
    for noisy in ("urllib3", "filelock", "datasets", "transformers"):
        logging.getLogger(noisy).setLevel(os.environ.get("PRMRL_HF_LOGLEVEL", "WARNING"))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
