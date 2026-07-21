from __future__ import annotations

from ..config import parse_cli
from ..training.sft import run_sft
from ..utils.logging import setup_logging


def main() -> None:
    setup_logging()
    cfg, _ = parse_cli()
    run_sft(cfg)


if __name__ == "__main__":
    main()
