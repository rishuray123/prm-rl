from __future__ import annotations

from ..config import parse_cli
from ..training.rl import run_rl
from ..utils.logging import setup_logging


def main() -> None:
    setup_logging()
    cfg, _ = parse_cli()
    run_rl(cfg)


if __name__ == "__main__":
    main()
