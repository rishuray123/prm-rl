from __future__ import annotations

from ..config import parse_cli
from ..training.prm_train import run_prm
from ..utils.logging import setup_logging


def main() -> None:
    setup_logging()
    cfg, _ = parse_cli()
    run_prm(cfg)


if __name__ == "__main__":
    main()
