.PHONY: help install install-dev test lint fmt data sft prm rl eval sweep clean

PYTHON ?= python
CONFIG ?= configs/experiments/arm2_naive_process.yaml

help:
	@echo "Common targets:"
	@echo "  make install         Install runtime deps in the active env"
	@echo "  make install-dev     Editable install with dev extras"
	@echo "  make test            Run the pure-Python unit tests"
	@echo "  make data            Fetch GSM8K + build golden + PRM datasets"
	@echo "  make sft             Run SFT with configs/experiments/sft.yaml"
	@echo "  make prm             Train the PRM with configs/experiments/prm.yaml"
	@echo "  make rl CONFIG=...   Run RL for a given arm YAML"
	@echo "  make eval CONFIG=... Evaluate a trained checkpoint"
	@echo "  make sweep           Submit every arm on TACC Vista (Slurm)"
	@echo "  make clean           Remove outputs/ caches/"

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	PYTHONPATH=src $(PYTHON) -m pytest tests/test_steps.py tests/test_rewards.py \
	    tests/test_traps.py tests/test_evaluation_scalars.py tests/test_registry.py

lint:
	ruff check src tests

fmt:
	ruff format src tests

data:
	$(PYTHON) -m wordwave.scripts.prepare_gsm8k --out data/gsm8k
	$(PYTHON) -m wordwave.scripts.build_golden  --n 5000 --out data/golden
	$(PYTHON) -m wordwave.scripts.build_prm_data --golden data/golden --out data/prm

sft:
	$(PYTHON) -m wordwave.scripts.train_sft --config configs/experiments/sft.yaml

prm:
	$(PYTHON) -m wordwave.scripts.train_prm --config configs/experiments/prm.yaml

rl:
	$(PYTHON) -m wordwave.scripts.train_rl --config $(CONFIG)

eval:
	$(PYTHON) -m wordwave.scripts.evaluate --config $(CONFIG)

sweep:
	bash slurm/sweep_all_arms.sh

clean:
	rm -rf outputs/ runs/ wandb/ logs/ .pytest_cache/ .ruff_cache/
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
