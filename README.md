# wordwave / prm_rl

Experimental framework for investigating **reward hacking in process-based RL** for
math reasoning, built on top of Hugging Face `transformers`, `datasets`, and `trl`.

The goal is to keep the framework *thin*: every experimental "arm" from the research
plan reduces to a different combination of reward functions plugged into
`trl.GRPOTrainer`, and everything else (SFT, PRM training, evaluation) uses stock TRL /
HF `Trainer` APIs.

## What lives where

```
prm_rl/
├── notebooks/quickstart_colab.ipynb    # Runnable on a free Colab GPU, Arms 1 & 2
├── configs/
│   ├── models/*.yaml                   # Model presets (tiny → 7B)
│   ├── training/{sft,prm,rl_base}.yaml # Trainer hyperparams
│   └── experiments/arm{1..7}.yaml      # One YAML per experimental arm
├── src/wordwave/
│   ├── data/     # GSM8K, golden dataset, PRM dataset builders
│   ├── models/   # Thin wrappers around AutoModel*, PRM loader, NLI loader
│   ├── rewards/  # One file per reward function (Arms 1–6) + registry
│   ├── training/ # sft.py / prm_train.py / rl.py — each is <100 LOC of TRL glue
│   ├── evaluation/  # accuracy, exploit, verbosity, CCT, CMA, EST, CRHS
│   └── utils/    # step splitters, sandbox helpers, logging
│   └── scripts/  # CLI entry points (`python -m wordwave.scripts.*`)
├── slurm/        # TACC Vista Slurm scripts for the `gh` and `gh-dev` queues
├── data/traps/   # Trap scenarios for measuring specification gaming
└── tests/        # pytest smoke tests for verifier + rewards + step parser
```

## Mapping from research plan → code

| Plan section                          | Code                                                      |
| ------------------------------------- | --------------------------------------------------------- |
| Stage 1 SFT                           | `scripts/train_sft.py`  → `trl.SFTTrainer`                |
| PRM training on golden dataset        | `scripts/train_prm.py`  → `trl.RewardTrainer`             |
| Stage 2 RL                            | `scripts/train_rl.py`   → `trl.GRPOTrainer` (+ PPO opt.)  |
| Arm 1  Outcome-based reward           | `rewards/outcome.py`                                      |
| Arm 2  Naive process reward           | `rewards/process.py`                                      |
| Arm 3  Prefix-consistency reward      | `rewards/prefix.py`                                       |
| Arm 4  Contradiction-aware reward     | `rewards/contradiction.py` (uses an NLI model)            |
| Arm 5  Counterfactual reward          | `rewards/counterfactual.py`                               |
| Arm 6  Hybrid process+outcome         | `rewards/hybrid.py` (PROGRS-style outcome-cond. centering)|
| Arm 7  Regularized (KL to SFT)        | GRPO `beta` in `configs/experiments/arm7_*.yaml`          |
| Arm 8  Adversarial co-training        | `training/adversarial.py` (loop that alternates PRM+policy)|
| Arm 9  VinePPO                        | `training/vineppo.py` (Monte-Carlo value estimates)       |
| Trap scenarios                        | `data/traps/*.json` + `evaluation/traps.py`               |
| Faithfulness (CCT / Phi-CCT)          | `evaluation/faithfulness.py`                              |
| Causal Mediation Analysis             | `evaluation/cma.py`                                       |
| Evaluator Stress Test                 | `evaluation/est.py`                                       |
| Composite Reward-Hacking Score        | `evaluation/crhs.py`                                      |

## Quick start (local / Colab)

```bash
pip install -e ".[dev]"

# 1. Prepare GSM8K + a tiny golden subset
python -m wordwave.scripts.prepare_gsm8k --out data/gsm8k
python -m wordwave.scripts.build_golden  --split train --n 200 --out data/golden

# 2. SFT a tiny model
python -m wordwave.scripts.train_sft --config configs/experiments/arm1_outcome.yaml

# 3. RL with an arm
python -m wordwave.scripts.train_rl  --config configs/experiments/arm2_naive_process.yaml

# 4. Evaluate
python -m wordwave.scripts.evaluate  --config configs/experiments/arm2_naive_process.yaml
```

## Quick start (TACC Vista)

```bash
# One-time setup on a login node (creates a $SCRATCH venv, installs deps)
bash slurm/setup_env.sh

# Submit jobs (uses your $ALLOCATION env var)
sbatch slurm/sft.slurm  configs/experiments/arm1_outcome.yaml
sbatch slurm/prm.slurm  configs/training/prm.yaml
sbatch slurm/rl.slurm   configs/experiments/arm2_naive_process.yaml
sbatch slurm/eval.slurm configs/experiments/arm2_naive_process.yaml
```

See `slurm/README.md` for interactive dev (`idev -p gh-dev`) and multi-arm sweeps.

## Design notes

* We treat GRPO (DeepSeek-style) as the default RL algorithm because it accepts
  a *list of Python reward callables with weights* natively — which is the
  cleanest possible mapping onto our "different rewards per arm" experimental
  design. PPO is available for Arm 9 / VinePPO variants that need a value
  network.
* KL regularization (Arm 7) is *already* a first-class feature of GRPO/PPO —
  we just expose `beta` in the arm's YAML and don't wrap the reward.
* Step-level rewards are aggregated to a single scalar per completion at the
  reward function boundary. We keep the aggregator explicit
  (`sum`, `sum_until_first_error`, `mean`, `outcome_conditioned_mean`, …)
  so the arms are auditable.
* All evaluations run in `evaluation/sandbox.py` — a subprocess with restricted
  env vars and a temp cwd — to prevent the policy from tampering with reward /
  eval files during study of reward hacking.
