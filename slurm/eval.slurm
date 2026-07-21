#!/bin/bash
#SBATCH -J pr-eval
#SBATCH -o logs/eval-%j.out
#SBATCH -e logs/eval-%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00

# Usage: sbatch slurm/eval.slurm configs/experiments/arm2_naive_process.yaml [overrides ...]
set -euo pipefail
CONFIG="${1:?usage: sbatch slurm/eval.slurm <config.yaml> [overrides ...]}"
shift || true
source slurm/_common.sh

python -m prm_rl.scripts.evaluate --config "$CONFIG" --override "$@"
