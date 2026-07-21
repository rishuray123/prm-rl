#!/bin/bash
#SBATCH -J ww-rl
#SBATCH -o logs/rl-%j.out
#SBATCH -e logs/rl-%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00

# Usage: sbatch slurm/rl.slurm configs/experiments/arm2_naive_process.yaml [overrides ...]
set -euo pipefail
CONFIG="${1:?usage: sbatch slurm/rl.slurm <config.yaml> [overrides ...]}"
shift || true
source slurm/_common.sh

python -m wordwave.scripts.train_rl --config "$CONFIG" --override "$@"
