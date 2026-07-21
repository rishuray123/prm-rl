#!/bin/bash
#SBATCH -J pr-sft
#SBATCH -o logs/sft-%j.out
#SBATCH -e logs/sft-%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00

# Usage: sbatch slurm/sft.slurm configs/experiments/sft.yaml [override1 override2 ...]
set -euo pipefail
CONFIG="${1:?usage: sbatch slurm/sft.slurm <config.yaml> [overrides ...]}"
shift || true
source slurm/_common.sh

python -m prm_rl.scripts.train_sft --config "$CONFIG" --override "$@"
