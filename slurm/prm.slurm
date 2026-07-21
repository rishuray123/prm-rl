#!/bin/bash
#SBATCH -J ww-prm
#SBATCH -o logs/prm-%j.out
#SBATCH -e logs/prm-%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00

# Usage: sbatch slurm/prm.slurm configs/experiments/prm.yaml [overrides ...]
set -euo pipefail
CONFIG="${1:?usage: sbatch slurm/prm.slurm <config.yaml> [overrides ...]}"
shift || true
source slurm/_common.sh

python -m wordwave.scripts.train_prm --config "$CONFIG" --override "$@"
