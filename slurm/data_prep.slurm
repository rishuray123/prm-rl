#!/bin/bash
#SBATCH -J ww-data
#SBATCH -o logs/data-%j.out
#SBATCH -e logs/data-%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00

set -euo pipefail
source slurm/_common.sh

python -m wordwave.scripts.prepare_gsm8k --out data/gsm8k
python -m wordwave.scripts.build_golden  --n 5000 --out data/golden
python -m wordwave.scripts.build_prm_data --golden data/golden --out data/prm

echo "Data prep complete."
