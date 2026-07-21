#!/bin/bash
# Fan out one RL job per arm config, each with an eval job chained after it.
# Uses Slurm dependencies so evals wait for training to finish.
#
# Usage:
#   bash slurm/sweep_all_arms.sh                     # all arms
#   bash slurm/sweep_all_arms.sh arm2 arm4 arm6      # specific arms

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -eq 0 ]]; then
    CONFIGS=(configs/experiments/arm*.yaml)
else
    CONFIGS=()
    for a in "$@"; do
        CONFIGS+=(configs/experiments/${a}*.yaml)
    done
fi

for cfg in "${CONFIGS[@]}"; do
    [[ -f "$cfg" ]] || { echo "skip: $cfg"; continue; }
    echo "→ submitting $cfg"
    RL_JOB=$(sbatch --parsable slurm/rl.slurm "$cfg")
    EV_JOB=$(sbatch --parsable --dependency=afterok:$RL_JOB slurm/eval.slurm "$cfg")
    echo "   RL=$RL_JOB  Eval=$EV_JOB"
done
