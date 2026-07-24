#!/bin/bash
# End-to-end SMOKE TEST for Arm 1 on a single Vista H200 gh-dev node.
#
# What this exercises:
#   1. Env loads, torch sees the H200.
#   2. Qwen2.5-1.5B-Instruct downloads/loads under the HF cache.
#   3. GSM8K downloads/loads (n=64).
#   4. trl.GRPOTrainer runs 10 steps with the outcome reward.
#   5. Trained policy is saved to $OUT.
#   6. Evaluation (n_test=20, no PRM) computes accuracy + CRHS and writes
#      $OUT/eval_results.json.
#
# On success this proves the pipeline is wired end-to-end on Vista and
# gives us a baseline numeric result for a 1.5B model. Scaling to a full
# run is then a matter of swapping the config or overriding hyperparams.
#
# Submit from repo root:
#     sbatch slurm/smoke_arm1.slurm
#
#SBATCH -J arm1-smoke
#SBATCH -o logs/arm1-smoke-%j.out
#SBATCH -e logs/arm1-smoke-%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00

set -euo pipefail

CONFIG="configs/experiments/arm1_smoke.yaml"

source slurm/_common.sh

echo
echo "=========================================="
echo "STAGE 1: RL training (Arm 1, 10 GRPO steps)"
echo "=========================================="
time python -m prm_rl.scripts.train_rl --config "$CONFIG"

echo
echo "=========================================="
echo "STAGE 2: Evaluation on 20 GSM8K test items"
echo "=========================================="
time python -m prm_rl.scripts.evaluate --config "$CONFIG"

RESULTS="outputs/arm1_smoke/eval_results.json"
echo
echo "=========================================="
echo "RESULTS ($RESULTS)"
echo "=========================================="
if [[ -f "$RESULTS" ]]; then
    cat "$RESULTS"
else
    echo "MISSING: $RESULTS was not produced." >&2
    exit 1
fi

echo
echo "SMOKE TEST PASSED."
