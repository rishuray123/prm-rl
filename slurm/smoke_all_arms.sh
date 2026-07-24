#!/bin/bash
# End-to-end SMOKE test of all six arms + PRM on a single GH200 node.
#
# Usage (from repo root, on a compute node with venv already activated
# and Vista modules loaded — see docs/knowledge-base.md §2.2):
#
#     bash slurm/smoke_all_arms.sh
#
# Or, non-interactively via sbatch:
#
#     sbatch -p gh -N 1 -n 1 -t 01:30:00 --wrap="bash slurm/smoke_all_arms.sh"
#
# What it does (sequential, ~35-40 min total on H200):
#   1. Build a tiny 64-example golden GSM8K subset.
#   2. Train a tiny PRM on it (~1 min, DeBERTa-v3-xsmall).
#   3. Run RL smoke (10 GRPO steps, Qwen2.5-1.5B) for arms 1..6 in
#      sequence, each writing to outputs/arm{i}_smoke/.
#   4. Evaluate each arm on 20 GSM8K test items + 5 trap prompts.
#   5. Aggregate every eval_results.json into a markdown table at
#      outputs/smoke_summary.md.
#
# ⚠️  With strategy='gsm8k_native' the PRM sees only positive steps,
# so its scores collapse to ~1.0 for everything. This is FINE for a
# plumbing smoke test but means Arms 2/3/4/6 process rewards won't
# meaningfully differentiate completions — cross-arm numbers below are
# for pipeline validation only, not scientific comparison.
# See docs/knowledge-base.md §6.4.

set -euo pipefail

if ! command -v python >/dev/null; then
    echo "ERROR: no python on PATH. Did you activate the venv?" >&2
    echo "       See docs/knowledge-base.md §2.2 for the correct order." >&2
    exit 1
fi
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "WARN: VIRTUAL_ENV is not set. If pip/python target the system Python," >&2
    echo "      abort now — this is the trap documented in KB §2.2." >&2
fi

mkdir -p logs data outputs

echo "======================================================================"
echo "STAGE 1/4 — build golden dataset (n=64, strategy=gsm8k_native)"
echo "======================================================================"
time python -m prm_rl.scripts.build_golden \
    --split train --n 64 --out data/golden_smoke --strategy gsm8k_native \
    2>&1 | tee logs/smoke-golden.log

echo
echo "======================================================================"
echo "STAGE 2/4 — train tiny PRM on the golden data"
echo "======================================================================"
time python -m prm_rl.scripts.train_prm \
    --config configs/experiments/prm_smoke.yaml \
    2>&1 | tee logs/smoke-prm.log

echo
echo "======================================================================"
echo "STAGE 3/4 — RL smoke (10 steps) + eval for each arm"
echo "======================================================================"
ARMS=(arm1_smoke arm2_smoke arm3_smoke arm4_smoke arm5_smoke arm6_smoke)
for arm in "${ARMS[@]}"; do
    echo
    echo "--- $arm : RL train ---"
    time python -m prm_rl.scripts.train_rl \
        --config "configs/experiments/${arm}.yaml" \
        2>&1 | tee "logs/smoke-${arm}-train.log"

    echo "--- $arm : eval ---"
    time python -m prm_rl.scripts.evaluate \
        --config "configs/experiments/${arm}.yaml" \
        2>&1 | tee "logs/smoke-${arm}-eval.log"
done

echo
echo "======================================================================"
echo "STAGE 4/4 — aggregate results into outputs/smoke_summary.md"
echo "======================================================================"
python -m prm_rl.scripts.summarize_smoke \
    --out outputs/smoke_summary.md \
    outputs/arm1_smoke/eval_results.json \
    outputs/arm2_smoke/eval_results.json \
    outputs/arm3_smoke/eval_results.json \
    outputs/arm4_smoke/eval_results.json \
    outputs/arm5_smoke/eval_results.json \
    outputs/arm6_smoke/eval_results.json

echo
echo "SMOKE ALL ARMS PASSED. Summary: outputs/smoke_summary.md"
