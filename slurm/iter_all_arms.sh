#!/bin/bash
# Phase 1.5 — mid-scale iteration of all six arms.
#
# Runs the six *_iter.yaml configs sequentially on a single GH200 node
# with the PRM v2 (synthetic negatives). Sized to fit inside gh-dev's
# 2 h wall cap for fast iteration without waiting in the gh queue.
#
# Usage (from repo root, with venv activated, modules loaded — see
# slurm/README.md):
#
#     bash slurm/iter_all_arms.sh
#
# Or interactively:
#
#     idev -p gh-dev -t 02:00:00
#     source $SCRATCH/venvs/prm-rl/bin/activate
#     bash slurm/iter_all_arms.sh
#
# Per-arm wall-clock budget on H200:
#   * build_golden (n=2000)           ~ 30 s
#   * build_prm_data (v2, negs=0.5)   ~ 30 s
#   * train_prm (v2, deberta-v3-base) ~ 3–5 min
#   * train_rl (50 GRPO steps)        ~ 8–12 min per arm
#   * evaluate (n_test=100)           ~ 2–4 min per arm
#   * total (PRM + 6 arms)            ~ 75–105 min. Safely under 2 h.
#
# ⚠️  If you change max_steps or n_test in the *_iter.yaml configs, redo
#     the math before submitting to gh-dev — a 2 h SIGKILL loses all
#     the arms after the one that was running when the wall hit.

set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "ERROR: VIRTUAL_ENV not set. Activate the venv first (KB §2.2)." >&2
    exit 1
fi

# Redirect HF / Triton / pip / matplotlib caches to $SCRATCH so we
# don't blow the tiny /home1 quota. See KB §2.9.
_this_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_this_dir/env_caches.sh"
unset _this_dir

echo "Cache targets:"
echo "  HF_HOME          = $HF_HOME"
echo "  TRITON_CACHE_DIR = $TRITON_CACHE_DIR"
echo "  XDG_CACHE_HOME   = $XDG_CACHE_HOME"

mkdir -p logs data outputs

echo "======================================================================"
echo "STAGE 1/5 — build golden dataset v2 (n=2000, gsm8k_native)"
echo "======================================================================"
time python -m prm_rl.scripts.build_golden \
    --split train --n 2000 --out data/golden_v2 --strategy gsm8k_native \
    2>&1 | tee logs/iter-golden.log

echo
echo "======================================================================"
echo "STAGE 2/5 — build PRM training data with synthetic negatives"
echo "           (--inject_negatives_prob 0.5, seed 0)"
echo "======================================================================"
time python -m prm_rl.scripts.build_prm_data \
    --golden data/golden_v2 --out data/prm_v2 \
    --inject_negatives_prob 0.5 --seed 0 \
    2>&1 | tee logs/iter-prm-data.log

echo
echo "======================================================================"
echo "STAGE 3/5 — train PRM v2 (DeBERTa-v3-base, 3 epochs)"
echo "======================================================================"
time python -m prm_rl.scripts.train_prm \
    --config configs/experiments/prm_v2.yaml \
    2>&1 | tee logs/iter-prm-train.log

echo
echo "======================================================================"
echo "STAGE 4/5 — RL iteration (50 GRPO steps) + eval for each arm"
echo "======================================================================"
ARMS=(arm1_iter arm2_iter arm3_iter arm4_iter arm5_iter arm6_iter)
for arm in "${ARMS[@]}"; do
    echo
    echo "--- $arm : RL train ---"
    time python -m prm_rl.scripts.train_rl \
        --config "configs/experiments/${arm}.yaml" \
        2>&1 | tee "logs/iter-${arm}-train.log"

    echo "--- $arm : eval ---"
    time python -m prm_rl.scripts.evaluate \
        --config "configs/experiments/${arm}.yaml" \
        2>&1 | tee "logs/iter-${arm}-eval.log"
done

echo
echo "======================================================================"
echo "STAGE 5/5 — aggregate into outputs/iter_summary.md"
echo "======================================================================"
python -m prm_rl.scripts.summarize_smoke \
    --out outputs/iter_summary.md \
    outputs/arm1_iter/eval_results.json \
    outputs/arm2_iter/eval_results.json \
    outputs/arm3_iter/eval_results.json \
    outputs/arm4_iter/eval_results.json \
    outputs/arm5_iter/eval_results.json \
    outputs/arm6_iter/eval_results.json

echo
echo "PHASE 1.5 ITER COMPLETE. Summary: outputs/iter_summary.md"
echo "Next step: (a) sanity-check process_correctness ≠ 1.000 across arms,"
echo "           (b) if OK, promote to Phase 2 via slurm/phase2_sweep.sh."
