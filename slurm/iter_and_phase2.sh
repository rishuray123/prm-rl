#!/bin/bash
# Overnight driver: rerun Phase 1.5 iter (retrains PRM v2 with the
# fixed hyperparameters), verify the PRM actually discriminates, then
# fire the Phase 2 sweep so it runs while you sleep.
#
# WHY THIS EXISTS
# ---------------
# We diagnosed two PRM failure modes:
#   1. fp16 inference NaN on H200 (fixed in load_prm, commit bc50aea).
#   2. Backbone collapsing to the class prior — pred=0.719 for every
#      input on a 71%-positive dataset — because of right-side
#      truncation, max_length=1024 vs native 512, bf16 gradient
#      underflow on the classifier, and too-conservative lr/epochs
#      (fixed in prm_v2.yaml + prm_train.py, commit 6d7327c).
# This driver validates both fixes end-to-end (STAGE A + STAGE B)
# before it commits ~90 SUs to Phase 2 (STAGE C).
#
# USAGE
# -----
# From a login node with the venv activated:
#
#     cd $WORK/prm-rl
#     sbatch slurm/iter_and_phase2.sh
#
# Then walk away. Sbatch queues iter on the gh partition; when it
# completes (~90-100 min), the same job submits the Phase 2 sweep
# (36 sub-jobs, all --dependency-free — they queue independently on
# gh). This wrapper job then exits. Overnight, Phase 2 finishes at
# its own pace (~4-8 h once nodes free up).
#
# In the morning:
#   bash slurm/phase2_summarize.sh
#
# WHY WE DON'T USE --dependency=afterok
# -------------------------------------
# Slurm dependencies would let iter and phase2 be separate jobs, but
# then we'd have to modify phase2_sweep.sh to accept a jid and add
# --dependency=afterok:$jid to every submission. Inlining iter and
# phase2 into one wrapper is simpler and produces identical timing:
# phase2 is sbatch'd from inside the wrapper after iter finishes,
# and those 36 jobs are then Slurm-managed independently.

#SBATCH -J iter+phase2
#SBATCH -o logs/iter-then-phase2-%j.out
#SBATCH -e logs/iter-then-phase2-%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00

# Note: no `set -e`. We want STAGE C to run even if STAGE A partially
# failed — arms 1/4/6 don't use the PRM and their eval numbers are
# thesis-usable regardless. We handle failures explicitly.
set -uo pipefail

source slurm/_common.sh

# ---------------------------------------------------------------------
# STAGE A — Phase 1.5 iter: rebuild PRM data + retrain PRM v2 with the
# new hyperparams + rerun the 6 iter arms.
# ---------------------------------------------------------------------
echo
echo "########################################################################"
echo "STAGE A: iter_all_arms (retrain PRM v2 with fixed hyperparams)"
echo "########################################################################"
STAGE_A_START=$SECONDS
bash slurm/iter_all_arms.sh
STAGE_A_EXIT=$?
STAGE_A_ELAPSED=$(( SECONDS - STAGE_A_START ))
echo
echo "STAGE A finished with exit=$STAGE_A_EXIT after ${STAGE_A_ELAPSED}s"

if (( STAGE_A_EXIT != 0 )); then
    echo "WARN: iter_all_arms.sh returned $STAGE_A_EXIT."
    echo "      outputs/prm_v2/ may be stale or partial."
    echo "      Continuing to STAGE B/C anyway — arms 1/4/6 don't need PRM"
    echo "      and arm 2/3/5 will fall back to whatever's on disk."
fi

# ---------------------------------------------------------------------
# STAGE B — Verify PRM v2 discriminates. Best-effort; failure here
# does not block STAGE C (we still want overnight Phase 2 results).
# ---------------------------------------------------------------------
echo
echo "########################################################################"
echo "STAGE B: verify PRM v2 discriminates positives from negatives"
echo "########################################################################"
python - <<'PY' || echo "STAGE B probe failed; continuing regardless."
import statistics
import torch
from datasets import load_from_disk
from prm_rl.models.prm import load_prm

d = load_from_disk('data/prm_v2')
prm = load_prm('outputs/prm_v2', device='cuda' if torch.cuda.is_available() else 'cpu')

pos_idx = [i for i, l in enumerate(d['label']) if l == 1][:15]
neg_idx = [i for i, l in enumerate(d['label']) if l == 0][:15]
pos_scores = [prm.score_steps(d[i]['question'], [d[i]['step']])[0] for i in pos_idx]
neg_scores = [prm.score_steps(d[i]['question'], [d[i]['step']])[0] for i in neg_idx]

pm, nm = statistics.mean(pos_scores), statistics.mean(neg_scores)
print(f'positives: mean={pm:.3f}  min={min(pos_scores):.3f}  max={max(pos_scores):.3f}')
print(f'negatives: mean={nm:.3f}  min={min(neg_scores):.3f}  max={max(neg_scores):.3f}')
print(f'discrimination gap = {pm - nm:.3f}')

if pm - nm > 0.20:
    print('OK: PRM discriminates cleanly. Phase 2 process-based arms will get real signal.')
elif pm - nm > 0.05:
    print('WEAK: PRM gap in [0.05, 0.20]. Some signal for process arms, but noisy.')
else:
    print('BROKEN: gap < 0.05. Process-based arms will train against near-constant reward.')
    print('        Arms 1/4/6 (no PRM) will still produce thesis-usable numbers.')
PY

# ---------------------------------------------------------------------
# STAGE C — Fire the Phase 2 sweep. 36 sub-jobs (18 train + 18 eval)
# get sbatch'd here and Slurm manages them independently.
# ---------------------------------------------------------------------
echo
echo "########################################################################"
echo "STAGE C: submit Phase 2 sweep (18 train + 18 eval sbatch jobs)"
echo "########################################################################"
# Qwen2.5-7B is already in HF_HOME from the earlier prefetch on the
# login node; skip the re-download attempt here (compute nodes may
# have flaky egress).
export SKIP_PREFETCH=1
bash slurm/phase2_sweep.sh
STAGE_C_EXIT=$?
echo
echo "STAGE C finished with exit=$STAGE_C_EXIT"

if (( STAGE_C_EXIT != 0 )); then
    echo "ERROR: phase2_sweep.sh failed to submit. Investigate the log above."
    exit 1
fi

echo
echo "########################################################################"
echo "ALL SCHEDULED. Phase 2 jobs are queued on gh; sleep well."
echo "########################################################################"
echo
echo "Morning routine:"
echo "  squeue -u \$USER                    # see remaining Phase 2 jobs"
echo "  cat outputs/iter_summary.md         # Phase 1.5 fix validation"
echo "  bash slurm/phase2_summarize.sh      # aggregate Phase 2 results"
