#!/bin/bash
# LOGIN-NODE driver: submits iter_all_arms.sh as an sbatch job, then
# queues the Phase 2 sweep with a Slurm dependency so it waits for
# iter (+ PRM retrain + probe) to finish before starting.
#
# Vista quirk: compute nodes DO NOT have sbatch on PATH, so a wrapper
# that itself tries to sbatch from within a running job fails. This
# script has to be run from the login node — everything past this
# point is Slurm-managed so you can safely close the SSH after.
#
# USAGE (from a login node with the venv activated):
#
#     cd $WORK/prm-rl
#     git pull
#     bash slurm/iter_and_phase2.sh
#
# WHAT IT DOES
#   1. Sbatch iter_all_arms.sh on gh (-t 03:00:00). Captures its jid.
#   2. Runs the Phase 2 sweep with DEPEND_ON=<iter_jid>. All 18 phase2
#      train sbatches are submitted with `--dependency=afterany:<jid>`,
#      so they queue up but stay pending until iter finishes.
#      `afterany` (not afterok) so partial iter failures still yield
#      arms 1/4/6 phase2 data overnight.
#   3. Prints a summary + morning-triage recipe.
#
# ENV OVERRIDES
#   ITER_WALLTIME=03:00:00   how long to give iter (default 3h)
#   SKIP_ITER=1              skip iter, run only phase2 (uses whatever
#                            PRM is currently on disk)
#   Everything phase2_sweep.sh accepts (SEEDS, ARMS, DRY_RUN, ...) is
#   forwarded through the current shell env.

set -euo pipefail

# ----- 0. pre-flight (matches phase2_sweep's checks) -----
if [[ -z "${SCRATCH:-}" || -z "${WORK:-}" ]]; then
    echo "ERROR: \$SCRATCH / \$WORK not set — are you on Vista?" >&2
    exit 1
fi
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "ERROR: no venv activated." >&2
    echo "  source \$SCRATCH/venvs/prm-rl/bin/activate" >&2
    exit 1
fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch not on PATH — are you on a login node?" >&2
    echo "  If you're inside an idev, 'exit' back to login2/login1." >&2
    exit 1
fi

_this_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$_this_dir/.." && pwd)"
cd "$REPO_ROOT"

ITER_WALLTIME="${ITER_WALLTIME:-03:00:00}"

# ----- 1. sbatch iter -----
ITER_JID=""
if [[ "${SKIP_ITER:-0}" == "1" ]]; then
    echo "SKIP_ITER=1 → skipping iter submission; phase2 will use existing PRM."
else
    echo "======================================================================"
    echo "STAGE A: submit iter_all_arms.sh to gh"
    echo "======================================================================"
    ITER_OUT=$(sbatch --parsable \
        -p gh -N 1 -n 1 -t "$ITER_WALLTIME" \
        -J "iter+prm" \
        -o "logs/iter-only-%j.out" \
        -e "logs/iter-only-%j.err" \
        --wrap "set -euo pipefail; source slurm/_common.sh; bash slurm/iter_all_arms.sh")
    # TACC banner strip (same trick as phase2_sweep.sh submit()).
    ITER_JID=$(printf '%s\n' "$ITER_OUT" | awk -F';' '/^[0-9]+/ {j=$1} END {print j}')
    if [[ ! "$ITER_JID" =~ ^[0-9]+$ ]]; then
        echo "ERROR: could not parse iter jid from sbatch output:" >&2
        printf '%s\n' "$ITER_OUT" >&2
        exit 1
    fi
    echo "iter jid = $ITER_JID  (walltime $ITER_WALLTIME on gh)"
fi

# ----- 2. submit phase2 sweep with dependency -----
echo
echo "======================================================================"
echo "STAGE B: submit Phase 2 sweep (18 train + 18 eval jobs)"
echo "======================================================================"
if [[ -n "$ITER_JID" ]]; then
    echo "Each Phase 2 train job will wait for iter jid $ITER_JID to finish"
    echo "(dependency=afterany, so partial iter failure still allows phase2)."
    DEPEND_ON="$ITER_JID" bash slurm/phase2_sweep.sh
else
    bash slurm/phase2_sweep.sh
fi

# ----- 3. summary -----
echo
echo "======================================================================"
echo "SCHEDULED. Overnight plan:"
echo "  * iter jid $ITER_JID runs first (~90-100 min)."
echo "  * 18 phase2 train jobs + 18 phase2 eval jobs are queued behind it."
echo "  * squeue -u \$USER to monitor."
echo
echo "In the morning:"
echo "  tail -100 logs/iter-only-${ITER_JID:-XXXXXX}.out   # iter + PRM probe verdict"
echo "  cat outputs/iter_summary.md                        # Phase 1.5 fix validation"
echo "  bash slurm/phase2_summarize.sh                     # phase2 aggregation"
echo "======================================================================"
