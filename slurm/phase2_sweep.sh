#!/bin/bash
# Phase 2 sweep — the "real" run.
#
# Fans out 6 arms × 3 seeds = 18 sbatch jobs on the gh queue at
# Qwen2.5-7B / 500 GRPO steps / n_test=500. Each job is standalone:
# it (a) trains, (b) evaluates, (c) writes eval_results.json into its
# own output_dir. Aggregation is a separate step (phase2_summarize.sh).
#
# All trained checkpoints and eval JSONs are routed to
#   $SCRATCH/prm-rl-outputs/{arm}_seed{S}/
# (NOT $WORK/prm-rl/outputs/) — /home1 and /work quotas are too small
# for 18 × 14 GB Qwen2.5-7B checkpoints. Logs stay in
# $WORK/prm-rl/logs/ (small text files).
#
# Usage (from a login node, with the venv activated and modules loaded):
#
#     bash slurm/phase2_sweep.sh
#
# Optional env overrides:
#   SEEDS="42 43"          # limit to a subset of seeds
#   ARMS="arm1_phase2"     # limit to a subset of arms
#   SKIP_PREFETCH=1        # skip the login-node model prefetch
#   DRY_RUN=1              # print sbatch commands but don't submit
#   DEPEND_ON=<jid>        # every train job is submitted with
#                          # --dependency=afterany:$jid so it waits for
#                          # the given job (typically iter_all_arms) to
#                          # finish. `afterany` (not afterok) so that a
#                          # partial iter failure still yields useful
#                          # arm 1/4/6 phase2 data overnight.
#
# Results summary: docs/results.md. Cache dirs: slurm/env_caches.sh.

set -euo pipefail

# -----------------------------------------------------------------------------
# 0. Pre-flight: environment
# -----------------------------------------------------------------------------
if [[ -z "${SCRATCH:-}" ]]; then
    echo "ERROR: \$SCRATCH is not set — are you on a Vista node?" >&2
    exit 1
fi
if [[ -z "${WORK:-}" ]]; then
    echo "ERROR: \$WORK is not set — are you on a Vista node?" >&2
    exit 1
fi
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "ERROR: no venv activated. Run:" >&2
    echo "  module reset && module load gcc cuda python3" >&2
    echo "  source \$SCRATCH/venvs/prm-rl/bin/activate" >&2
    echo "  # env_caches.sh will be sourced automatically below" >&2
    exit 1
fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch not on PATH — are you on a login node?" >&2
    exit 1
fi

# Source the shared cache-dir env so HF/Triton/etc land on $SCRATCH
_this_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_this_dir/env_caches.sh"

# -----------------------------------------------------------------------------
# 1. Pre-flight: repository state (Phase 1.5 must have completed)
# -----------------------------------------------------------------------------
REPO_ROOT="$(cd "$_this_dir/.." && pwd)"
cd "$REPO_ROOT"

MISSING=()
[[ -d data/golden_v2 ]] || MISSING+=("data/golden_v2 (build via iter_all_arms.sh stage 1)")
[[ -d data/prm_v2 ]]    || MISSING+=("data/prm_v2    (build via iter_all_arms.sh stage 2)")
[[ -d outputs/prm_v2 ]] || MISSING+=("outputs/prm_v2 (train via iter_all_arms.sh stage 3)")
if (( ${#MISSING[@]} )); then
    echo "ERROR: Phase 1.5 artefacts missing. Run slurm/iter_all_arms.sh first." >&2
    printf '  missing: %s\n' "${MISSING[@]}" >&2
    exit 1
fi

# Sanity: config files exist for every arm/phase2.
DEFAULT_ARMS="arm1_phase2 arm2_phase2 arm3_phase2 arm4_phase2 arm5_phase2 arm6_phase2"
ARMS="${ARMS:-$DEFAULT_ARMS}"
SEEDS="${SEEDS:-42 43 44}"
for arm in $ARMS; do
    cfg="configs/experiments/${arm}.yaml"
    [[ -f "$cfg" ]] || { echo "ERROR: $cfg missing" >&2; exit 1; }
done

# -----------------------------------------------------------------------------
# 2. Output layout on $SCRATCH
# -----------------------------------------------------------------------------
SCRATCH_OUT="${SCRATCH}/prm-rl-outputs"
mkdir -p "$SCRATCH_OUT" logs

# Report disk situation before we commit to submitting.
echo "======================================================================"
echo "Phase 2 sweep pre-flight"
echo "======================================================================"
echo "Repo root:         $REPO_ROOT"
echo "Output root:       $SCRATCH_OUT"
echo "HF_HOME:           $HF_HOME"
echo "TRITON_CACHE_DIR:  $TRITON_CACHE_DIR"
echo "Arms:              $ARMS"
echo "Seeds:             $SEEDS"
if command -v df >/dev/null; then
    echo
    echo "Disk usage:"
    df -h "$WORK" "$SCRATCH" 2>/dev/null | awk 'NR==1||/\/(work|scratch)/'
fi

# Rough cost/space estimate
N_ARMS=$(echo "$ARMS" | wc -w)
N_SEEDS=$(echo "$SEEDS" | wc -w)
N_JOBS=$(( N_ARMS * N_SEEDS ))
EST_SU=$(( N_JOBS * 4 ))     # ~4 SUs / job at 5h walltime request
EST_STORE=$(( N_JOBS * 14 )) # ~14 GB / job final checkpoint
echo
echo "Estimated cost:    $N_JOBS train jobs × ~4 SU each ≈ ${EST_SU} SU (+eval ≈ ${N_JOBS} SU)"
echo "Estimated storage: ~${EST_STORE} GB of final checkpoints on \$SCRATCH"
echo "                   (peak ~$(( EST_STORE * 3 )) GB during training with save_total_limit=2)"

# Optional Slurm dependency (used by the overnight wrapper so phase2
# waits for iter_all_arms + PRM retrain before starting).
DEPEND_ON="${DEPEND_ON:-}"
if [[ -n "$DEPEND_ON" ]]; then
    if [[ ! "$DEPEND_ON" =~ ^[0-9]+$ ]]; then
        echo "ERROR: DEPEND_ON must be a numeric jid, got: $DEPEND_ON" >&2
        exit 1
    fi
    echo "Dependency:        every train job will --dependency=afterany:${DEPEND_ON}"
fi

# -----------------------------------------------------------------------------
# 3. Optional: prefetch Qwen2.5-7B on the login node so 18 concurrent
#    GPU jobs don't all race to download the same 14 GB of weights.
# -----------------------------------------------------------------------------
if [[ "${SKIP_PREFETCH:-0}" != "1" ]]; then
    echo
    echo "======================================================================"
    echo "Prefetching Qwen/Qwen2.5-7B-Instruct into HF_HOME (login node)"
    echo "======================================================================"
    # snapshot_download will re-use existing cache entries and only fetch missing files.
    python - <<'PY'
import os, sys, time
from huggingface_hub import snapshot_download
start = time.time()
path = snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    allow_patterns=[
        "*.json", "*.txt", "*.model", "*.safetensors",
        "tokenizer*", "vocab.json", "merges.txt",
    ],
    max_workers=8,
)
print(f"OK: cached at {path}  ({time.time() - start:.0f}s)", file=sys.stderr)
PY
fi

# -----------------------------------------------------------------------------
# 4. Submit train + dependent eval per (arm, seed)
# -----------------------------------------------------------------------------
echo
echo "======================================================================"
echo "Submitting ${N_JOBS} train + ${N_JOBS} eval jobs"
echo "======================================================================"

submit() {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "  [dry-run] $*" >&2
        echo "DRY_$RANDOM"
        return 0
    fi
    local out jid
    # TACC's sbatch wrapper prints a multi-line validation banner on
    # stdout ("Welcome to the Vista Supercomputer ... --> Checking
    # allocation ...") IN ADDITION to the jid; --parsable does NOT
    # suppress the banner. Naively capturing `sbatch --parsable` puts
    # the whole banner into $JID, which then breaks `--dependency=
    # afterok:$JID`. So we capture stdout, then extract the last line
    # that looks like a jid ("<digits>" or "<digits>;<cluster>").
    if ! out=$(sbatch --parsable "$@"); then
        echo "ERROR: sbatch exited non-zero:" >&2
        printf '%s\n' "$out" >&2
        return 1
    fi
    jid=$(printf '%s\n' "$out" | awk -F';' '/^[0-9]+/ {j=$1} END {print j}')
    if [[ ! "$jid" =~ ^[0-9]+$ ]]; then
        echo "ERROR: could not parse jid from sbatch output:" >&2
        printf '%s\n' "$out" >&2
        return 1
    fi
    printf '%s\n' "$jid"
}

TRAIN_IDS=()
EVAL_IDS=()

for arm in $ARMS; do
    for seed in $SEEDS; do
        name="${arm}-s${seed}"
        outdir="${SCRATCH_OUT}/${arm}_seed${seed}"
        cfg="configs/experiments/${arm}.yaml"
        mkdir -p "$outdir"

        echo
        echo "→ $name"
        echo "   config = $cfg"
        echo "   outdir = $outdir"

        JID=$(submit \
            -p gh \
            --time=05:00:00 \
            -J "$name" \
            -o "logs/${name}-%j.out" \
            -e "logs/${name}-%j.err" \
            ${DEPEND_ON:+--dependency=afterany:${DEPEND_ON}} \
            slurm/rl.slurm "$cfg" \
                "seed=${seed}" \
                "output_dir=${outdir}" \
                "run_name=${name}" \
                "eval.policy_path=${outdir}") || exit 1
        TRAIN_IDS+=("$JID")
        echo "   train jid = $JID"

        EID=$(submit \
            -p gh \
            --time=01:00:00 \
            -J "${name}-eval" \
            -o "logs/${name}-eval-%j.out" \
            -e "logs/${name}-eval-%j.err" \
            --dependency=afterok:${JID} \
            slurm/eval.slurm "$cfg" \
                "seed=${seed}" \
                "output_dir=${outdir}" \
                "eval.policy_path=${outdir}") || exit 1
        EVAL_IDS+=("$EID")
        echo "   eval  jid = $EID  (depends on ${JID})"
    done
done

# -----------------------------------------------------------------------------
# 5. Summary
# -----------------------------------------------------------------------------
echo
echo "======================================================================"
echo "Submitted ${#TRAIN_IDS[@]} train jobs and ${#EVAL_IDS[@]} eval jobs"
echo "======================================================================"
echo
echo "Watch progress:"
echo "  squeue -u \$USER --sort=+i"
echo
echo "When all jobs complete (or as they trickle in), aggregate:"
echo "  bash slurm/phase2_summarize.sh"
echo
echo "Results will land under: $SCRATCH_OUT/{arm}_seed{S}/eval_results.json"
