#!/bin/bash
# Aggregate every Phase 2 arm × seed eval_results.json into a single
# markdown comparison table.
#
# Phase 2 outputs live under $SCRATCH/prm-rl-outputs/{arm}_seed{S}/
# (see slurm/phase2_sweep.sh). The final summary table is written to
# $WORK/prm-rl/outputs/phase2_summary.md so it lives next to the repo.
#
# Missing files are noted as "MISSING" rather than aborting — so this
# script is safe to run before every last seed has finished.
#
# Usage (from a login node, with the venv activated):
#   bash slurm/phase2_summarize.sh
#
# Optional env overrides:
#   SEEDS="42 43 44"
#   ARMS="arm1_phase2 arm2_phase2 ..."
#   OUT_ROOT=/some/other/path   # overrides $SCRATCH/prm-rl-outputs

set -euo pipefail

if [[ -z "${SCRATCH:-}" && -z "${OUT_ROOT:-}" ]]; then
    echo "ERROR: \$SCRATCH is not set (are you on a Vista node?)" >&2
    echo "       or pass OUT_ROOT=/path/to/prm-rl-outputs explicitly." >&2
    exit 1
fi

OUT_ROOT="${OUT_ROOT:-${SCRATCH}/prm-rl-outputs}"
ARMS="${ARMS:-arm1_phase2 arm2_phase2 arm3_phase2 arm4_phase2 arm5_phase2 arm6_phase2}"
SEEDS="${SEEDS:-42 43 44}"

echo "Scanning $OUT_ROOT"
echo "  arms:  $ARMS"
echo "  seeds: $SEEDS"

FILES=()
MISSING_COUNT=0
for arm in $ARMS; do
    for seed in $SEEDS; do
        f="${OUT_ROOT}/${arm}_seed${seed}/eval_results.json"
        if [[ -f "$f" ]]; then
            FILES+=("$f")
        else
            echo "WARN: $f is missing — leaving it out of the summary." >&2
            MISSING_COUNT=$(( MISSING_COUNT + 1 ))
        fi
    done
done

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "ERROR: no eval_results.json files found under $OUT_ROOT" >&2
    exit 1
fi

mkdir -p outputs
python -m prm_rl.scripts.summarize_smoke \
    --out outputs/phase2_summary.md \
    "${FILES[@]}"

echo
echo "Wrote outputs/phase2_summary.md  (${#FILES[@]} results in, ${MISSING_COUNT} missing)"
echo "Next: paste the table into thesis_draft/04_chapter4_experiments.md §4.8"
echo "      and re-run pandoc from thesis_draft/README_HOW_TO_ASSEMBLE.md."
