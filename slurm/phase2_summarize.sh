#!/bin/bash
# Aggregate every Phase 2 arm × seed eval_results.json into a single
# markdown comparison table at outputs/phase2_summary.md.
#
# Assumes the 18 outputs/arm{i}_phase2_seed{S}/eval_results.json files
# exist (produced by slurm/phase2_sweep.sh). Missing files are noted
# in the summary as "MISSING" rather than aborting.

set -euo pipefail

FILES=()
for arm in arm1_phase2 arm2_phase2 arm3_phase2 arm4_phase2 arm5_phase2 arm6_phase2; do
    for seed in 42 43 44; do
        f="outputs/${arm}_seed${seed}/eval_results.json"
        if [[ -f "$f" ]]; then
            FILES+=("$f")
        else
            echo "WARN: $f is missing — leaving it out of the summary." >&2
        fi
    done
done

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "ERROR: no eval_results.json files found under outputs/*_phase2_seed*/" >&2
    exit 1
fi

python -m prm_rl.scripts.summarize_smoke \
    --out outputs/phase2_summary.md \
    "${FILES[@]}"

echo
echo "Wrote outputs/phase2_summary.md"
echo "Next: paste the table into thesis_draft/04_chapter4_experiments.md §4.8"
echo "      and re-run pandoc from thesis_draft/README_HOW_TO_ASSEMBLE.md."
