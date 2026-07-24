#!/bin/bash
# Phase 2 sweep — the "real" run.
#
# Fans out 6 arms × 3 seeds = 18 sbatch jobs onto the gh queue at
# Qwen2.5-7B / 500 GRPO steps / n_test=500. Each job is standalone:
# it (a) trains, (b) evaluates, (c) writes eval_results.json into its
# own output_dir. Aggregation is a separate step below.
#
# Usage (from repo root, on a login node):
#
#     bash slurm/phase2_sweep.sh
#
# What the sweep assumes exists BEFORE it runs:
#   * data/golden_v2/       — built by iter_all_arms.sh or manually via
#                             `python -m prm_rl.scripts.build_golden
#                             --n 2000 --out data/golden_v2 --strategy
#                             gsm8k_native`
#   * data/prm_v2/          — built by iter_all_arms.sh or manually via
#                             `python -m prm_rl.scripts.build_prm_data
#                             --golden data/golden_v2 --out data/prm_v2
#                             --inject_negatives_prob 0.5 --seed 0`
#   * outputs/prm_v2/       — trained by iter_all_arms.sh or via
#                             `python -m prm_rl.scripts.train_prm
#                             --config configs/experiments/prm_v2.yaml`
#
# Budget:
#   * Wall time per job: ≤ 5 h (requested via --time=05:00:00).
#   * Cost per job:      ~2–4 SUs on ASC26008.
#   * Total sweep cost:  ~40–70 SUs (well under our 9504 SU budget).
#
# After all 18 jobs complete, run slurm/phase2_summarize.sh (see below)
# to aggregate eval_results.json across seeds into one markdown table
# with mean ± std per metric per arm.

set -euo pipefail

ARMS=(arm1_phase2 arm2_phase2 arm3_phase2 arm4_phase2 arm5_phase2 arm6_phase2)
SEEDS=(42 43 44)

mkdir -p logs

JOBIDS=()
for arm in "${ARMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        name="${arm}-s${seed}"
        outdir="outputs/${arm}_seed${seed}"
        cfg="configs/experiments/${arm}.yaml"

        echo "→ submitting $name  (config=$cfg, seed=$seed, out=$outdir)"

        JID=$(sbatch --parsable \
            -p gh \
            --time=05:00:00 \
            -J "$name" \
            -o "logs/${name}-%j.out" \
            -e "logs/${name}-%j.err" \
            slurm/rl.slurm "$cfg" \
                "seed=${seed}" \
                "output_dir=${outdir}" \
                "run_name=${name}" \
                "eval.policy_path=${outdir}")
        JOBIDS+=("$JID")
        echo "   train jid=$JID"

        # Chain the eval job so it starts only after train is done.
        EID=$(sbatch --parsable \
            -p gh \
            --time=01:00:00 \
            -J "${name}-eval" \
            -o "logs/${name}-eval-%j.out" \
            -e "logs/${name}-eval-%j.err" \
            --dependency=afterok:${JID} \
            slurm/eval.slurm "$cfg" \
                "seed=${seed}" \
                "output_dir=${outdir}" \
                "eval.policy_path=${outdir}")
        JOBIDS+=("$EID")
        echo "   eval  jid=$EID  (depends on $JID)"
    done
done

echo
echo "Submitted ${#JOBIDS[@]} jobs."
echo "Watch with:   squeue -u \$USER --sort=+i"
echo "When all jobs are complete, aggregate with:"
echo "  bash slurm/phase2_summarize.sh"
