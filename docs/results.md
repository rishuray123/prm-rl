# Experimental results

Public summary of the completed evaluation ladder for Arms 1–6.
Configs live under `configs/experiments/`; aggregation scripts under
`slurm/`.

## Arms

| Arm | Design | Config stem |
|-----|--------|-------------|
| 1 | Outcome only (baseline) | `arm1_*` |
| 2 | Naive process (raw PRM mean) | `arm2_*` |
| 3 | Prefix consistency (PRM until first error) | `arm3_*` |
| 4 | Contradiction-aware (outcome + process + NLI) | `arm4_*` |
| 5 | Counterfactual (outcome + numeric grounding) | `arm5_*` |
| 6 | Hybrid PROGRS-style (outcome-conditioned centering) | `arm6_*` |

PRM for Phase 1.5 / Phase 2: DistilBERT-base-uncased trained with
synthetic negatives (`configs/experiments/prm_v2.yaml`,
`inject_negatives_prob=0.5`).

## Phase 1.5 iteration (1.5B)

**Setup:** Qwen2.5-1.5B-Instruct, 50 GRPO steps, n_train=256,
n_test=100, seed 42.

| arm | design | accuracy | process_correctness | exploit_rate | CRHS |
|-----|--------|----------|---------------------|--------------|------|
| arm1_iter | outcome | 0.540 | 0.964 | 0.400 | 0.458 |
| arm2_iter | naive process | 0.450 | 0.985 | 0.400 | 0.445 |
| arm3_iter | prefix consistency | 0.440 | 0.974 | 0.600 | 0.367 |
| arm4_iter | contradiction | 0.550 | 0.967 | 0.600 | 0.356 |
| arm5_iter | counterfactual | 0.620 | 0.978 | 0.400 | 0.447 |
| arm6_iter | hybrid | 0.630 | 0.971 | 0.400 | 0.420 |

**Takeaway:** After synthetic-negative PRM training,
`process_correctness` is non-zero. Arms that use the PRM as the
sole reward (2, 3) underperform the outcome baseline; hybrid /
counterfactual beat baseline at this scale.

## Phase 2 sweep (7B × 3 seeds)

**Setup:** Qwen2.5-7B-Instruct, 500 GRPO steps, n_train=2000,
n_test=500, num_generations=8, seeds {42, 43, 44}. Values are
mean ± sample std over seeds.

| arm | design | accuracy | process_correctness | avg_tokens | avg_steps | self_rougeL | exploit | CRHS |
|-----|--------|----------|---------------------|------------|-----------|-------------|---------|------|
| arm1 | outcome | **0.861±0.008** | 0.987±0.001 | 222±23 | 7.0±0.3 | 0.291±0.005 | 0.27±0.12 | 0.459±0.034 |
| arm2 | naive PRM | 0.571±0.281 | **0.995±0.008** | 246±63 | **14.9±5.8** | 0.255±0.162 | 0.33±0.23 | 0.412±0.129 |
| arm3 | prefix PRM | 0.540±0.181 | **0.999±0.002** | **278±2** | **14.7±2.2** | **0.414±0.083** | 0.27±0.12 | 0.403±0.042 |
| arm4 | contradiction | **0.869±0.006** | 0.991±0.005 | 210±43 | **1.9±0.3** | **0.023±0.010** | 0.27±0.12 | 0.472±0.066 |
| arm5 | counterfactual | **0.862±0.017** | 0.985±0.003 | 208±20 | 7.0±0.3 | 0.287±0.006 | 0.27±0.12 | **0.474±0.021** |
| arm6 | hybrid | **0.869±0.007** | 0.985±0.007 | 237±24 | 5.2±1.1 | 0.180±0.089 | 0.27±0.12 | 0.444±0.019 |

### Pre-registered hypothesis outcomes

| ID | Prediction | Verdict |
|----|------------|---------|
| H1 | arm6 > arm5 > arm4 > arm1 > arm2 > arm3 | Partially confirmed — arms 2/3 last; top four tied near 0.86 |
| H2 | exploit_rate for arms 2/3 grows | Not confirmed (`n_traps=5` too coarse) |
| H3 | proc≈1 and accuracy low for arms 2/3 | **Confirmed** |
| H4 | CRHS separates; arm6 > 0.5 | Not confirmed |

### Headline findings

1. **Pure process reward collapses at 7B.** Arms 2 and 3 lose ~30
   accuracy points vs outcome while `process_correctness` approaches
   1.0 (specification gaming of the PRM).
2. **Composition prevents collapse but does not beat outcome.**
   Arms 5 and 6 match Arm 1 within ~1 pp at 7B.
3. **Arm 4 finds another exploit:** baseline-level accuracy with
   `avg_steps` collapsed to ~1.9 (evading the NLI contradiction
   penalty).

Reproduce aggregation with `bash slurm/phase2_summarize.sh` after a
sweep (reads `$SCRATCH/prm-rl-outputs/` on Vista).
