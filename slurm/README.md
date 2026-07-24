# TACC Vista Slurm scripts

Vista uses Slurm with three queues:

| Queue    | Node    | Max nodes/job | Wall time | Charge  |
| -------- | ------- | ------------- | --------- | ------- |
| `gh-dev` | 1× H200 | 8             | 2h        | 1 SU/hr |
| `gh`     | 1× H200 | 64            | 48h       | 1 SU/hr |
| `gg`     | 144 CPU | 32            | 48h       | 0.33 SU |

All scripts here target the `gh` / `gh-dev` GPU queues (each GH node has
**one H200 with 96 GB HBM3**), which is what we need for the 0.5B–7B models
in this project.

## One-time setup

```bash
# From a login node
export ALLOCATION=<your_project_id>       # e.g. TG-ABC12345
bash slurm/setup_env.sh
```

`setup_env.sh` will:

1. `module load gcc cuda python3` (the modules Vista publishes for GH nodes).
2. Create a venv under `$SCRATCH/venvs/prm-rl` (never in `$HOME`).
3. `pip install -e .` for this repo.
4. Cache HF assets under `$SCRATCH/hf-cache` so different jobs share downloads.

## Interactive dev

```bash
idev -p gh-dev -N 1 -n 1 -t 01:00:00
source $SCRATCH/venvs/prm-rl/bin/activate
source $SCRATCH/venvs/prm-rl/vista_env.sh
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Smoke test (before running anything real)

**Purpose:** verify the full RL pipeline (env → HF download → GRPO
train → save → eval) works on a Vista H200 with a "meaningfully bigger"
policy than the Colab quickstart (135M → **Qwen2.5-1.5B-Instruct**),
before spending SUs on a full run.

**One-shot smoke:** trains 10 GRPO steps of Arm 1 (outcome reward only)
on 64 GSM8K problems, then evaluates on 20 test items. Runs on the
`gh-dev` queue in **~15–30 minutes wall time** (well under the 45 min
scripted walltime and the 2 h queue limit).

```bash
# After the one-time setup_env.sh has succeeded:
sbatch slurm/smoke_arm1.slurm

# Watch it:
squeue -u $USER
tail -f logs/arm1-smoke-*.out
```

**Success criteria** (all must hold in the log):

1. `torch: ... True H200` in the preamble.
2. `STAGE 1: RL training` reaches step 10 with a decreasing / non-NaN
   `loss` and non-zero `reward` values in the GRPO logs.
3. `outputs/arm1_smoke/` contains a saved policy (`config.json`,
   `model.safetensors`, tokenizer files).
4. `STAGE 2: Evaluation` prints `outputs/arm1_smoke/eval_results.json`
   and the file has non-null `accuracy.accuracy`, `behavior.avg_tokens`,
   `traps.exploit_rate`, and `crhs.CRHS`.
5. Final line reads `SMOKE TEST PASSED.`

**Scaling up after smoke passes** — reuse the same script + config with
CLI overrides, or copy `arm1_smoke.yaml` to a new experiment file:

```bash
# Bigger model, more steps, more data, same script.
sbatch --time=04:00:00 --job-name=arm1-full -p gh slurm/rl.slurm \
    configs/experiments/arm1_smoke.yaml \
    model.name=Qwen/Qwen2.5-7B-Instruct \
    training.max_steps=500 \
    data.n=2000 \
    output_dir=outputs/arm1_qwen7b
```

## Batch jobs

Every training script takes a config path as its **first positional argument**:

```bash
sbatch slurm/sft.slurm     configs/experiments/sft.yaml
sbatch slurm/prm.slurm     configs/experiments/prm.yaml

# Single arm
sbatch slurm/rl.slurm      configs/experiments/arm2_naive_process.yaml

# Sweep every arm (spawns one job per config)
for c in configs/experiments/arm*.yaml; do sbatch slurm/rl.slurm "$c"; done

# Evaluation on a trained checkpoint
sbatch slurm/eval.slurm    configs/experiments/arm2_naive_process.yaml
```

Job dependencies (RL after PRM after SFT):

```bash
J1=$(sbatch --parsable slurm/sft.slurm configs/experiments/sft.yaml)
J2=$(sbatch --parsable --dependency=afterok:$J1 slurm/prm.slurm configs/experiments/prm.yaml)
J3=$(sbatch --parsable --dependency=afterok:$J2 slurm/rl.slurm  configs/experiments/arm2_naive_process.yaml)
sbatch --dependency=afterok:$J3 slurm/eval.slurm configs/experiments/arm2_naive_process.yaml
```
