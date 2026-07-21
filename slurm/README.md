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
2. Create a venv under `$SCRATCH/venvs/wordwave` (never in `$HOME`).
3. `pip install -e .` for this repo.
4. Cache HF assets under `$SCRATCH/hf-cache` so different jobs share downloads.

## Interactive dev

```bash
idev -p gh-dev -N 1 -n 1 -t 01:00:00
source $SCRATCH/venvs/wordwave/bin/activate
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
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
