# Shared preamble sourced by every Slurm script.
# NOTE: no shebang — this file is `source`d, not executed.

module reset
module load gcc cuda python3

VENV="${VENV:-$SCRATCH/venvs/prm-rl}"
if [[ ! -d "$VENV" ]]; then
    echo "ERROR: venv $VENV not found. Run slurm/setup_env.sh first." >&2
    exit 1
fi
source "$VENV/bin/activate"
if [[ -f "$VENV/vista_env.sh" ]]; then
    source "$VENV/vista_env.sh"
fi

# Redirect every cache dir off /home1 (quota is tiny). Belt-and-braces
# with vista_env.sh above — safe to source both.
_this_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$_this_dir/env_caches.sh" ]]; then
    # shellcheck disable=SC1091
    source "$_this_dir/env_caches.sh"
fi
unset _this_dir

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
export WANDB_MODE=${WANDB_MODE:-disabled}
export OMP_NUM_TPHREADS=${OMP_NUM_TPHREADS:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

cd "${PRMRL_HOME:-$SLURM_SUBMIT_DIR}"
mkdir -p logs

echo "----------------------------------------"
echo "Job:       $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node(s):   $SLURM_JOB_NODELIST"
echo "cwd:       $PWD"
echo "python:    $(which python)"
echo "torch:     $(python -c 'import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-cuda")')"
echo "----------------------------------------"
