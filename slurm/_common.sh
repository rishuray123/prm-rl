# Shared preamble sourced by every Slurm script.
# NOTE: no shebang — this file is `source`d, not executed.

module reset
module load gcc cuda python3

VENV="${VENV:-$SCRATCH/venvs/wordwave}"
if [[ ! -d "$VENV" ]]; then
    echo "ERROR: venv $VENV not found. Run slurm/setup_env.sh first." >&2
    exit 1
fi
source "$VENV/bin/activate"
if [[ -f "$VENV/vista_env.sh" ]]; then
    source "$VENV/vista_env.sh"
fi

export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
export WANDB_MODE=${WANDB_MODE:-disabled}
export OMP_NUM_TPHREADS=${OMP_NUM_TPHREADS:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

cd "${WORDWAVE_HOME:-$SLURM_SUBMIT_DIR}"
mkdir -p logs

echo "----------------------------------------"
echo "Job:       $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node(s):   $SLURM_JOB_NODELIST"
echo "cwd:       $PWD"
echo "python:    $(which python)"
echo "torch:     $(python -c 'import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-cuda")')"
echo "----------------------------------------"
