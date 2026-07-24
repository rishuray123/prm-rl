#!/bin/bash
# One-time environment bootstrap for TACC Vista (login node).
# Creates a venv on $SCRATCH, installs prm_rl in editable mode, and sets
# HF caches to live on $SCRATCH so multiple jobs share downloads.

set -euo pipefail

module reset
module load gcc cuda python3

VENV="${VENV:-$SCRATCH/venvs/prm-rl}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-$SCRATCH/hf-cache}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$HF_CACHE_ROOT"
if [[ ! -d "$VENV" ]]; then
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# Torch for GH200 (H200 + Grace ARM CPU). Vista's system CUDA module is
# forward-compatible; the CUDA 12.4 wheels published by pytorch match the
# H200 driver stack.
#
# NOTE: torch >= 2.6 is required by TRL 0.17+ / 1.x (it imports FSDPModule
# from torch.distributed.fsdp, which only exists in the FSDP2 API added
# in torch 2.6). Do NOT downgrade this constraint.
pip install --index-url https://download.pytorch.org/whl/cu124 "torch>=2.6"

pip install -e "$REPO_ROOT"
# Optional: flash-attn (requires CUDA + gcc). Skip on failure — SDPA is a
# fine fallback for the model sizes in this project. Vista's gcc/14 stack
# currently fails to build flash-attn (nvcc supports gcc <= 13), which is
# expected; the `|| true` keeps setup non-fatal.
pip install "flash-attn>=2.6" --no-build-isolation || true

# Persist HF cache locations + module loads for interactive/batch shells.
# Sourcing this on a compute node is what makes libpython3.11.so.1.0 and
# nvcc visible in the linker/PATH; without it, `python` inside the venv
# fails with `error while loading shared libraries`.
cat > "$VENV/vista_env.sh" <<EOF
# Load the Vista modules the venv links against. Safe on login and
# compute nodes; \`module reset\` is a no-op if modules are already set.
module reset >/dev/null 2>&1 || true
module load gcc cuda python3 >/dev/null 2>&1 || true

export HF_HOME="$HF_CACHE_ROOT"
export HF_DATASETS_CACHE="$HF_CACHE_ROOT/datasets"
export TRANSFORMERS_CACHE="$HF_CACHE_ROOT/transformers"
export TOKENIZERS_PARALLELISM=false
export PRMRL_HOME="$REPO_ROOT"
EOF

echo
echo "Setup complete."
echo "  venv:      $VENV"
echo "  HF cache:  $HF_CACHE_ROOT"
echo "  repo:      $REPO_ROOT"
echo
echo "Activate future sessions with:"
echo "  source $VENV/bin/activate && source $VENV/vista_env.sh"
