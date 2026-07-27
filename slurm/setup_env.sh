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

# Persist HF cache locations for interactive/batch shells.
#
# IMPORTANT: this file MUST NOT run `module load` — module loads prepend
# the system Python's bin directory to $PATH, and if that happens AFTER
# venv activation, it clobbers the venv's bin ordering so `pip` writes
# to the *system* site-packages (which is not writable, triggering the
# infamous "Defaulting to user installation" fallback) and `python -m`
# invokes the wrong interpreter. Load modules BEFORE activating the venv
# (see the printed instructions and slurm/_common.sh).
cat > "$VENV/vista_env.sh" <<EOF
# Cache-dir redirection: keep everything off of /home1 (23 GB quota).
# Mirrors slurm/env_caches.sh — both should stay in sync.
_prmrl_cache_root="\${SCRATCH}/prm-rl-caches"
export HF_HOME="\${HF_HOME:-\${_prmrl_cache_root}/huggingface}"
export HF_HUB_CACHE="\${HF_HUB_CACHE:-\${HF_HOME}/hub}"
export HF_DATASETS_CACHE="\${HF_DATASETS_CACHE:-\${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="\${TRANSFORMERS_CACHE:-\${HF_HOME}/transformers}"
export TRITON_CACHE_DIR="\${TRITON_CACHE_DIR:-\${_prmrl_cache_root}/triton}"
export XDG_CACHE_HOME="\${XDG_CACHE_HOME:-\${_prmrl_cache_root}/xdg}"
export PIP_CACHE_DIR="\${PIP_CACHE_DIR:-\${_prmrl_cache_root}/pip}"
export MPLCONFIGDIR="\${MPLCONFIGDIR:-\${_prmrl_cache_root}/matplotlib}"
export TORCHINDUCTOR_CACHE_DIR="\${TORCHINDUCTOR_CACHE_DIR:-\${_prmrl_cache_root}/torchinductor}"
mkdir -p "\$HF_HOME" "\$HF_HUB_CACHE" "\$HF_DATASETS_CACHE" \\
    "\$TRANSFORMERS_CACHE" "\$TRITON_CACHE_DIR" "\$XDG_CACHE_HOME" \\
    "\$PIP_CACHE_DIR" "\$MPLCONFIGDIR" "\$TORCHINDUCTOR_CACHE_DIR"
unset _prmrl_cache_root

export TOKENIZERS_PARALLELISM=false
export PRMRL_HOME="$REPO_ROOT"
EOF

echo
echo "Setup complete."
echo "  venv:      $VENV"
echo "  HF cache:  $HF_CACHE_ROOT"
echo "  repo:      $REPO_ROOT"
echo
echo "Activate future sessions with (order matters — modules FIRST):"
echo "  module reset && module load gcc cuda python3"
echo "  source $VENV/bin/activate"
echo "  source $VENV/vista_env.sh"
