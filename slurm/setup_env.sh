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
pip install --index-url https://download.pytorch.org/whl/cu124 "torch>=2.4"

pip install -e "$REPO_ROOT"
# Optional: flash-attn (requires CUDA + gcc). Skip on failure.
pip install "flash-attn>=2.6" --no-build-isolation || true

# Persist HF cache locations for interactive shells.
cat > "$VENV/vista_env.sh" <<EOF
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
