# slurm/env_caches.sh — Redirect every cache directory off of $HOME.
#
# Vista's /home1 quota is ~23 GB and easily filled by HuggingFace and
# Triton caches when running any HF-backed workload. This file exports
# the env vars needed to send those caches to $SCRATCH instead.
#
# It's sourced (not executed) by:
#   * slurm/_common.sh                 — every batch job.
#   * slurm/iter_all_arms.sh           — Phase 1.5 driver.
#   * slurm/smoke_all_arms.sh          — original smoke driver.
#
# Idempotent: safe to source multiple times. Creates the target
# directories if they don't exist.
#
# See docs/knowledge-base.md §2.1 (home quota) and §2.9 (this fix).

if [[ -z "${SCRATCH:-}" ]]; then
    echo "WARN [env_caches.sh]: \$SCRATCH is empty; cache redirection skipped." >&2
    return 0 2>/dev/null || exit 0
fi

_prmrl_cache_root="${SCRATCH}/prm-rl-caches"

# HuggingFace: HF_HOME governs hub/datasets/transformers caches in
# huggingface_hub >= 0.20. We also set the two more specific vars for
# older loaders that still honour them.
export HF_HOME="${HF_HOME:-${_prmrl_cache_root}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

# Triton compiles kernels JIT and caches them in ~/.triton by default.
# On Vista this is the second-biggest home-quota offender after HF.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${_prmrl_cache_root}/triton}"

# Generic XDG + pip + matplotlib fallbacks — cheap insurance against
# anything that still writes to $HOME/.cache without asking.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${_prmrl_cache_root}/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${_prmrl_cache_root}/pip}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${_prmrl_cache_root}/matplotlib}"

# Torch inductor / dynamo also cache under ~/.cache by default.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${_prmrl_cache_root}/torchinductor}"

mkdir -p \
    "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" \
    "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$MPLCONFIGDIR" \
    "$TORCHINDUCTOR_CACHE_DIR"

unset _prmrl_cache_root
