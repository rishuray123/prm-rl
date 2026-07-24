#!/bin/bash
# slurm/fix_home_quota.sh — Move existing home-directory caches to
# $SCRATCH and symlink them back, freeing up /home1 quota without
# losing already-downloaded HuggingFace weights or Triton kernels.
#
# One-shot. Safe to re-run — is idempotent (existing symlinks are
# treated as already-migrated and skipped).
#
# Usage (from a login node or an idev shell, venv NOT required):
#
#     bash slurm/fix_home_quota.sh
#
# After this completes, source slurm/env_caches.sh in the current
# shell (or start a fresh shell that will source it via _common.sh)
# so newly-written cache entries go to $SCRATCH directly instead of
# through the symlink.
#
# See docs/knowledge-base.md §2.9 for background.

set -euo pipefail

if [[ -z "${SCRATCH:-}" ]]; then
    echo "ERROR: \$SCRATCH is not set. Run this on a TACC Vista node." >&2
    exit 1
fi

DEST_ROOT="${SCRATCH}/prm-rl-caches/home-mirror"
mkdir -p "$DEST_ROOT"

# Pairs of (source_in_home, target_in_scratch).
declare -a PAIRS=(
    "$HOME/.cache/huggingface|${DEST_ROOT}/huggingface"
    "$HOME/.cache/pip|${DEST_ROOT}/pip"
    "$HOME/.cache/torch|${DEST_ROOT}/torch"
    "$HOME/.cache/matplotlib|${DEST_ROOT}/matplotlib"
    "$HOME/.triton|${DEST_ROOT}/triton"
    "$HOME/.cache/wandb|${DEST_ROOT}/wandb"
)

migrate_one() {
    local src="$1"
    local dst="$2"

    if [[ -L "$src" ]]; then
        echo "  [skip] $src is already a symlink to $(readlink "$src")"
        return 0
    fi

    if [[ ! -e "$src" ]]; then
        echo "  [skip] $src does not exist"
        # Still create the dest + symlink so future writes land in $SCRATCH.
        mkdir -p "$dst"
        mkdir -p "$(dirname "$src")"
        ln -sfn "$dst" "$src"
        echo "  [link] $src -> $dst (created empty)"
        return 0
    fi

    echo "  [move] $src (size: $(du -sh "$src" 2>/dev/null | awk '{print $1}')) -> $dst"
    mkdir -p "$dst"
    # rsync preserves timestamps and permissions; --remove-source-files
    # deletes source files as they land safely.
    rsync -aH --remove-source-files "$src/" "$dst/"
    # rsync leaves empty dirs; wipe them.
    find "$src" -type d -empty -delete 2>/dev/null || true
    rm -rf "$src"
    ln -sfn "$dst" "$src"
    echo "  [ok]   $src -> $dst"
}

echo "Migrating home-directory caches to $DEST_ROOT ..."
for pair in "${PAIRS[@]}"; do
    src="${pair%|*}"
    dst="${pair#*|}"
    migrate_one "$src" "$dst"
done
echo

echo "Home quota after migration:"
if command -v /usr/local/etc/taccinfo >/dev/null 2>&1; then
    /usr/local/etc/taccinfo | grep -i "home\|quota" || true
fi
du -sh "$HOME"/.cache "$HOME"/.triton 2>/dev/null | head -20 || true
echo
echo "Done. Now source the env-vars so future writes go directly to \$SCRATCH:"
echo "  source $(dirname "$0")/env_caches.sh"
