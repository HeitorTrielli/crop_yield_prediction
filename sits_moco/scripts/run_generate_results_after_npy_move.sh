#!/usr/bin/env bash
# Wait until files/npy has been fully cut to ~/files/npy, then run generate_results.
#
# Usage (from repo root, WSL, venv active or not):
#   bash scripts/run_generate_results_after_npy_move.sh
#   bash scripts/run_generate_results_after_npy_move.sh --chunk-size 25000
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SRC_NPY="${SRC_NPY:-$REPO_ROOT/files/npy}"
DEST_NPY="${DEST_NPY:-$HOME/files/npy}"
CHECKPOINT="${CHECKPOINT:-results/tuning/productivity_top5_spectral_xavier_full/trial_003/training/model_best.pth}"
HOLDOUT_YEAR="${HOLDOUT_YEAR:-2021}"
CHUNK_SIZE="${CHUNK_SIZE:-20000}"
POLL_SECONDS="${POLL_SECONDS:-30}"

# Optional extra args forwarded to generate_results.py
EXTRA_ARGS=("$@")

count_npy() {
  local root="$1"
  if [[ ! -d "$root" ]]; then
    echo 0
    return
  fi
  find "$root" -type f -name '*.npy' 2>/dev/null | wc -l
}

echo "Repo:        $REPO_ROOT"
echo "Source npy:  $SRC_NPY"
echo "Dest npy:    $DEST_NPY"
echo "Checkpoint:  $CHECKPOINT"
echo "Holdout:     $HOLDOUT_YEAR"
echo "Chunk size:  $CHUNK_SIZE"
echo "Poll every:  ${POLL_SECONDS}s"
echo

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

echo "Waiting for npy cut to finish (no rsync, source empty, dest non-empty)..."
while true; do
  src_n="$(count_npy "$SRC_NPY")"
  dest_n="$(count_npy "$DEST_NPY")"
  rsync_n="$(pgrep -c rsync 2>/dev/null || true)"
  rsync_n="${rsync_n:-0}"

  printf '[%s] src_npy=%s  dest_npy=%s  rsync_procs=%s\n' \
    "$(date '+%H:%M:%S')" "$src_n" "$dest_n" "$rsync_n"

  if (( rsync_n == 0 && src_n == 0 && dest_n > 0 )); then
    echo "Move complete."
    break
  fi
  sleep "$POLL_SECONDS"
done

echo
du -sh "$DEST_NPY" || true
echo

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
fi

export SITS_MOCO_DATAPATH="$DEST_NPY"

echo "Starting generate_results..."
exec python generate_results.py \
  --checkpoint "$CHECKPOINT" \
  --holdout-year "$HOLDOUT_YEAR" \
  --datapath "$DEST_NPY" \
  --chunk-size "$CHUNK_SIZE" \
  "${EXTRA_ARGS[@]}"
