#!/usr/bin/env bash
# Populate daily_tiff from raw_tiff (merge + municipal clip).
# Streaming merge is low-RAM; prefer Linux FS outputs + optional --stage-raw.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
LOG="${ROOT}/files/build_parana_daily_tiff.log"
MERGE_WORKERS="${MERGE_WORKERS:-6}"
CLIP_WORKERS="${CLIP_WORKERS:-${WORKERS:-8}}"
GDAL_THREADS="${GDAL_THREADS:-4}"
MOSAIC_ROOT="${MOSAIC_ROOT:-$HOME/sits_moco_data/state_tiff}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/sits_moco_data/daily_tiff}"

mkdir -p "${ROOT}/files" "$MOSAIC_ROOT" "$OUTPUT_DIR"
exec > >(tee -a "$LOG") 2>&1
echo "==== $(date -Iseconds) start build_parana_daily_tiff ===="
echo "ROOT=$ROOT MERGE_WORKERS=$MERGE_WORKERS CLIP_WORKERS=$CLIP_WORKERS GDAL_THREADS=$GDAL_THREADS"
echo "MOSAIC_ROOT=$MOSAIC_ROOT OUTPUT_DIR=$OUTPUT_DIR"
"$PY" scripts/build_parana_daily_tiff.py \
  --merge-workers "$MERGE_WORKERS" \
  --clip-workers "$CLIP_WORKERS" \
  --gdal-threads "$GDAL_THREADS" \
  --mosaic-root "$MOSAIC_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --fast \
  --stage-raw \
  "$@"
echo "==== $(date -Iseconds) done ===="
