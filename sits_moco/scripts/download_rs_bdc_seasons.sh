#!/usr/bin/env bash
# Download RS daily soy GeoTIFFs from BDC, one harvest season at a time.
# Season Y is Oct (Y-1)–Mar Y → files/raw_tiff/{Y-1}-{Y}/rs/
#
# BDC S2_L2A-1 has no daily COGs over RS before ~Oct 2021, so default
# years start at 2022 (Oct 2021–Mar 2022). 2020/2021 would write a soy
# mask then find 0 STAC items.
#
# One python process per year so a crash does not skip the rest.
#
# Light dates: 4 workers × 512 (full-mosaic profile ~303 s/date).
# Heavy dates (≥15 granules): 2 workers × 2048 (window s/Mpx was ~2.5× better
# than 512 on a 32-granule day; 4-wide statewide days saturate INPE).
#
# Usage (WSL, sits_moco venv, repo root):
#   bash scripts/download_rs_bdc_seasons.sh
#   bash scripts/download_rs_bdc_seasons.sh 2022 2023 2024
#   WORKERS=4 BLOCK_SIZE=512 bash scripts/download_rs_bdc_seasons.sh

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SHAPEFILE="${SHAPEFILE:-files/rs_state/rs.shp}"
WORKERS="${WORKERS:-4}"
BLOCK_SIZE="${BLOCK_SIZE:-512}"

if [[ $# -gt 0 ]]; then
  years=("$@")
else
  years=(2022 2023 2024)
fi

if [[ ! -f "$SHAPEFILE" ]]; then
  echo "Shapefile not found: $SHAPEFILE" >&2
  exit 1
fi

ok=()
fail=()

for year in "${years[@]}"; do
  echo
  echo "======== season-year ${year} (Oct $((year - 1))–Mar ${year}) workers=${WORKERS} block=${BLOCK_SIZE} ========"
  if python data_download/download_soy_bdc_daily_tiff.py \
    --shapefile "$SHAPEFILE" \
    --season-year "$year" \
    --skip-existing \
    --workers "$WORKERS" \
    --block-size "$BLOCK_SIZE"; then
    ok+=("$year")
  else
    status=$?
    echo "Season ${year} failed (exit ${status})" >&2
    fail+=("$year")
  fi
done

echo
echo "Done. ok: ${ok[*]:-none}  failed: ${fail[*]:-none}"
if [[ ${#fail[@]} -gt 0 ]]; then
  exit 1
fi
