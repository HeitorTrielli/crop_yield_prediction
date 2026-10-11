#!/usr/bin/env bash
# Link municipal daily climate sidecars into the pixel npy tree so DualSTNet
# can broadcast the same [T,5] climate to every soy pixel in the municipality.
set -euo pipefail

SEASON="${1:-2020-2021}"
PIX_ROOT="${PIX_ROOT:-/home/emap/sits_moco_data/npy}"
MEAN_ROOT="${MEAN_ROOT:-/home/emap/sits_moco_data/npy_muni_mean}"

PIX="$PIX_ROOT/$SEASON"
MEAN="$MEAN_ROOT/$SEASON"

if [[ ! -d "$PIX" || ! -d "$MEAN" ]]; then
  echo "Missing season dirs: PIX=$PIX MEAN=$MEAN" >&2
  exit 1
fi

linked=0
missing=0
for mdir in "$PIX"/*/; do
  code=$(basename "$mdir")
  src="$MEAN/$code/${code}_climate_daily.npy"
  dst="$PIX/$code/${code}_climate_daily.npy"
  if [[ ! -f "$src" ]]; then
    echo "missing climate for $code" >&2
    missing=$((missing + 1))
    continue
  fi
  if [[ -e "$dst" || -L "$dst" ]]; then
    continue
  fi
  ln -s "$src" "$dst"
  linked=$((linked + 1))
done

echo "season=$SEASON linked=$linked already_present_or_skipped=$(( $(find "$PIX" -name '*_climate_daily.npy' | wc -l) )) missing_src=$missing"
