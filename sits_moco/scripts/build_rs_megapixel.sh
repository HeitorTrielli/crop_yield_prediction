#!/usr/bin/env bash
# Build RS mega-pixels the Paraná way:
#   BDC state daily TIFFs (already GEE-harmonized) → zonal CSV → [1,T,C] .npy
#
# Seasons with local RS TIFFs: 2021-2022, 2022-2023, 2023-2024
# 2019-2020: no BDC daily COGs; reuse existing files/zonal_mean RS CSVs
#
# Usage (WSL, repo root, repaired venv):
#   bash scripts/build_rs_megapixel.sh
#   bash scripts/build_rs_megapixel.sh 2022-2023
#   WORKERS=8 MUNI_WORKERS=8 bash scripts/build_rs_megapixel.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3 || true)"
fi
if [[ -z "${PYTHON}" ]]; then
  echo "No python found (tried .venv/bin/python and python3)" >&2
  exit 1
fi
echo "Using Python: $PYTHON"

WORKERS="${WORKERS:-12}"
MUNI_WORKERS="${MUNI_WORKERS:-1}"
SHAPEFILE_DIR="${SHAPEFILE_DIR:-files/shapefiles}"
ZONAL_ROOT="${ZONAL_ROOT:-files/zonal_mean}"
NPY_OUT="${NPY_OUT:-$HOME/sits_moco_data/npy_muni_mean}"
PREFIX="${PREFIX:-43}"

if [[ $# -gt 0 ]]; then
  seasons=("$@")
else
  seasons=(2021-2022 2022-2023 2023-2024)
fi

ok=()
fail=()

for season in "${seasons[@]}"; do
  tiff_dir="files/raw_tiff/${season}/rs"
  echo
  echo "======== RS ${season}: TIFF → zonal CSV (dates=${WORKERS} munis=${MUNI_WORKERS}) ========"
  if [[ ! -d "$tiff_dir" ]]; then
    echo "Missing $tiff_dir" >&2
    fail+=("$season")
    continue
  fi
  n_tiff=$(find "$tiff_dir" -maxdepth 1 \( -name '*.tiff' -o -name '*.tif' \) ! -name '_*' | wc -l)
  if [[ "$n_tiff" -eq 0 ]]; then
    echo "No daily TIFFs in $tiff_dir (BDC often empty before Oct 2021). Skip." >&2
    fail+=("$season")
    continue
  fi
  if "$PYTHON" preprocessing/state_tiff_to_zonal_csv.py \
    --tiff-dir "$tiff_dir" \
    --shapefile-dir "$SHAPEFILE_DIR" \
    --muni-prefix "$PREFIX" \
    --output-root "$ZONAL_ROOT" \
    --season "$season" \
    --overwrite-csvs \
    --skip-existing \
    -j "$WORKERS" \
    --muni-workers "$MUNI_WORKERS"; then
    ok+=("$season")
  else
    fail+=("$season")
  fi
done

echo
echo "======== zonal CSV → mega-pixel .npy → ${NPY_OUT} ========"
# Convert all RS zonal CSVs present (includes 2019-2020 if already on disk)
"$PYTHON" - <<'PY'
from pathlib import Path
import csv
root = Path("files/zonal_mean")
out = Path("files/zonal_mean_rs_keep.csv")
rows = []
for season_dir in sorted(root.iterdir()):
    if not season_dir.is_dir() or "-" not in season_dir.name:
        continue
    y2 = season_dir.name.split("-")[1]
    if not y2.isdigit():
        continue
    harvest = int(y2)
    for csv_path in season_dir.glob("43*/43*_zonal.csv"):
        code = csv_path.parent.name
        rows.append({"municipality_code": code, "year": harvest})
# unique
seen = set()
uniq = []
for r in rows:
    k = (r["municipality_code"], r["year"])
    if k not in seen:
        seen.add(k)
        uniq.append(r)
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["municipality_code", "year"])
    w.writeheader()
    w.writerows(sorted(uniq, key=lambda r: (r["year"], r["municipality_code"])))
print(f"Wrote {out} with {len(uniq)} municipality-years")
PY

"$PYTHON" preprocessing/zonal_csv_to_muni_npy.py \
  --zonal-root "$ZONAL_ROOT" \
  --keep-list files/zonal_mean_rs_keep.csv \
  --output-dir "$NPY_OUT" \
  --skip-existing

echo
echo "Done. TIFF→CSV ok: ${ok[*]:-none}  failed: ${fail[*]:-none}"
echo "Mega-pixels: $NPY_OUT"
if [[ ${#fail[@]} -gt 0 ]]; then
  exit 1
fi
