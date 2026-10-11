#!/usr/bin/env bash
# Build RS mega-pixel sidecars + GEE spectral enhancement for weak seasons.
#
# Order:
#   1) GEE chunked zonal Ã¢â€ â€™ files/zonal_mean_gee (seasons 2020, 2021, 2024)
#   2) Merge GEE into files/zonal_mean
#   3) Rebuild npy_muni_mean for RS
#   4) MapBiomas Solo RS clip + soil sidecars
#   5) Xavier RS climate NetCDFs + daily climate sidecars
#
# GEE account MUST be heigthtrielli@gmail.com (not heitortrielli@gmail.com).
#
# Usage (WSL, repo root, venv):
#   bash scripts/build_rs_sidecars_and_gee.sh
#   SKIP_GEE=1 bash scripts/build_rs_sidecars_and_gee.sh   # soil+climate only
#   SKIP_SOIL=1 SKIP_CLIMATE=1 bash scripts/build_rs_sidecars_and_gee.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/emap/.venvs/sits_moco/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
fi
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python3 || true)"
fi
if [[ -z "${PYTHON}" ]]; then
  echo "No python found" >&2
  exit 1
fi

GEE_ACCOUNT="${GEE_ACCOUNT:-heigthtrielli@gmail.com}"
GEE_PROJECT="${GEE_PROJECT:-crop-yield-pred-506711}"
NPY_OUT="${NPY_OUT:-$HOME/sits_moco_data/npy_muni_mean_soy}"
ZONAL_ROOT="${ZONAL_ROOT:-files/zonal_mean}"
ZONAL_GEE="${ZONAL_GEE:-files/zonal_mean_gee}"
SHAPEFILE_DIR="${SHAPEFILE_DIR:-files/shapefiles}"
SOLO_DIR="${SOLO_DIR:-$HOME/sits_moco_data/mapbiomas_solo_rs}"
CLIMATE_DIR="${CLIMATE_DIR:-files/climate}"
LOG="${LOG:-files/build_rs_sidecars_and_gee.log}"
SEASONS_SIDECAR=(2019-2020 2020-2021 2021-2022 2022-2023 2023-2024)
MERGE_SEASONS="2019-2020,2020-2021,2023-2024"
GEE_SEASON_YEARS="${GEE_SEASON_YEARS:-2020,2021,2024}"

SKIP_GEE="${SKIP_GEE:-0}"
SKIP_MERGE="${SKIP_MERGE:-0}"
SKIP_NPY="${SKIP_NPY:-0}"
SKIP_SOIL="${SKIP_SOIL:-0}"
SKIP_CLIMATE="${SKIP_CLIMATE:-0}"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -Is) build_rs_sidecars_and_gee ==="
echo "PYTHON=$PYTHON"
echo "GEE_ACCOUNT=$GEE_ACCOUNT"
echo "NPY_OUT=$NPY_OUT"

if [[ "$GEE_ACCOUNT" != "heigthtrielli@gmail.com" ]]; then
  echo "Refusing to run: GEE_ACCOUNT must be heigthtrielli@gmail.com (got $GEE_ACCOUNT)" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 1) GEE zonal (cheap reduceRegions path)
# ---------------------------------------------------------------------------
if [[ "$SKIP_GEE" != "1" ]]; then
  echo
  echo "=== 1/5 GEE zonal seasons ${GEE_SEASON_YEARS} Ã¢â€ â€™ ${ZONAL_GEE} ==="
  echo "Verifying GEE auth as ${GEE_ACCOUNT}..."
  "$PYTHON" - <<PY
from data_download.gee_export import initialize_gee
initialize_gee("${GEE_PROJECT}", "${GEE_ACCOUNT}")
print("GEE account OK")
PY

  "$PYTHON" data_download/download_soy_gee_zonal_mean.py \
    --shapefile-dir "$SHAPEFILE_DIR" \
    --season-years "$GEE_SEASON_YEARS" \
    --states rs \
    --output-dir "$ZONAL_GEE" \
    --skip-existing \
    --workers "${WORKERS:-10}" \
    --export-chunk-size "${EXPORT_CHUNK_SIZE:-40}" \
    --gee-project "$GEE_PROJECT" \
    --gee-account "$GEE_ACCOUNT"
else
  echo "=== 1/5 SKIP_GEE=1 ==="
fi

# ---------------------------------------------------------------------------
# 2) Merge GEE into primary zonal root
# ---------------------------------------------------------------------------
if [[ "$SKIP_MERGE" != "1" ]]; then
  echo
  echo "=== 2/5 merge ${ZONAL_GEE} Ã¢â€ â€™ ${ZONAL_ROOT} (${MERGE_SEASONS}) ==="
  if [[ -d "$ZONAL_GEE" ]]; then
    "$PYTHON" preprocessing/merge_zonal_csvs.py \
      --bdc-root "$ZONAL_ROOT" \
      --gee-root "$ZONAL_GEE" \
      --seasons "$MERGE_SEASONS" \
      --muni-prefix 43
  else
    echo "No $ZONAL_GEE yet; skip merge"
  fi
else
  echo "=== 2/5 SKIP_MERGE=1 ==="
fi

# ---------------------------------------------------------------------------
# 3) Rebuild RS mega-pixels
# ---------------------------------------------------------------------------
if [[ "$SKIP_NPY" != "1" ]]; then
  echo
  echo "=== 3/5 zonal CSV Ã¢â€ â€™ npy_muni_mean ==="
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
    --overwrite
else
  echo "=== 3/5 SKIP_NPY=1 ==="
fi

# ---------------------------------------------------------------------------
# 4) Soil
# ---------------------------------------------------------------------------
if [[ "$SKIP_SOIL" != "1" ]]; then
  echo
  echo "=== 4/5 MapBiomas Solo RS + soil sidecars ==="
  "$PYTHON" data_download/download_mapbiomas_solo.py \
    --shapefile files/rs_state/rs.shp \
    --output-dir "$SOLO_DIR" \
    --region-tag rs \
    --soc-years 2019 2020 2021 2022 2023 2024

  for yr in "${SEASONS_SIDECAR[@]}"; do
    echo "--- soil $yr ---"
    codes="$("$PYTHON" - <<PY
from pathlib import Path
root = Path("$NPY_OUT") / "$yr"
codes = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("43") and (p / f"{p.name}.npy").is_file()) if root.is_dir() else []
print(",".join(codes))
PY
)"
    if [[ -z "$codes" ]]; then
      echo "  no RS cubes in $NPY_OUT/$yr; skip"
      continue
    fi
    "$PYTHON" data_download/build_mapbiomas_solo_sidecars.py \
      --mode muni_mean \
      --npy-root "$NPY_OUT" \
      --shapefile-dir "$SHAPEFILE_DIR" \
      --solo-dir "$SOLO_DIR" \
      --region-tag rs \
      --year-range "$yr" \
      --codes "$codes" \
      -j "${SOIL_WORKERS:-4}"
  done
else
  echo "=== 4/5 SKIP_SOIL=1 ==="
fi

# ---------------------------------------------------------------------------
# 5) Climate
# ---------------------------------------------------------------------------
if [[ "$SKIP_CLIMATE" != "1" ]]; then
  echo
  echo "=== 5/5 Xavier RS climate NetCDFs + daily sidecars ==="
  "$PYTHON" data_download/download_xavier_state_climate.py \
    --states rs \
    --year-range 2019-2024 \
    --download \
    --skip-existing \
    --output-dir "$CLIMATE_DIR" \
    --nc-dir files/xavier_nc \
    --shapefile-dir "$SHAPEFILE_DIR"

  for yr in "${SEASONS_SIDECAR[@]}"; do
    echo "--- climate $yr ---"
    codes="$("$PYTHON" - <<PY
from pathlib import Path
root = Path("$NPY_OUT") / "$yr"
codes = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("43") and (p / f"{p.name}.npy").is_file()) if root.is_dir() else []
print(",".join(codes))
PY
)"
    if [[ -z "$codes" ]]; then
      echo "  no RS cubes in $NPY_OUT/$yr; skip"
      continue
    fi
    "$PYTHON" data_download/build_muni_daily_climate_npy.py \
      --npy-root "$NPY_OUT" \
      --year-range "$yr" \
      --shapefile-dir "$SHAPEFILE_DIR" \
      --codes "$codes" \
      --xavier-pr-nc "$CLIMATE_DIR/pr_RS_daily_2019_2024.nc" \
      --xavier-eto-nc "$CLIMATE_DIR/ETo_RS_daily_2019_2024.nc" \
      --xavier-rs-nc "$CLIMATE_DIR/Rs_RS_daily_2019_2024.nc" \
      --xavier-tmax-nc "$CLIMATE_DIR/Tmax_RS_daily_2019_2024.nc" \
      --xavier-tmin-nc "$CLIMATE_DIR/Tmin_RS_daily_2019_2024.nc"
  done
else
  echo "=== 5/5 SKIP_CLIMATE=1 ==="
fi

echo
echo "=== $(date -Is) DONE ==="
echo "Log: $LOG"
echo "Mega-pixels: $NPY_OUT"
