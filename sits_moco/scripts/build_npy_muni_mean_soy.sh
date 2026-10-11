#!/usr/bin/env bash
# Materialize ~/sits_moco_data/npy_muni_mean_soy as the DualSTNet default tree:
#   - hardlink (or copy) spectral {code}.npy from npy_muni_mean
#   - rebuild soy-masked climate + soil sidecars
#
# Legacy unmasked tree (npy_muni_mean) is left untouched.
#
# Usage (WSL, repo root):
#   bash scripts/build_npy_muni_mean_soy.sh
#   FORCE=1 bash scripts/build_npy_muni_mean_soy.sh   # rewrite sidecars
#   SKIP_LINK=1 SKIP_CLIMATE=1 bash scripts/build_npy_muni_mean_soy.sh  # soil only

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

SRC="${SRC:-$HOME/sits_moco_data/npy_muni_mean}"
DST="${DST:-$HOME/sits_moco_data/npy_muni_mean_soy}"
CLIMATE_DIR="${CLIMATE_DIR:-$ROOT/files/climate}"
SHAPEFILE_DIR="${SHAPEFILE_DIR:-$ROOT/files/shapefiles}"
SOLO_PR="${SOLO_PR:-$ROOT/files/mapbiomas_solo}"
SOLO_RS="${SOLO_RS:-$HOME/sits_moco_data/mapbiomas_solo_rs}"
SEASONS=(2019-2020 2020-2021 2021-2022 2022-2023 2023-2024)
LOG="${LOG:-$ROOT/files/build_npy_muni_mean_soy.log}"
FORCE="${FORCE:-0}"
SKIP_LINK="${SKIP_LINK:-0}"
SKIP_CLIMATE="${SKIP_CLIMATE:-0}"
SKIP_SOIL="${SKIP_SOIL:-0}"

FORCE_FLAG=()
if [[ "$FORCE" == "1" ]]; then
  FORCE_FLAG=(--force)
fi

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -Is) build_npy_muni_mean_soy ==="
echo "PYTHON=$PYTHON"
echo "SRC=$SRC"
echo "DST=$DST"

link_or_copy() {
  local src="$1" dst="$2"
  if [[ -e "$dst" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "$dst")"
  if ln "$src" "$dst" 2>/dev/null; then
    return 0
  fi
  if ln -s "$src" "$dst" 2>/dev/null; then
    return 0
  fi
  cp -a "$src" "$dst"
}

codes_csv_for_prefix() {
  local season="$1" prefix="$2"
  local d="$DST/$season"
  [[ -d "$d" ]] || { echo ""; return 0; }
  "$PYTHON" - <<PY
from pathlib import Path
root = Path("$d")
codes = sorted(
    p.name for p in root.iterdir()
    if p.is_dir() and p.name.startswith("$prefix") and (p / f"{p.name}.npy").is_file()
)
print(",".join(codes))
PY
}

# ---------------------------------------------------------------------------
# 1) Hardlink / copy spectral cubes
# ---------------------------------------------------------------------------
if [[ "$SKIP_LINK" != "1" ]]; then
  echo
  echo "=== 1/3 link spectral {code}.npy → $DST ==="
  n_link=0
  for season in "${SEASONS[@]}"; do
    src_season="$SRC/$season"
    [[ -d "$src_season" ]] || continue
    for muni in "$src_season"/*/; do
      [[ -d "$muni" ]] || continue
      code="$(basename "$muni")"
      src_npy="$muni${code}.npy"
      [[ -f "$src_npy" ]] || continue
      dst_npy="$DST/$season/$code/${code}.npy"
      link_or_copy "$src_npy" "$dst_npy"
      n_link=$((n_link + 1))
    done
  done
  echo "linked/copied spectral cubes: $n_link"
else
  echo "=== 1/3 SKIP_LINK=1 ==="
fi

# ---------------------------------------------------------------------------
# 2) Soy-masked climate (PR + RS Xavier NCs)
# ---------------------------------------------------------------------------
if [[ "$SKIP_CLIMATE" != "1" ]]; then
  echo
  echo "=== 2/3 soy-masked climate sidecars ==="
  for season in "${SEASONS[@]}"; do
    for uf in PR RS; do
      if [[ "$uf" == "PR" ]]; then
        prefix=41
        pr_nc="$CLIMATE_DIR/pr_PR_daily_2020_2024.nc"
        eto_nc="$CLIMATE_DIR/ETo_PR_daily_2020_2024.nc"
        rs_nc="$CLIMATE_DIR/Rs_PR_daily_2020_2024.nc"
        tmax_nc="$CLIMATE_DIR/Tmax_PR_daily_2020_2024.nc"
        tmin_nc="$CLIMATE_DIR/Tmin_PR_daily_2020_2024.nc"
      else
        prefix=43
        pr_nc="$CLIMATE_DIR/pr_RS_daily_2019_2024.nc"
        eto_nc="$CLIMATE_DIR/ETo_RS_daily_2019_2024.nc"
        rs_nc="$CLIMATE_DIR/Rs_RS_daily_2019_2024.nc"
        tmax_nc="$CLIMATE_DIR/Tmax_RS_daily_2019_2024.nc"
        tmin_nc="$CLIMATE_DIR/Tmin_RS_daily_2019_2024.nc"
      fi
      codes="$(codes_csv_for_prefix "$season" "$prefix")"
      [[ -n "$codes" ]] || { echo "--- climate $season $uf: no codes ---"; continue; }
      for f in "$pr_nc" "$eto_nc" "$rs_nc" "$tmax_nc" "$tmin_nc"; do
        if [[ ! -f "$f" ]]; then
          echo "Missing NC $f — skip $season $uf" >&2
          continue 2
        fi
      done
      echo "--- climate $season $uf ---"
      "$PYTHON" data_download/build_muni_daily_climate_npy.py \
        --npy-root "$DST" \
        --year-range "$season" \
        --shapefile-dir "$SHAPEFILE_DIR" \
        --codes "$codes" \
        --soy-mask \
        --xavier-pr-nc "$pr_nc" \
        --xavier-eto-nc "$eto_nc" \
        --xavier-rs-nc "$rs_nc" \
        --xavier-tmax-nc "$tmax_nc" \
        --xavier-tmin-nc "$tmin_nc" \
        "${FORCE_FLAG[@]}"
    done
  done
else
  echo "=== 2/3 SKIP_CLIMATE=1 ==="
fi

# ---------------------------------------------------------------------------
# 3) Soy-masked soil (PR + RS Solo products)
# ---------------------------------------------------------------------------
if [[ "$SKIP_SOIL" != "1" ]]; then
  echo
  echo "=== 3/3 soy-masked soil sidecars ==="
  for season in "${SEASONS[@]}"; do
    for uf in PR RS; do
      if [[ "$uf" == "PR" ]]; then
        prefix=41
        solo_dir="$SOLO_PR"
        tag=pr
      else
        prefix=43
        solo_dir="$SOLO_RS"
        tag=rs
      fi
      codes="$(codes_csv_for_prefix "$season" "$prefix")"
      [[ -n "$codes" ]] || { echo "--- soil $season $uf: no codes ---"; continue; }
      if [[ ! -d "$solo_dir" ]]; then
        echo "Missing solo dir $solo_dir — skip $season $uf" >&2
        continue
      fi
      echo "--- soil $season $uf ---"
      "$PYTHON" data_download/build_mapbiomas_solo_sidecars.py \
        --mode muni_mean \
        --npy-root "$DST" \
        --year-range "$season" \
        --shapefile-dir "$SHAPEFILE_DIR" \
        --solo-dir "$solo_dir" \
        --region-tag "$tag" \
        --codes "$codes" \
        --soy-mask \
        -j 2 \
        "${FORCE_FLAG[@]}"
    done
  done
else
  echo "=== 3/3 SKIP_SOIL=1 ==="
fi

echo
echo "=== verification sample ==="
"$PYTHON" - <<'PY'
import numpy as np
from pathlib import Path
dst = Path.home() / "sits_moco_data" / "npy_muni_mean_soy"
src = Path.home() / "sits_moco_data" / "npy_muni_mean"
for season in ["2019-2020", "2021-2022", "2023-2024"]:
    for code in ["4100103", "4300034"]:
        d = dst / season / code
        if not (d / f"{code}.npy").is_file():
            continue
        soil = d / f"{code}_soil.npy"
        clim = d / f"{code}_climate_daily.npy"
        print(f"{season}/{code}: npy=yes soil={soil.is_file()} climate={clim.is_file()}", end="")
        if soil.is_file() and clim.is_file():
            s = np.load(soil)
            c = np.load(clim)
            print(f" soil={s.shape} climate={c.shape}", end="")
            old_s = src / season / code / f"{code}_soil.npy"
            old_c = src / season / code / f"{code}_climate_daily.npy"
            if old_s.is_file() and old_c.is_file():
                os_ = np.load(old_s)
                oc = np.load(old_c)
                sd = float(np.nanmean(np.abs(s.astype(float) - os_.astype(float))))
                cd = float(np.nanmean(np.abs(c.astype(float) - oc.astype(float))))
                print(f" |Δsoil|={sd:.4g} |Δclimate|={cd:.4g}", end="")
        print()
PY

echo "=== $(date -Is) build_npy_muni_mean_soy DONE ==="
echo "DualSTNet datapath default: $DST"
