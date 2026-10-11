#!/usr/bin/env bash
# Build npy_muni_mean_soy + soy-masked climate (+ soil) sidecars for DualSTNet.
# Legacy unmasked archive remains at npy_muni_mean; DualSTNet default is soy.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate

NPY_PIXEL="${NPY_PIXEL:-/home/heitort/files/npy}"
NPY_MEAN="${NPY_MEAN:-$HOME/sits_moco_data/npy_muni_mean_soy}"
NPY_MEAN_LEGACY="${NPY_MEAN_LEGACY:-$HOME/sits_moco_data/npy_muni_mean}"
CLIMATE_DIR="${CLIMATE_DIR:-$REPO/files/climate}"
SHAPE_DIR="${SHAPE_DIR:-$REPO/files/shapefiles}"
SOLO_DIR="${SOLO_DIR:-$REPO/files/mapbiomas_solo}"
SEASONS=(2019-2020 2020-2021 2021-2022 2022-2023 2023-2024)
WORKERS="${WORKERS:-6}"
LOG="${LOG:-$REPO/files/build_muni_mean_soy.log}"

exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -Is) build muni mean (soy-masked DualSTNet default) ==="
echo "REPO=$REPO"
echo "NPY_PIXEL=$NPY_PIXEL"
echo "NPY_MEAN=$NPY_MEAN"

echo
echo "=== 1/3 Sentinel municipal mean ==="
python preprocessing/aggregate_municipal_npy.py \
  --stat mean \
  --input-dir "$NPY_PIXEL" \
  --output-dir "$NPY_MEAN" \
  -j "$WORKERS"

echo
echo "=== 2/3 daily climate sidecars (soy-masked) ==="
for yr in "${SEASONS[@]}"; do
  echo "--- climate $yr ---"
  python data_download/build_muni_daily_climate_npy.py \
    --npy-root "$NPY_MEAN" \
    --year-range "$yr" \
    --shapefile-dir "$SHAPE_DIR" \
    --soy-mask \
    --xavier-pr-nc "$CLIMATE_DIR/pr_PR_daily_2020_2024.nc" \
    --xavier-eto-nc "$CLIMATE_DIR/ETo_PR_daily_2020_2024.nc" \
    --xavier-rs-nc "$CLIMATE_DIR/Rs_PR_daily_2020_2024.nc" \
    --xavier-tmax-nc "$CLIMATE_DIR/Tmax_PR_daily_2020_2024.nc"
  # Tmin densified from cumulative channel in the .npy when --xavier-tmin-nc omitted
done

echo
echo "=== 3/3 MapBiomas soil muni_mean sidecars (soy-masked) ==="
for yr in "${SEASONS[@]}"; do
  echo "--- soil $yr ---"
  python data_download/build_mapbiomas_solo_sidecars.py \
    --mode muni_mean \
    --npy-root "$NPY_MEAN" \
    --year-range "$yr" \
    --shapefile-dir "$SHAPE_DIR" \
    --solo-dir "$SOLO_DIR" \
    --soy-mask \
    -j 2
done

echo "=== $(date -Is) DONE ==="
echo "Point DualSTNet / generate_results at: --datapath $NPY_MEAN"
echo "Legacy unmasked tree (if kept): $NPY_MEAN_LEGACY"
