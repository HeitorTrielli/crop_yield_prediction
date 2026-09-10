#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/ADM/Desktop/Produtividade/crop_yield_prediction/sits_moco
PY=.venv/bin/python
NPY=/home/emap/sits_moco_data/npy
for yr in 2019-2020 2020-2021 2021-2022 2022-2023 2023-2024; do
  echo "======== ${yr} ========"
  "$PY" data_download/build_mapbiomas_solo_sidecars.py \
    --mode muni_mean \
    --npy-root "$NPY" \
    --solo-dir files/mapbiomas_solo \
    --shapefile-dir files/shapefiles_pr \
    --year-range "$yr" \
    --force
done
echo DONE
ls "$NPY/2020-2021/4100103/" | head
"$PY" -c "import numpy as np; a=np.load('$NPY/2020-2021/4100103/4100103_soil.npy'); print(a.shape, a[0])"
