#!/usr/bin/env bash
ROOT=/mnt/c/Users/ADM/Desktop/Produtividade/crop_yield_prediction/sits_moco
echo "=== processes ==="
pgrep -af 'build_parana|merge_gee|clip_tiffs' || echo none
echo "=== log (tail) ==="
tail -20 "$ROOT/files/build_parana_daily_tiff.log" 2>/dev/null || echo no_log
echo "=== parana merged (maxdepth 1) ==="
for s in 2018-2019 2019-2020 2020-2021 2021-2022 2022-2023 2023-2024; do
  n=$(find "$ROOT/files/raw_tiff/$s/parana" -maxdepth 1 -name '*.tiff' 2>/dev/null | wc -l)
  echo "  $s: $n"
done
echo "=== daily_tiff ==="
if [[ -d "$ROOT/files/daily_tiff" ]]; then
  find "$ROOT/files/daily_tiff" -mindepth 1 -maxdepth 1 -type d | sort
  echo "muni folders: $(find "$ROOT/files/daily_tiff" -mindepth 2 -maxdepth 2 -type d | wc -l)"
  echo "tiff files: $(find "$ROOT/files/daily_tiff" -type f \( -name '*.tif' -o -name '*.tiff' \) | wc -l)"
else
  echo missing
fi
