#!/usr/bin/env bash
set -euo pipefail
ROOT="/mnt/c/Users/ADM/Desktop/Produtividade/crop_yield_prediction/sits_moco"
SP="$ROOT/.venv/lib/python3.12/site-packages"
cd "$SP"
n=0
for d in *.dist-info; do
  if [ ! -f "$d/METADATA" ]; then
    echo "orphan: $d"
    rm -rf "$d"
    n=$((n + 1))
  fi
done
echo "removed $n orphaned dist-info dirs"
export VIRTUAL_ENV="$ROOT/.venv"
uv pip install --python "$ROOT/.venv/bin/python" --link-mode=copy \
  rasterio geopandas tqdm 2>&1 | tee /tmp/venv_fix.log | tail -40
"$ROOT/.venv/bin/python" - <<'PY'
for m in ("rasterio", "geopandas", "tqdm", "numpy", "polars"):
    try:
        mod = __import__(m)
        print("OK", m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print("FAIL", m, e)
PY
