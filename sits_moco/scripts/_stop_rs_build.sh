#!/usr/bin/env bash
set -euo pipefail
echo "Before:"
ps -ef | grep -E 'state_tiff|build_rs' | grep -v grep || echo "(none)"
for pat in state_tiff_to_zonal build_rs_megapixel; do
  pids=$(pgrep -f "$pat" || true)
  if [[ -n "${pids}" ]]; then
    echo "Killing $pat: $pids"
    kill -9 $pids || true
  fi
done
sleep 2
echo "After:"
ps -ef | grep -E 'state_tiff|build_rs' | grep -v grep || echo ALL_STOPPED
