#!/usr/bin/env bash
# Copy .npy data from Windows mount to native WSL ext4 for faster mmap reads.
# Usage (from WSL): bash scripts/sync_npy_to_wsl.sh
#   or: bash scripts/sync_npy_to_wsl.sh [/mnt/c/.../files/npy] [~/sits_moco_data/npy]
set -euo pipefail
SRC="${1:-/mnt/c/Users/ADM/Desktop/Produtividade/crop_yield_prediction/sits_moco/files/npy}"
DST="${2:-$HOME/sits_moco_data/npy}"
echo "Syncing $SRC -> $DST"

# A symlink into /mnt/c makes rsync a no-op (src == dest). Replace with a real dir.
if [[ -L "$DST" ]]; then
  echo "Removing symlink $DST -> $(readlink "$DST")"
  rm -f "$DST"
fi
mkdir -p "$DST"
if [[ "$(readlink -f "$SRC")" == "$(readlink -f "$DST")" ]]; then
  echo "ERROR: SRC and DST resolve to the same path. Aborting." >&2
  exit 1
fi

rsync -a --info=progress2 "$SRC/" "$DST/"
echo "Done. Train with: --datapath $DST"
df -Th "$DST"
