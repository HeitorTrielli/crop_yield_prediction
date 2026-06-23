#!/usr/bin/env bash
# Copy .npy data from Windows mount to native WSL ext4 for faster mmap reads.
# Usage (from WSL): bash scripts/sync_npy_to_wsl.sh
#   or: bash scripts/sync_npy_to_wsl.sh [/mnt/c/.../files/npy] [~/sits_moco_data/npy]
set -euo pipefail
SRC="${1:-/mnt/c/Users/ADM/Desktop/Produtividade/crop_yield_prediction/sits_moco/files/npy}"
DST="${2:-$HOME/sits_moco_data/npy}"
echo "Syncing $SRC -> $DST"
mkdir -p "$DST"
rsync -a --info=progress2 "$SRC/" "$DST/"
echo "Done. Train with: --datapath $DST"
