#!/usr/bin/env bash
# Install deps with uv on WSL when the repo is on a Windows mount (/mnt/c/...).
# uv cannot reliably write to .venv on drvfs; use a Linux-native venv instead.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck source=wsl_env.sh
source "$ROOT/scripts/wsl_env.sh"

if [[ "$ROOT" == /mnt/* ]]; then
  echo "WSL + Windows mount: venv -> $UV_PROJECT_ENVIRONMENT"
else
  echo "Using venv -> $UV_PROJECT_ENVIRONMENT"
fi

uv sync "$@"

echo
echo "Done. Activate with:"
echo "  source \"$UV_PROJECT_ENVIRONMENT/bin/activate\""
