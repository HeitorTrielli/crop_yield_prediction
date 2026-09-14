# Source from WSL when the repo lives under /mnt/c/ (Windows mount).
# Keeps code + data on the mount, venv on the Linux filesystem.
#
#   source scripts/wsl_env.sh
#   python Calor_municipio_tiff.py
#   uv sync          # or: bash scripts/wsl_uv_sync.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this file instead of executing it:" >&2
  echo "  source scripts/wsl_env.sh" >&2
  exit 1
fi

_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$_ROOT"

export UV_LINK_MODE=copy
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/sits_moco}"

if [[ ! -f "$UV_PROJECT_ENVIRONMENT/bin/activate" ]]; then
  echo "Creating venv at $UV_PROJECT_ENVIRONMENT ..."
  mkdir -p "$(dirname "$UV_PROJECT_ENVIRONMENT")"
  uv venv "$UV_PROJECT_ENVIRONMENT"
  echo "Installing packages (first time may take a while)..."
  uv sync
fi

# shellcheck disable=SC1091
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

echo "venv: $UV_PROJECT_ENVIRONMENT"
echo "cwd:  $_ROOT"
unset _ROOT
