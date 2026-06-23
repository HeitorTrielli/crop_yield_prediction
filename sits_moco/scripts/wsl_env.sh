# Source from WSL when the repo lives under /mnt/c/ (Windows mount).
# Keeps code + data on the mount, venv on the Linux filesystem.
#
#   source scripts/wsl_env.sh
#   uv sync          # or: bash scripts/wsl_uv_sync.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this file instead of executing it:" >&2
  echo "  source scripts/wsl_env.sh" >&2
  exit 1
fi

export UV_LINK_MODE=copy
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/sits_moco}"
