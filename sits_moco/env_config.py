"""
Load local paths from .env (see .env.example).

Primary variable:
  SITS_MOCO_DATAPATH — root directory of municipal .npy files
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ENV_DATAPATH_KEY = "SITS_MOCO_DATAPATH"


@lru_cache(maxsize=1)
def load_dotenv() -> bool:
    """Load ``.env`` from the repo root if python-dotenv is installed."""
    env_file = REPO_ROOT / ".env"
    try:
        from dotenv import load_dotenv as _load_dotenv
    except ImportError:
        if env_file.is_file():
            _load_dotenv_file_manual(env_file)
            return True
        return False
    return bool(_load_dotenv(env_file))


def _load_dotenv_file_manual(env_file: Path) -> None:
    """Minimal KEY=VALUE parser when python-dotenv is not installed."""
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def resolve_datapath(override: str | Path | None = None) -> Path:
    """
    Resolve the NPY dataset root.

    Precedence: explicit ``override`` (CLI/YAML) > ``SITS_MOCO_DATAPATH`` from environment/.env.
    """
    if override is not None and str(override).strip() not in ("", "env"):
        return Path(override).expanduser().resolve()

    load_dotenv()
    raw = os.environ.get(ENV_DATAPATH_KEY)
    if not raw or not str(raw).strip():
        raise EnvironmentError(
            f"{ENV_DATAPATH_KEY} is not set. "
            f"Copy {REPO_ROOT / '.env.example'} to {REPO_ROOT / '.env'} "
            "and set your NPY root path."
        )
    path = Path(raw).expanduser().resolve()
    return path


def datapath_default_str() -> str:
    """String form for argparse help / logging."""
    return f"${ENV_DATAPATH_KEY} from .env"
