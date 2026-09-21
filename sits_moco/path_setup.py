"""Ensure the sits_moco repo root (and eval/viz CLI folders) are on ``sys.path``.

Call ``ensure_repo_on_path()`` at the top of scripts under ``eval/``, ``viz/``,
or ``tools/`` so ``import datasets`` / sibling CLIs still work when launched as
``python eval/generate_results.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def ensure_repo_on_path(*, include_cli_dirs: bool = True) -> Path:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    if include_cli_dirs:
        for name in ("eval", "viz"):
            p = str(REPO_ROOT / name)
            if p not in sys.path:
                sys.path.insert(0, p)
    return REPO_ROOT
