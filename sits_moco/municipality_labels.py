"""Resolve IBGE municipality codes to human-readable names for plots and reports."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_YIELD_CSV = _REPO_ROOT / "files" / "pam_soy_pr_2019_2025.csv"


def normalize_municipality_code(code: str | int) -> str:
    return str(code).replace(".0", "", 1).strip().zfill(7)


def clean_municipality_display_name(name: str) -> str:
    """e.g. 'Guarapuava (PR)' -> 'Guarapuava'."""
    text = str(name).strip()
    text = re.sub(r"\s*\([A-Z]{2}\)\s*$", "", text)
    return text


def slugify_municipality_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", name)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text.lower().strip())
    text = re.sub(r"[-\s]+", "_", text)
    return text or "municipio"


@lru_cache(maxsize=1)
def _yield_csv_name_map(yield_csv: str) -> dict[str, str]:
    path = Path(yield_csv)
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    code_col = None
    for candidate in ("municipality_code", "geo_cod", "cod_municipio_ibge"):
        if candidate in df.columns:
            code_col = candidate
            break
    if code_col is None:
        return {}
    name_col = None
    for candidate in ("municipality_name", "nome_municipio", "name_muni"):
        if candidate in df.columns:
            name_col = candidate
            break
    if name_col is None:
        return {}
    out: dict[str, str] = {}
    for code, name in zip(df[code_col], df[name_col]):
        if pd.isna(name):
            continue
        key = normalize_municipality_code(code)
        out[key] = clean_municipality_display_name(str(name))
    return out


def _name_from_tiff_root(tiff_root: Path, code: str) -> str | None:
    if not tiff_root.is_dir():
        return None
    prefix = f"{code}_"
    for dir_path in sorted(tiff_root.iterdir()):
        if dir_path.is_dir() and dir_path.name.startswith(prefix):
            return clean_municipality_display_name(dir_path.name[len(prefix) :])
    return None


def resolve_municipality_display_name(
    code: str | int,
    *,
    yield_csv: Path | str | None = None,
    tiff_root: Path | str | None = None,
) -> str:
    """Return display name for an IBGE municipality code (fallback: the code)."""
    norm = normalize_municipality_code(code)
    csv_path = Path(yield_csv) if yield_csv is not None else DEFAULT_YIELD_CSV
    name = _yield_csv_name_map(str(csv_path.resolve())).get(norm)
    if name:
        return name
    if tiff_root is not None:
        from_tiff = _name_from_tiff_root(Path(tiff_root), norm)
        if from_tiff:
            return from_tiff
    return norm


def municipality_output_stem(
    code: str | int,
    *,
    yield_csv: Path | str | None = None,
    tiff_root: Path | str | None = None,
) -> str:
    """Filesystem-safe stem from municipality name (e.g. guarapuava)."""
    display = resolve_municipality_display_name(
        code, yield_csv=yield_csv, tiff_root=tiff_root
    )
    norm = normalize_municipality_code(code)
    if display == norm:
        return norm
    return slugify_municipality_name(display)
