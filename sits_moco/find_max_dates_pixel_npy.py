#!/usr/bin/env python3
"""
Summarize time dimension in daily municipal .npy files under files/npy.

Layout: {root}/{season}/{municipality}/{municipality}.npy
Array shape: [num_pixels, num_days, 11] — num_days is the same for every pixel in
that file (one row per stacked observation date).

For each season (folder name YYYY-YYYY), reports:
  • max_T — largest num_days among all municipality files (upper bound on sequence
    length before padding/subsampling for that season).
  • max_valid_days — largest count of timesteps where a pixel has any valid spectral
    band (not -9999 / 0 / NaN in bands 0–9). Always ≤ max_T for that season.

Use this to choose --sequencelength (≥ max_T over the seasons you train on, or your
subsampling cap).

Usage:
  python find_max_dates_pixel_npy.py --root files/npy
  python find_max_dates_pixel_npy.py --root files/npy/2022-2023
  python find_max_dates_pixel_npy.py --root files/npy --exclude 2020-2021
  python find_max_dates_pixel_npy.py --root files/npy --exclude 2020-2021 2021-2022
  python find_max_dates_pixel_npy.py --root files/npy --exclude 2020-2021,2021-2022
  python find_max_dates_pixel_npy.py --root files/npy --no-progress
"""
from __future__ import annotations

import argparse
import re
from typing import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

NO_DATA_VALUE = -9999
SEASON_DIR_RE = re.compile(r"^\d{4}-\d{4}$")


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def _parse_exclude_parts(parts: list[str]) -> set[str]:
    """Split comma-separated tokens; season folder names must match exactly (e.g. 2020-2021)."""
    out: set[str] = set()
    for raw in parts:
        for token in raw.split(","):
            s = token.strip()
            if s:
                out.add(s)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path("files/npy"),
        help="Dataset root (several YYYY-YYYY dirs) or one season folder",
    )
    p.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        metavar="SEASON",
        help=(
            "Season folder names under --root to skip (exact names, e.g. 2020-2021). "
            "Pass multiple tokens and/or comma-separated: --exclude A B or --exclude A,B"
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help="If > 0, only scan first N .npy files per season (debug)",
    )
    p.add_argument(
        "--chunk",
        type=int,
        default=4096,
        help="Rows per chunk when scanning pixels (default: 4096)",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    return p.parse_args()


def max_valid_days_and_row_mmap(
    data: np.ndarray, chunk: int
) -> tuple[int, int]:
    """Returns (max valid-day count, row index) for one file."""
    if data.ndim != 3 or data.shape[2] < 10:
        raise ValueError(f"Expected (N, T, >=10), got {data.shape}")
    n = data.shape[0]
    best_c, best_i = -1, -1
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        spec = np.array(data[start:end, :, :10], dtype=np.float32, copy=False)
        valid = (spec != NO_DATA_VALUE) & (spec != 0) & ~np.isnan(spec)
        day_ok = np.any(valid, axis=2)
        counts = np.sum(day_ok, axis=1)
        j = int(np.argmax(counts))
        c = int(counts[j])
        if c > best_c:
            best_c, best_i = c, start + j
    return best_c, best_i


@dataclass
class SeasonStats:
    season: str
    n_files: int
    max_T: int
    max_valid_days: int
    best_T_path: Path | None
    best_valid_path: Path | None
    best_valid_row: int


def scan_season(
    season_dir: Path,
    limit: int,
    chunk: int,
    *,
    show_progress: bool,
    nested: bool,
) -> SeasonStats:
    paths = sorted(season_dir.rglob("*.npy"))
    if limit > 0:
        paths = paths[:limit]

    season_name = season_dir.name
    max_T = 0
    max_valid = 0
    best_T_path: Path | None = None
    best_valid_path: Path | None = None
    best_valid_row = -1

    file_iter = paths
    if show_progress:
        file_iter = tqdm(
            paths,
            desc=season_name,
            unit="file",
            leave=not nested,
            dynamic_ncols=True,
            position=1 if nested else 0,
        )

    for p in file_iter:
        data = np.load(p, mmap_mode="r")
        if data.ndim != 3 or data.shape[0] == 0:
            continue
        t = int(data.shape[1])
        if t > max_T:
            max_T = t
            best_T_path = p
        v, row = max_valid_days_and_row_mmap(data, chunk)
        if v > max_valid:
            max_valid = v
            best_valid_path = p
            best_valid_row = row

    return SeasonStats(
        season=season_name,
        n_files=len(paths),
        max_T=max_T,
        max_valid_days=max_valid,
        best_T_path=best_T_path,
        best_valid_path=best_valid_path,
        best_valid_row=best_valid_row,
    )


def season_directories(root: Path) -> list[Path]:
    if SEASON_DIR_RE.match(root.name):
        return [root.resolve()]
    out = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and SEASON_DIR_RE.match(child.name):
            out.append(child.resolve())
    return out


def rel(p: Path, base: Path) -> str:
    try:
        return str(p.relative_to(base))
    except ValueError:
        return str(p)


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    seasons = season_directories(root)
    if not seasons:
        raise SystemExit(
            f"No season folders (names like 2020-2021) under {root}. "
            f"Point --root at files/npy or at one season folder."
        )

    _log(f"Root: {root}")

    exclude = _parse_exclude_parts(args.exclude)
    if exclude:
        before = [s.name for s in seasons]
        seasons = [s for s in seasons if s.name not in exclude]
        skipped = sorted(set(before) & exclude)
        if skipped:
            _log(f"Skipping season folders: {skipped}")
        unknown = sorted(exclude - set(before))
        if unknown:
            _log(
                f"⚠️  --exclude names not found under root (ignored): {unknown}"
            )
    if not seasons:
        raise SystemExit(
            "No seasons left to scan (all excluded, or --root points only at an excluded folder)."
        )

    print()
    rows: list[SeasonStats] = []
    show_p = not args.no_progress
    multi = len(seasons) > 1
    season_iter: Iterable[Path] = seasons
    if show_p and multi:
        season_iter = tqdm(
            seasons,
            desc="Seasons",
            unit="season",
            dynamic_ncols=True,
            position=0,
            leave=True,
        )

    for sd in season_iter:
        if not (show_p and multi):
            _log(f"Scanning season {sd.name}...")
        st = scan_season(
            sd,
            args.limit,
            args.chunk,
            show_progress=show_p,
            nested=show_p and multi,
        )
        rows.append(st)

    # Table
    hdr = f"{'Season':<14} {'files':>6} {'max_T':>7} {'max_valid':>10}  (T = timesteps in .npy; use max_T for --sequencelength cap)"
    _log(hdr)
    _log("-" * len(hdr))
    for st in rows:
        _log(
            f"{st.season:<14} {st.n_files:>6} {st.max_T:>7} {st.max_valid_days:>10}"
        )

    overall_T = max(st.max_T for st in rows)
    overall_v = max(st.max_valid_days for st in rows)
    _log("-" * len(hdr))
    _log(
        f"{'ALL SEASONS':<14} {'':>6} {overall_T:>7} {overall_v:>10}  <- worst-case caps"
    )
    print()

    for st in rows:
        _log(f"--- {st.season} ---")
        if st.best_T_path:
            _log(
                f"  Longest time axis (max_T={st.max_T}): {rel(st.best_T_path, root)}"
            )
        if st.best_valid_path:
            _log(
                f"  Most valid-day pixel ({st.max_valid_days} days, row {st.best_valid_row}): "
                f"{rel(st.best_valid_path, root)}"
            )
        print()


if __name__ == "__main__":
    main()
