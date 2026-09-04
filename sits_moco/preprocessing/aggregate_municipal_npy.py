#!/usr/bin/env python3
"""
Collapse each municipal .npy from [N, T, C] to a single mega-pixel [1, T, C].

For every date (axis T), spectral bands 0–9 are reduced over pixels that have
data that day (finite, not 0, not -9999). Extra channels (rain/climate at 11+)
use the same rule except 0 is valid. DOY (channel 10) is copied, not reduced.

Output layout matches training: {out}/{YYYY-YYYY}/{code}/{code}.npy

This does not change the original pixel .npy files.

Examples:
  python preprocessing/aggregate_municipal_npy.py --stat mean
  python preprocessing/aggregate_municipal_npy.py --stat median -j 8
  python preprocessing/aggregate_municipal_npy.py --stat mean median \\
      --input-dir ~/sits_moco_data/npy
"""

from __future__ import annotations

import argparse
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from datasets.pixel_transform import DOY_CHANNEL, NO_DATA_VALUE, NUM_SPECTRAL_CHANNELS
from env_config import resolve_datapath

SEASON_DIR_RE = re.compile(r"^\d{4}-\d{4}$")
CHUNK_PIXELS = 8192
STATS = ("mean", "median")


def _save_npy_atomic(out_path: Path, arr: np.ndarray, *, max_retries: int = 4) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
    last_err: OSError | None = None
    for attempt in range(max_retries):
        try:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            np.save(os.fspath(tmp), arr)
            os.replace(os.fspath(tmp), os.fspath(out_path))
            return
        except OSError as e:
            last_err = e
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            time.sleep(0.4 * (attempt + 1))
    assert last_err is not None
    raise last_err


def _spectral_valid(spec: np.ndarray) -> np.ndarray:
    return np.isfinite(spec) & (spec != 0) & (spec != NO_DATA_VALUE)


def _extra_valid(extra: np.ndarray) -> np.ndarray:
    return np.isfinite(extra) & (extra != NO_DATA_VALUE)


def reduce_to_megapixel(data: np.ndarray, stat: str) -> np.ndarray:
    """Reduce [N, T, C] → [T, C] with per-date available-pixel mean or median."""
    if stat not in STATS:
        raise ValueError(f"stat must be one of {STATS}, got {stat!r}")
    if data.ndim != 3:
        raise ValueError(f"Expected [N, T, C], got shape {data.shape}")
    n, t, c = data.shape
    if n == 0 or t == 0 or c <= DOY_CHANNEL:
        raise ValueError(f"Unusable array shape {data.shape}")

    out = np.full((t, c), NO_DATA_VALUE, dtype=np.float32)
    out[:, DOY_CHANNEL] = np.asarray(data[0, :, DOY_CHANNEL], dtype=np.float32)
    n_extra = max(0, c - (DOY_CHANNEL + 1))

    if stat == "mean":
        spec_sum = np.zeros((t, NUM_SPECTRAL_CHANNELS), dtype=np.float64)
        spec_cnt = np.zeros((t, NUM_SPECTRAL_CHANNELS), dtype=np.int64)
        extra_sum = np.zeros((t, n_extra), dtype=np.float64) if n_extra else None
        extra_cnt = np.zeros((t, n_extra), dtype=np.int64) if n_extra else None
        for start in range(0, n, CHUNK_PIXELS):
            chunk = np.asarray(data[start : start + CHUNK_PIXELS], dtype=np.float32)
            spec = chunk[:, :, :NUM_SPECTRAL_CHANNELS]
            valid = _spectral_valid(spec)
            spec_sum += np.where(valid, spec, 0.0).sum(axis=0)
            spec_cnt += valid.sum(axis=0)
            if n_extra and extra_sum is not None and extra_cnt is not None:
                extra = chunk[:, :, DOY_CHANNEL + 1 :]
                ev = _extra_valid(extra)
                extra_sum += np.where(ev, extra, 0.0).sum(axis=0)
                extra_cnt += ev.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            spec_mean = spec_sum / spec_cnt
        out[:, :NUM_SPECTRAL_CHANNELS] = np.where(
            spec_cnt > 0, spec_mean, NO_DATA_VALUE
        ).astype(np.float32)
        if n_extra and extra_sum is not None and extra_cnt is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                extra_mean = extra_sum / extra_cnt
            out[:, DOY_CHANNEL + 1 :] = np.where(
                extra_cnt > 0, extra_mean, NO_DATA_VALUE
            ).astype(np.float32)
        return out

    for ti in range(t):
        spec = np.asarray(data[:, ti, :NUM_SPECTRAL_CHANNELS], dtype=np.float32)
        valid = _spectral_valid(spec)
        masked = np.where(valid, spec, np.nan)
        with np.errstate(all="ignore"):
            med = np.nanmedian(masked, axis=0)
        out[ti, :NUM_SPECTRAL_CHANNELS] = np.where(
            np.isfinite(med), med, NO_DATA_VALUE
        ).astype(np.float32)
        if n_extra:
            extra = np.asarray(data[:, ti, DOY_CHANNEL + 1 :], dtype=np.float32)
            ev = _extra_valid(extra)
            masked_e = np.where(ev, extra, np.nan)
            with np.errstate(all="ignore"):
                med_e = np.nanmedian(masked_e, axis=0)
            out[ti, DOY_CHANNEL + 1 :] = np.where(
                np.isfinite(med_e), med_e, NO_DATA_VALUE
            ).astype(np.float32)
    return out


def default_output_dir(input_dir: Path, stat: str) -> Path:
    return input_dir.parent / f"{input_dir.name}_muni_{stat}"


def discover_seasons(root: Path, year_ranges: list[str] | None) -> list[str]:
    if year_ranges:
        return list(year_ranges)
    children = sorted(
        p.name for p in root.iterdir() if p.is_dir() and SEASON_DIR_RE.match(p.name)
    )
    if children:
        return children
    if SEASON_DIR_RE.match(root.name):
        return [root.name]
    return []


def list_jobs(input_dir: Path, seasons: list[str]) -> list[tuple[str, str, Path]]:
    """Return (season, code, npy_path) for municipal arrays (skip keep_mask)."""
    jobs: list[tuple[str, str, Path]] = []
    for season in seasons:
        season_dir = input_dir if input_dir.name == season else input_dir / season
        if not season_dir.is_dir():
            continue
        for muni_dir in sorted(season_dir.iterdir()):
            if not muni_dir.is_dir():
                continue
            code = muni_dir.name
            npy = muni_dir / f"{code}.npy"
            if npy.is_file():
                jobs.append((season, code, npy))
    return jobs


def process_one(
    src: Path,
    dest: Path,
    stat: str,
    skip_existing: bool,
) -> tuple[str, int, str | None]:
    try:
        if skip_existing and dest.is_file():
            existing = np.load(dest, mmap_mode="r")
            n = int(existing.shape[0]) if existing.ndim >= 1 else 0
            return (src.as_posix(), n, None)
        data = np.load(src, mmap_mode="r")
        reduced = reduce_to_megapixel(data, stat)
        mega = np.ascontiguousarray(reduced[np.newaxis, ...], dtype=np.float32)
        _save_npy_atomic(dest, mega)
        return (src.as_posix(), int(data.shape[0]), None)
    except Exception as e:
        return (src.as_posix(), 0, str(e))


def _worker(src: str, dest: str, stat: str, skip_existing: bool) -> tuple[str, int, str | None]:
    return process_one(Path(src), Path(dest), stat, skip_existing)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Original pixel .npy root (default: SITS_MOCO_DATAPATH from .env)",
    )
    p.add_argument(
        "--stat",
        nargs="+",
        choices=list(STATS),
        default=["mean"],
        help="Reduction(s) to write (default: mean). Pass both: --stat mean median",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root for a single --stat. Ignored when writing both stats.",
    )
    p.add_argument(
        "--year-range",
        action="append",
        default=[],
        metavar="YYYY-YYYY",
        help="Season folder to process (repeatable). Default: all YYYY-YYYY under input.",
    )
    p.add_argument(
        "-j",
        "--workers",
        type=int,
        default=4,
        help="Parallel municipality workers (default: 4)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing mega-pixel files",
    )
    return p.parse_args()


def output_dir_for_stat(args: argparse.Namespace, input_dir: Path, stat: str, n_stats: int) -> Path:
    if n_stats == 1 and args.output_dir is not None:
        return Path(args.output_dir).expanduser().resolve()
    if n_stats > 1 and args.output_dir is not None:
        return Path(args.output_dir).expanduser().resolve() / stat
    return default_output_dir(input_dir, stat).resolve()


def run_stat(
    jobs: list[tuple[str, str, Path]],
    out_root: Path,
    stat: str,
    workers: int,
    skip_existing: bool,
) -> None:
    print(f"\n=== {stat}: {len(jobs)} files → {out_root} ===")
    dests = [
        (src, out_root / season / code / f"{code}.npy")
        for season, code, src in jobs
    ]
    failed: list[tuple[str, str]] = []
    n_src_pixels = 0
    if workers <= 1:
        for src, dest in tqdm(dests, desc=stat, unit="muni"):
            _path, n, err = process_one(src, dest, stat, skip_existing)
            if err:
                failed.append((_path, err))
                print(f"[ERROR] {src}: {err}")
            else:
                n_src_pixels += n
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(_worker, str(src), str(dest), stat, skip_existing): src
                for src, dest in dests
            }
            for fut in tqdm(as_completed(futs), total=len(futs), desc=stat, unit="muni"):
                _path, n, err = fut.result()
                if err:
                    failed.append((_path, err))
                    print(f"[ERROR] {_path}: {err}")
                else:
                    n_src_pixels += n
    print(f"Done {stat}. Source pixels reduced: {n_src_pixels:,}")
    if failed:
        print(f"Failed {len(failed)} file(s): {[p for p, _ in failed[:10]]}")


def main() -> None:
    args = parse_args()
    input_dir = (
        Path(args.input_dir).expanduser().resolve()
        if args.input_dir is not None
        else resolve_datapath()
    )
    if not input_dir.is_dir():
        raise SystemExit(f"Input dir not found: {input_dir}")

    stats = list(dict.fromkeys(args.stat))
    seasons = discover_seasons(input_dir, args.year_range or None)
    if not seasons:
        raise SystemExit(f"No YYYY-YYYY season folders under {input_dir}")

    jobs = list_jobs(input_dir, seasons)
    if not jobs:
        raise SystemExit(f"No municipal .npy files under {input_dir} for {seasons}")

    print(
        f"Input: {input_dir}\n"
        f"Seasons: {seasons}\n"
        f"Municipal files: {len(jobs)}\n"
        f"Stats: {stats}"
    )
    skip_existing = not args.overwrite
    workers = max(1, int(args.workers))
    for stat in stats:
        out_root = output_dir_for_stat(args, input_dir, stat, len(stats))
        run_stat(jobs, out_root, stat, workers, skip_existing)
        print(f"Train with --datapath {out_root}")


if __name__ == "__main__":
    main()
