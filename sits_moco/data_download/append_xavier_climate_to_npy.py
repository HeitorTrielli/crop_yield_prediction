#!/usr/bin/env python3
"""
Append Xavier climate channels (cum ETo, Rs, Tmax, Tmin) to existing 13-ch .npy files.

Input .npy must be [N, T, 13] (10 spectral + DOY + 2 rain). Output is [N, T, 17]
with channels 13:17 = cum ETo, cum Rs, cum Tmax, cum Tmin.

Pixel rows/cols and observation dates are reconstructed from the same daily TIFF
stack used by preprocessing/preprocess_daily_to_npy.py (must match the original build).

Example:
  python data_download/append_xavier_climate_to_npy.py \\
    --npy-root files/npy --tiff-root files/daily_tiff --year-range 2020-2021 \\
    --xavier-eto-nc path/to/ETo.nc --xavier-rs-nc path/to/Rs.nc \\
    --xavier-tmax-nc path/to/Tmax.nc --xavier-tmin-nc path/to/Tmin.nc
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm

# Repo root on sys.path when run as a script
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data_download.xavier_climate_for_daily_npy import (  # noqa: E402
    N_CLIMATE_CHANNELS,
    compute_climate_block_for_municipality,
    copy_nc_to_temp,
)
from preprocessing.preprocess_daily_to_npy import (  # noqa: E402
    NO_DATA_VALUE,
    NUM_SPECTRAL_BANDS,
    _read_bands_on_reference_grid,
    _save_npy_atomic,
    parse_date_from_stem,
)
from preprocessing.preprocess_tiff_to_npy import (  # noqa: E402
    season_start_from_year_range,
)

EXPECTED_IN_CH = 13
EXPECTED_OUT_CH = 13 + N_CLIMATE_CHANNELS  # 17


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Append Xavier cum ETo/Rs/Tmax/Tmin channels to 13-ch municipal .npy files."
    )
    p.add_argument(
        "--npy-root",
        type=Path,
        required=True,
        help="Base dir with {year_range}/{code}/{code}.npy",
    )
    p.add_argument(
        "--tiff-root",
        type=Path,
        required=True,
        help="Daily TIFF root {year_range}/{code}/*.tiff (same as preprocess --input-dir)",
    )
    p.add_argument(
        "--year-range",
        type=str,
        required=True,
        metavar="YYYY-YYYY",
        help="Season folder label, e.g. 2020-2021",
    )
    p.add_argument("--xavier-eto-nc", type=Path, required=True)
    p.add_argument("--xavier-rs-nc", type=Path, required=True)
    p.add_argument("--xavier-tmax-nc", type=Path, required=True)
    p.add_argument("--xavier-tmin-nc", type=Path, required=True)
    p.add_argument(
        "-j",
        "--workers",
        type=int,
        default=1,
        help="Parallel municipality workers (default: 1)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List municipalities that would be updated without writing",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rewrite even if .npy already has 17 channels",
    )
    p.add_argument(
        "--codes",
        type=str,
        default=None,
        help="Optional comma-separated municipality codes to process",
    )
    return p.parse_args()


def _pixel_geometry_from_tiffs(
    code: str,
    tiff_paths: list[Path],
) -> tuple[list[date], np.ndarray, np.ndarray, object, object]:
    """
    Reconstruct observation dates and (row, col) for kept pixels — same filter as
    preprocess_daily_to_npy.process_municipality.
    """
    dated: list[tuple[date, Path]] = []
    for p in tiff_paths:
        d = parse_date_from_stem(p.stem, code)
        if d is not None:
            dated.append((d, p))
    dated.sort(key=lambda x: x[0])
    if not dated:
        raise ValueError(f"No dated TIFFs for {code}")

    dates = [x[0] for x in dated]
    paths = [x[1] for x in dated]
    num_days = len(dates)

    with rasterio.open(paths[0]) as src0:
        height, width = src0.height, src0.width
        nbands = min(src0.count, NUM_SPECTRAL_BANDS)
        ref_transform = src0.transform
        ref_crs = src0.crs
        if ref_crs is None:
            raise ValueError(f"TIFF has no CRS: {paths[0]}")

    stack = np.zeros((num_days, nbands, height, width), dtype=np.float32)
    for t, path in enumerate(paths):
        stack[t] = _read_bands_on_reference_grid(
            path, ref_transform, ref_crs, height, width, nbands
        )

    stack = np.transpose(stack, (2, 3, 0, 1)).reshape(height * width, num_days, nbands)
    valid = ~np.isnan(stack) & (stack != 0) & (stack != NO_DATA_VALUE)
    has_data = np.any(valid, axis=(1, 2))
    idx_kept = np.flatnonzero(has_data)
    stack_kept = stack[has_data]
    days_with_data = np.any(~np.isnan(stack_kept) & (stack_kept != 0), axis=2)
    enough = np.sum(days_with_data, axis=1) > 2
    idx_final = idx_kept[enough]
    rows = (idx_final // width).astype(np.int64)
    cols = (idx_final % width).astype(np.int64)
    return dates, rows, cols, ref_transform, ref_crs


def append_climate_one(
    code: str,
    npy_path: Path,
    tiff_dir: Path,
    year_range: str,
    nc_work: dict[str, Path],
    *,
    dry_run: bool,
    force: bool,
) -> tuple[str, str]:
    """
    Returns (code, status) where status is 'ok', 'skip', 'dry', or 'error:...'.
    """
    if not npy_path.is_file():
        return code, f"error: missing npy {npy_path}"

    arr = np.load(npy_path, mmap_mode="r")
    if arr.ndim != 3:
        return code, f"error: expected 3D npy, got shape {arr.shape}"
    n, t, c = arr.shape
    if c == EXPECTED_OUT_CH and not force:
        return code, "skip: already 17 channels"
    if c != EXPECTED_IN_CH and not (c == EXPECTED_OUT_CH and force):
        return code, f"error: expected {EXPECTED_IN_CH} channels, got {c}"

    tiff_paths = sorted(tiff_dir.glob("*.tiff")) + sorted(tiff_dir.glob("*.tif"))
    if not tiff_paths:
        return code, f"error: no TIFFs in {tiff_dir}"

    try:
        dates, rows, cols, transform, crs = _pixel_geometry_from_tiffs(code, tiff_paths)
    except Exception as e:
        return code, f"error: geometry {e}"

    if len(rows) != n:
        return (
            code,
            f"error: TIFF pixel count {len(rows)} != npy N={n} "
            "(re-preprocess or check tiff-root)",
        )
    if len(dates) != t:
        return (
            code,
            f"error: TIFF days {len(dates)} != npy T={t}",
        )

    if dry_run:
        return code, "dry: would append 4 climate channels"

    try:
        climate = compute_climate_block_for_municipality(
            nc_work,
            year_range,
            rows,
            cols,
            transform,
            crs,
            list(dates),
        )
    except Exception as e:
        return code, f"error: climate {e}"

    climate = np.where(np.isfinite(climate), climate, NO_DATA_VALUE).astype(np.float32)
    base = np.array(arr[:, :, :EXPECTED_IN_CH], dtype=np.float32, copy=True)
    out = np.concatenate([base, climate], axis=-1)
    if out.shape != (n, t, EXPECTED_OUT_CH):
        return code, f"error: bad out shape {out.shape}"

    try:
        _save_npy_atomic(npy_path, out)
    except Exception as e:
        return code, f"error: save {e}"
    return code, "ok"


def _worker(
    code: str,
    npy_path: str,
    tiff_dir: str,
    year_range: str,
    nc_work: dict[str, str],
    dry_run: bool,
    force: bool,
) -> tuple[str, str]:
    return append_climate_one(
        code,
        Path(npy_path),
        Path(tiff_dir),
        year_range,
        {k: Path(v) for k, v in nc_work.items()},
        dry_run=dry_run,
        force=force,
    )


def main() -> None:
    args = parse_args()
    year_range = args.year_range.strip()
    try:
        season_start_from_year_range(year_range)
    except ValueError as e:
        raise SystemExit(str(e)) from e

    npy_season = Path(args.npy_root) / year_range
    tiff_season = Path(args.tiff_root) / year_range
    if not npy_season.is_dir():
        raise SystemExit(f"Not a directory: {npy_season}")
    if not tiff_season.is_dir():
        raise SystemExit(f"Not a directory: {tiff_season}")

    for label, path in (
        ("eto", args.xavier_eto_nc),
        ("rs", args.xavier_rs_nc),
        ("tmax", args.xavier_tmax_nc),
        ("tmin", args.xavier_tmin_nc),
    ):
        if not Path(path).is_file():
            raise SystemExit(f"--xavier-{label}-nc not a file: {path}")

    codes_filter = None
    if args.codes:
        codes_filter = {c.strip() for c in args.codes.split(",") if c.strip()}

    jobs: list[tuple[str, Path, Path]] = []
    for muni_dir in sorted(npy_season.iterdir()):
        if not muni_dir.is_dir():
            continue
        code = muni_dir.name
        if codes_filter is not None and code not in codes_filter:
            continue
        npy_path = muni_dir / f"{code}.npy"
        tiff_dir = tiff_season / code
        if npy_path.is_file():
            jobs.append((code, npy_path, tiff_dir))

    if not jobs:
        print("No municipality .npy files found.")
        return

    print(
        f"Append climate → {EXPECTED_OUT_CH}ch for {len(jobs)} municipalities, "
        f"year-range={year_range}, dry_run={args.dry_run}"
    )

    nc_work_paths: dict[str, Path] = {}
    try:
        os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
        nc_work_paths = {
            "ETo": copy_nc_to_temp(Path(args.xavier_eto_nc), prefix="xavier_eto"),
            "Rs": copy_nc_to_temp(Path(args.xavier_rs_nc), prefix="xavier_rs"),
            "Tmax": copy_nc_to_temp(Path(args.xavier_tmax_nc), prefix="xavier_tmax"),
            "Tmin": copy_nc_to_temp(Path(args.xavier_tmin_nc), prefix="xavier_tmin"),
        }
        nc_work_str = {k: os.fspath(v) for k, v in nc_work_paths.items()}

        counts: dict[str, int] = {}
        workers = max(1, int(args.workers))
        if workers <= 1:
            for code, npy_path, tiff_dir in tqdm(jobs, desc="Municipalities", unit="muni"):
                _c, status = append_climate_one(
                    code,
                    npy_path,
                    tiff_dir,
                    year_range,
                    nc_work_paths,
                    dry_run=args.dry_run,
                    force=args.force,
                )
                key = status.split(":")[0]
                counts[key] = counts.get(key, 0) + 1
                if status.startswith("error"):
                    print(f"[ERROR] {code}: {status}")
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {
                    ex.submit(
                        _worker,
                        code,
                        str(npy_path),
                        str(tiff_dir),
                        year_range,
                        nc_work_str,
                        args.dry_run,
                        args.force,
                    ): code
                    for code, npy_path, tiff_dir in jobs
                }
                for fut in tqdm(
                    as_completed(futs), total=len(futs), desc="Municipalities", unit="muni"
                ):
                    _c, status = fut.result()
                    key = status.split(":")[0]
                    counts[key] = counts.get(key, 0) + 1
                    if status.startswith("error"):
                        print(f"[ERROR] {_c}: {status}")

        print("Summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    finally:
        for p in nc_work_paths.values():
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass


if __name__ == "__main__":
    main()
