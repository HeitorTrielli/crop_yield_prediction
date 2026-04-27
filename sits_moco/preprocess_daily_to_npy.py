#!/usr/bin/env python3
"""
Preprocess daily municipal TIFFs (from clip_tiffs_to_shapefiles) to .npy.

Reads TIFFs under {input_dir}/{resolution}/{municipal_code}/{year_range}/
(e.g. files/gee_soy_daily/municipalities/30/4100103/2022-2023/*.tiff), parses date
from filename (4100103_2022_10_02.tiff), stacks into [num_pixels, num_days, 11]
(10 spectral bands + DOY). DOY = (date - Oct 1 of first year).days + 1 from --year-range
(YYYY-YYYY, e.g. 2020-2021 -> 2020-10-01 is day 1). Saves one .npy
per municipality under {output_dir}/{year_range}/{code}/{code}.npy.

Usage:
  python preprocess_daily_to_npy.py
  python preprocess_daily_to_npy.py --input-dir files/gee_soy_daily/municipalities --resolution 30 --year-range 2022-2023 --output-dir files/yield_dataset -j 10
"""

from __future__ import annotations

import argparse
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from tqdm import tqdm

from preprocess_tiff_to_npy import date_to_season_doy, season_start_from_year_range

NO_DATA_VALUE = -9999
NUM_SPECTRAL_BANDS = 10


def _save_npy_atomic(out_path: Path, arr: np.ndarray, *, max_retries: int = 4) -> None:
    """
    Write .npy via a temp file then os.replace. Retries on OSError (disk full,
    OneDrive sync locking, transient I/O). Direct np.save can fail with
    "X requested and 0 written" on cloud-synced folders.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Path must end with .npy: np.save appends ".npy" if the filename does not,
    # so the old "x.npy.tmp" became "x.npy.tmp.npy" and os.replace failed.
    tmp = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
    legacy_double = out_path.parent / f"{out_path.name}.tmp.npy"
    last_err: OSError | None = None
    for attempt in range(max_retries):
        try:
            for p in (tmp, legacy_double):
                if p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass
            np.save(os.fspath(tmp), arr)
            os.replace(os.fspath(tmp), os.fspath(out_path))
            return
        except OSError as e:
            last_err = e
            for p in (tmp, legacy_double):
                if p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass
            time.sleep(0.4 * (attempt + 1))
    assert last_err is not None
    raise last_err


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert daily municipal TIFFs (clip output) to .npy time series."
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("files/gee_soy_daily/municipalities"),
        help="Base dir containing {resolution}/{municipal_code}/{year_range}/ (default: files/gee_soy_daily/municipalities)",
    )
    p.add_argument(
        "--resolution",
        type=int,
        default=30,
        help="Resolution subfolder name (default: 30)",
    )
    p.add_argument(
        "--year-range",
        type=str,
        default="2022-2023",
        metavar="YYYY-YYYY",
        help="Year-range subfolder YYYY-YYYY; season DOY day 1 = Oct 1 of first year (default: 2022-2023)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("files/yield_dataset"),
        help="Output base dir; saves to {output_dir}/{year_range}/{code}/{code}.npy (default: files/yield_dataset)",
    )
    p.add_argument(
        "-j",
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Parallel workers for processing municipalities (default: 1). Use e.g. 10 on multi-core machines.",
    )
    return p.parse_args()


def parse_date_from_stem(stem: str, code: str) -> date | None:
    """From 4100103_2022_10_02 or 4100103_2022-10-02 return date(2022,10,2)."""
    if not stem.startswith(code + "_"):
        return None
    suffix = stem[len(code) + 1 :].replace("-", "_")
    parts = suffix.split("_")
    if len(parts) >= 3 and all(p.isdigit() for p in parts[:3]):
        y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
        if 1 <= m <= 12 and 1 <= d <= 31:
            try:
                return date(y, m, d)
            except ValueError:
                return None
    match = re.search(r"(\d{4})-?(\d{2})-?(\d{2})$", stem)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            pass
    return None


def _read_bands_on_reference_grid(
    path: Path,
    ref_transform: object,
    ref_crs: object,
    height: int,
    width: int,
    nbands: int,
) -> np.ndarray:
    """
    Read spectral bands and warp to the reference grid. Daily clips can differ in
    height/width; stacking without reprojection raises broadcast errors.
    """
    out = np.full((nbands, height, width), np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        src_crs = src.crs if src.crs is not None else ref_crs
        n = min(src.count, nbands)
        nodata = src.nodata if src.nodata is not None else NO_DATA_VALUE
        for b in range(n):
            reproject(
                rasterio.band(src, b + 1),
                out[b],
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.nearest,
                dst_nodata=np.nan,
            )
        for b in range(n):
            band = out[b]
            band[band == nodata] = np.nan
            band[band == 0] = np.nan
            band[band == NO_DATA_VALUE] = np.nan
    return out


def process_municipality(
    code: str,
    tiff_paths: list[Path],
    year_range: str,
    output_dir: Path,
    season_start: date,
) -> int:
    """Load daily TIFFs for one municipality, build [num_pixels, num_days, 11], save .npy. Returns pixel count."""
    if not tiff_paths:
        return 0

    # Parse date from each path and sort
    dated = []
    for p in tiff_paths:
        d = parse_date_from_stem(p.stem, code)
        if d is not None:
            dated.append((d, p))
    dated.sort(key=lambda x: x[0])
    if not dated:
        return 0

    dates, paths = [x[0] for x in dated], [x[1] for x in dated]
    num_days = len(dates)
    doys = np.array(
        [date_to_season_doy(d, season_start) for d in dates], dtype=np.float32
    )

    # Reference grid from first day (other days are warped here)
    with rasterio.open(paths[0]) as src0:
        height, width = src0.height, src0.width
        nbands = min(src0.count, NUM_SPECTRAL_BANDS)
        ref_transform = src0.transform
        ref_crs = src0.crs
        if ref_crs is None:
            raise ValueError(f"TIFF has no CRS (needed for reprojection): {paths[0]}")

    # Stack all days: [num_days, bands, H, W]
    stack = np.zeros((num_days, nbands, height, width), dtype=np.float32)
    for t, path in enumerate(paths):
        data = _read_bands_on_reference_grid(
            path, ref_transform, ref_crs, height, width, nbands
        )
        stack[t] = data

    # Reshape to [num_pixels, num_days, 10] then add DOY -> [num_pixels, num_days, 11]
    stack = np.transpose(stack, (2, 3, 0, 1))  # [H, W, T, 10]
    num_pixels = height * width
    stack = stack.reshape(num_pixels, num_days, nbands)

    # Filter: drop pixels that are all NaN/zero in every band every day
    valid = ~np.isnan(stack) & (stack != 0) & (stack != NO_DATA_VALUE)
    has_data = np.any(valid, axis=(1, 2))
    stack = stack[has_data]  # [num_pixels_kept, num_days, 10]

    # Require at least 3 days with some valid band (like preprocess "months with data > 2")
    days_with_data = np.any(~np.isnan(stack) & (stack != 0), axis=2)
    enough = np.sum(days_with_data, axis=1) > 2
    stack = stack[enough]

    if len(stack) == 0:
        return 0

    # Fill remaining NaN with NO_DATA_VALUE for storage
    stack = np.where(np.isnan(stack), NO_DATA_VALUE, stack)

    # Append DOY: [num_pixels, num_days, 11]
    doy_broadcast = np.broadcast_to(doys, (len(stack), num_days))
    out = np.zeros((len(stack), num_days, 11), dtype=np.float32)
    out[:, :, :nbands] = stack
    out[:, :, 10] = doy_broadcast

    out_dir = output_dir / year_range / code
    out_path = out_dir / f"{code}.npy"
    _save_npy_atomic(out_path, out)
    return len(stack)


def _process_municipality_worker(
    code: str,
    tiff_paths: list[Path],
    year_range: str,
    output_dir: Path,
    season_start: date,
) -> tuple[str, int, str | None]:
    try:
        n = process_municipality(
            code, tiff_paths, year_range, output_dir, season_start
        )
        return (code, n, None)
    except Exception as e:
        return (code, 0, str(e))


def main() -> None:
    args = parse_args()
    base = Path(args.input_dir).resolve()
    res_dir = str(args.resolution)
    year_range = args.year_range.strip()
    out_base = Path(args.output_dir).resolve()
    try:
        season_start = season_start_from_year_range(year_range)
    except ValueError as e:
        raise SystemExit(str(e)) from e

    # Discover municipalities: subdirs of base/resolution/
    res_path = base / res_dir
    if not res_path.is_dir():
        raise SystemExit(f"Not a directory: {res_path}")

    muni_codes = sorted([d.name for d in res_path.iterdir() if d.is_dir()])
    if not muni_codes:
        print("No municipality directories found.")
        return

    # Collect TIFFs per municipality (only those with at least one TIFF in {code}/{year_range}/)
    muni_to_paths: dict[str, list[Path]] = {}
    skipped_no_dir: list[str] = []
    skipped_no_tiffs: list[str] = []
    for code in muni_codes:
        muni_dir = res_path / code / year_range
        if not muni_dir.is_dir():
            skipped_no_dir.append(code)
            continue
        paths = sorted(muni_dir.glob("*.tiff")) + sorted(muni_dir.glob("*.tif"))
        if paths:
            muni_to_paths[code] = paths
        else:
            skipped_no_tiffs.append(code)

    if not muni_to_paths:
        print(f"No TIFFs found under {res_path}/{{code}}/{year_range}/")
        return

    n_skipped = len(skipped_no_dir) + len(skipped_no_tiffs)
    if n_skipped > 0:
        print(
            f"Skipped {n_skipped} municipalities (no {year_range}/ dir or no TIFFs): {skipped_no_dir + skipped_no_tiffs}"
        )
    workers = max(1, int(args.workers))
    print(
        f"Processing {len(muni_to_paths)} municipalities, year-range {year_range}, "
        f"season_start={season_start} (DOY 1), workers={workers}"
    )
    total_pixels = 0
    failed: list[tuple[str, str]] = []
    items = sorted(muni_to_paths.items(), key=lambda x: x[0])

    if workers <= 1:
        for code, paths in tqdm(items, desc="Municipalities", unit="muni"):
            try:
                n = process_municipality(
                    code, paths, year_range, out_base, season_start
                )
                total_pixels += n
            except Exception as e:
                failed.append((code, str(e)))
                print(f"[ERROR] {code}: {e}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _process_municipality_worker,
                    code,
                    paths,
                    year_range,
                    out_base,
                    season_start,
                ): code
                for code, paths in items
            }
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Municipalities",
                unit="muni",
            ):
                _code, n, err = fut.result()
                if err:
                    failed.append((_code, err))
                    print(f"[ERROR] {_code}: {err}")
                else:
                    total_pixels += n

    print(f"Done. Output: {out_base / year_range}/<code>/<code>.npy")
    print(f"Total pixels: {total_pixels:,}")
    if failed:
        print(
            f"Failed {len(failed)} municipality/municipalities (re-run after fixing disk/OneDrive): {[c for c, _ in failed]}"
        )


if __name__ == "__main__":
    main()
