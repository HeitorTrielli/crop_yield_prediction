#!/usr/bin/env python3
"""
Preprocess daily municipal TIFFs (from clip_tiffs_to_shapefiles) to .npy.

Reads TIFFs under {input_dir}/{year_range}/{municipal_code}/, stacks to
[num_pixels, num_days, 11] (10 bands + DOY), [N, T, 13] with --xavier-pr-nc
(two rain channels), or [N, T, 17] when ETo/Rs/Tmax/Tmin NetCDFs are also set
(see data_download/xavier_climate_for_daily_npy.py).
"""

from __future__ import annotations

import argparse
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path


def _ensure_pyproj_proj_data() -> None:
    """Point PROJ at pyproj's proj.db before rasterio loads GDAL (avoids PostGIS PROJ on Windows)."""
    pl = os.environ.get("PROJ_LIB", "")
    if pl and ("postgresql" in pl.lower() or "postgis" in pl.lower()):
        os.environ.pop("PROJ_LIB", None)
    try:
        import pyproj.datadir

        d = pyproj.datadir.get_data_dir()
        if d and os.path.isdir(d):
            os.environ["PROJ_DATA"] = d
    except Exception:
        pass


_ensure_pyproj_proj_data()

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from tqdm import tqdm

from preprocessing.preprocess_tiff_to_npy import (
    date_to_season_doy,
    season_start_from_year_range,
)

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert daily municipal TIFFs (clip output) to .npy time series."
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("files/daily_tiff"),
        help="Base dir containing {year_range}/{municipal_code}/*.tiff (default: files/daily_tiff)",
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
        default=Path("files/npy"),
        help="Output base dir; saves to {output_dir}/{year_range}/{code}/{code}.npy (default: files/npy)",
    )
    p.add_argument(
        "-j",
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Parallel workers for processing municipalities (default: 1). Use e.g. 10 on multi-core machines.",
    )
    p.add_argument(
        "--xavier-pr-nc",
        type=Path,
        default=None,
        help="Optional Xavier NetCDF (e.g. xavier_pr.nc). When set, appends 2 rain channels per timestep.",
    )
    p.add_argument(
        "--xavier-eto-nc",
        type=Path,
        default=None,
        help="Optional Xavier ETo NetCDF. With Rs/Tmax/Tmin, appends 4 climate channels (requires --xavier-pr-nc).",
    )
    p.add_argument(
        "--xavier-rs-nc",
        type=Path,
        default=None,
        help="Optional Xavier Rs (solar radiation) NetCDF.",
    )
    p.add_argument(
        "--xavier-tmax-nc",
        type=Path,
        default=None,
        help="Optional Xavier Tmax NetCDF.",
    )
    p.add_argument(
        "--xavier-tmin-nc",
        type=Path,
        default=None,
        help="Optional Xavier Tmin NetCDF.",
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
    xavier_pr_work: Path | None = None,
    xavier_climate_work: dict[str, Path] | None = None,
) -> int:
    """Load daily TIFFs for one municipality, build [num_pixels, num_days, 11|13|17], save .npy. Returns pixel count."""
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
    idx_kept = np.flatnonzero(has_data)
    stack = stack[has_data]  # [num_pixels_kept, num_days, 10]

    # Require at least 3 days with some valid band (like preprocess "months with data > 2")
    days_with_data = np.any(~np.isnan(stack) & (stack != 0), axis=2)
    enough = np.sum(days_with_data, axis=1) > 2
    idx_final = idx_kept[enough]
    stack = stack[enough]

    if len(stack) == 0:
        return 0

    # Fill remaining NaN with NO_DATA_VALUE for storage
    stack = np.where(np.isnan(stack), NO_DATA_VALUE, stack)

    n_extra = 0
    if xavier_pr_work is not None:
        n_extra += 2
    if xavier_climate_work is not None:
        n_extra += 4
    n_out_ch = 11 + n_extra

    # Append DOY: [num_pixels, num_days, 11..]
    doy_broadcast = np.broadcast_to(doys, (len(stack), num_days))
    out = np.zeros((len(stack), num_days, n_out_ch), dtype=np.float32)
    out[:, :, :nbands] = stack
    out[:, :, 10] = doy_broadcast

    rows = (idx_final // width).astype(np.int64)
    cols = (idx_final % width).astype(np.int64)

    if xavier_pr_work is not None:
        from data_download.xavier_rain_for_daily_npy import (
            compute_rain_block_for_municipality,
        )

        rain = compute_rain_block_for_municipality(
            xavier_pr_work,
            year_range,
            rows,
            cols,
            ref_transform,
            ref_crs,
            list(dates),
        )
        rain = np.where(np.isfinite(rain), rain, NO_DATA_VALUE)
        out[:, :, 11:13] = rain

    if xavier_climate_work is not None:
        from data_download.xavier_climate_for_daily_npy import (
            compute_climate_block_for_municipality,
        )

        climate = compute_climate_block_for_municipality(
            xavier_climate_work,
            year_range,
            rows,
            cols,
            ref_transform,
            ref_crs,
            list(dates),
        )
        climate = np.where(np.isfinite(climate), climate, NO_DATA_VALUE)
        out[:, :, 13:17] = climate

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
    xavier_pr_work: str | None,
    xavier_climate_work: dict[str, str] | None,
) -> tuple[str, int, str | None]:
    try:
        xp = Path(xavier_pr_work) if xavier_pr_work else None
        climate = (
            {k: Path(v) for k, v in xavier_climate_work.items()}
            if xavier_climate_work
            else None
        )
        n = process_municipality(
            code,
            tiff_paths,
            year_range,
            output_dir,
            season_start,
            xavier_pr_work=xp,
            xavier_climate_work=climate,
        )
        return (code, n, None)
    except Exception as e:
        return (code, 0, str(e))


def _prepare_xavier_work_copy(src: Path | None, prefix: str) -> Path | None:
    if src is None:
        return None
    from data_download.xavier_climate_for_daily_npy import copy_nc_to_temp

    return copy_nc_to_temp(Path(src), prefix=prefix)


def main() -> None:
    args = parse_args()
    base = Path(args.input_dir).resolve()
    year_range = args.year_range.strip()
    out_base = Path(args.output_dir).resolve()
    try:
        season_start = season_start_from_year_range(year_range)
    except ValueError as e:
        raise SystemExit(str(e)) from e

    # Discover municipalities: subdirs of base/{year_range}/
    season_path = base / year_range
    if not season_path.is_dir():
        raise SystemExit(f"Not a directory: {season_path}")

    muni_codes = sorted([d.name for d in season_path.iterdir() if d.is_dir()])
    if not muni_codes:
        print("No municipality directories found.")
        return

    # Collect TIFFs per municipality (TIFFs directly under base/{year_range}/{code}/)
    muni_to_paths: dict[str, list[Path]] = {}
    skipped_no_dir: list[str] = []
    skipped_no_tiffs: list[str] = []
    for code in muni_codes:
        muni_dir = season_path / code
        if not muni_dir.is_dir():
            skipped_no_dir.append(code)
            continue
        paths = sorted(muni_dir.glob("*.tiff")) + sorted(muni_dir.glob("*.tif"))
        if paths:
            muni_to_paths[code] = paths
        else:
            skipped_no_tiffs.append(code)

    if not muni_to_paths:
        print(f"No TIFFs found under {season_path}/{{code}}/")
        return

    n_skipped = len(skipped_no_dir) + len(skipped_no_tiffs)
    if n_skipped > 0:
        print(
            f"Skipped {n_skipped} municipalities (missing dir or no TIFFs): {skipped_no_dir + skipped_no_tiffs}"
        )
    workers = max(1, int(args.workers))

    climate_srcs = {
        "ETo": args.xavier_eto_nc,
        "Rs": args.xavier_rs_nc,
        "Tmax": args.xavier_tmax_nc,
        "Tmin": args.xavier_tmin_nc,
    }
    climate_any = any(v is not None for v in climate_srcs.values())
    climate_all = all(v is not None for v in climate_srcs.values())
    if climate_any and not climate_all:
        missing = [k for k, v in climate_srcs.items() if v is None]
        raise SystemExit(
            "Xavier climate channels require all of --xavier-eto-nc, --xavier-rs-nc, "
            f"--xavier-tmax-nc, --xavier-tmin-nc (missing: {missing})"
        )
    if climate_all and args.xavier_pr_nc is None:
        raise SystemExit(
            "Xavier climate channels (ETo/Rs/Tmax/Tmin) require --xavier-pr-nc "
            "so rain stays in channels 11:13 and climate in 13:17."
        )

    for label, path in [("pr", args.xavier_pr_nc), *climate_srcs.items()]:
        if path is not None and not Path(path).is_file():
            raise SystemExit(f"Xavier NetCDF not a file ({label}): {path}")

    xavier_work_path: Path | None = None
    climate_work: dict[str, Path] | None = None
    temp_paths: list[Path] = []

    try:
        os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
        if args.xavier_pr_nc is not None:
            xavier_work_path = _prepare_xavier_work_copy(args.xavier_pr_nc, "xavier_pr")
            if xavier_work_path is not None:
                temp_paths.append(xavier_work_path)
        if climate_all:
            climate_work = {}
            for key, src in climate_srcs.items():
                wp = _prepare_xavier_work_copy(src, f"xavier_{key.lower()}")
                assert wp is not None
                climate_work[key] = wp
                temp_paths.append(wp)

        xavier_str = (
            str(Path(args.xavier_pr_nc).resolve()) if args.xavier_pr_nc else None
        )
        print(
            f"Processing {len(muni_to_paths)} municipalities, year-range {year_range}, "
            f"season_start={season_start} (DOY 1), workers={workers}"
            + (f", xavier_pr_nc={xavier_str}" if xavier_str else "")
            + (f", xavier_climate={list(climate_srcs)}" if climate_all else "")
        )
        total_pixels = 0
        failed: list[tuple[str, str]] = []
        items = sorted(muni_to_paths.items(), key=lambda x: x[0])
        xavier_work_str = os.fspath(xavier_work_path) if xavier_work_path else None
        climate_work_str = (
            {k: os.fspath(v) for k, v in climate_work.items()} if climate_work else None
        )

        if workers <= 1:
            for code, paths in tqdm(items, desc="Municipalities", unit="muni"):
                try:
                    n = process_municipality(
                        code,
                        paths,
                        year_range,
                        out_base,
                        season_start,
                        xavier_pr_work=xavier_work_path,
                        xavier_climate_work=climate_work,
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
                        xavier_work_str,
                        climate_work_str,
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
    finally:
        for p in temp_paths:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass


if __name__ == "__main__":
    main()
