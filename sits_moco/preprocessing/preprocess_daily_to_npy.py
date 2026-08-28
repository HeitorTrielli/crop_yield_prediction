#!/usr/bin/env python3
"""
Preprocess daily municipal TIFFs (from clip_tiffs_to_shapefiles) to .npy.

Reads TIFFs under {input_dir}/{year_range}/{municipal_code}/, stacks to
[num_pixels, num_days, 11] (10 bands + DOY), [N, T, 13] with --xavier-pr-nc
(cumulative mm + dry streak), or [N, T, 17] with rain plus cumulative Tmax,
Tmin, Rs, ETo (see data_download/xavier_rain_for_daily_npy.py).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
import time
import uuid
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
# After DOY (10): rain 11:13, then cumulative Tmax, Tmin, Rs, ETo at 13:17
CUM_CLIMATE_SPECS: tuple[tuple[str, str], ...] = (
    ("tmax", "Tmax"),
    ("tmin", "Tmin"),
    ("rs", "Rs"),
    ("eto", "ETo"),
)


def _copy_nc_to_work(src: Path, prefix: str) -> Path:
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    src = src.expanduser().resolve()
    if not src.is_file():
        raise SystemExit(f"NetCDF is not a file: {src}")
    try:
        blob = src.read_bytes()
    except OSError as e:
        raise SystemExit(f"Could not read {src}: {e}") from e
    tmp = tempfile.NamedTemporaryFile(
        prefix=f"{prefix}_{uuid.uuid4().hex}_",
        suffix=".nc",
        delete=False,
    )
    try:
        tmp.write(blob)
        tmp.flush()
        os.fsync(tmp.fileno())
    finally:
        tmp.close()
    return Path(tmp.name)


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
        default=None,
        metavar="N",
        help="Parallel workers for processing municipalities (default: min(12, CPU-2)).",
    )
    p.add_argument(
        "--xavier-pr-nc",
        type=Path,
        default=None,
        help="Xavier precip NetCDF. Appends 2 rain channels (cumulative mm, dry streak).",
    )
    p.add_argument(
        "--xavier-tmax-nc",
        type=Path,
        default=None,
        help="Xavier Tmax NetCDF. Appends cumulative Tmax from Oct 1 (channel 13).",
    )
    p.add_argument(
        "--xavier-tmin-nc",
        type=Path,
        default=None,
        help="Xavier Tmin NetCDF. Appends cumulative Tmin from Oct 1 (channel 14).",
    )
    p.add_argument(
        "--xavier-rs-nc",
        type=Path,
        default=None,
        help="Xavier Rs NetCDF (MJ m-2). Appends cumulative Rs from Oct 1 (channel 15).",
    )
    p.add_argument(
        "--xavier-eto-nc",
        type=Path,
        default=None,
        help="Xavier ETo NetCDF (mm). Appends cumulative ETo from Oct 1 (channel 16).",
    )
    p.add_argument(
        "--limit-munis",
        type=int,
        default=None,
        help="Process only the first N municipalities (smoke test).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip municipalities whose {code}.npy already exists (resume a killed run).",
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
        same_grid = (
            src.height == height
            and src.width == width
            and src.transform.almost_equals(ref_transform)
            and (src.crs is None or src.crs == ref_crs)
        )
        if same_grid:
            out[:n] = src.read(indexes=list(range(1, n + 1)), out_dtype="float32")
        else:
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


def _fill_climate_channels(
    out: np.ndarray,
    *,
    year_range: str,
    rows: np.ndarray,
    cols: np.ndarray,
    ref_transform,
    ref_crs,
    dates: list[date],
    xavier_pr_work: Path | None,
    climate_work: dict[str, Path | None],
) -> None:
    from data_download.xavier_rain_for_daily_npy import (
        compute_cumulative_block_for_municipality,
        compute_rain_block_for_municipality,
        pixel_centroids_lonlat,
    )

    lon, lat = pixel_centroids_lonlat(rows, cols, ref_transform, ref_crs)

    if xavier_pr_work is not None:
        rain = compute_rain_block_for_municipality(
            xavier_pr_work,
            year_range,
            rows,
            cols,
            ref_transform,
            ref_crs,
            dates,
            lon=lon,
            lat=lat,
        )
        rain = np.where(np.isfinite(rain), rain, NO_DATA_VALUE)
        out[:, :, 11:13] = rain

    for offset, (key, var_name) in enumerate(CUM_CLIMATE_SPECS):
        nc = climate_work.get(key)
        if nc is None:
            continue
        ch = 13 + offset
        block = compute_cumulative_block_for_municipality(
            nc,
            var_name,
            year_range,
            rows,
            cols,
            ref_transform,
            ref_crs,
            dates,
            lon=lon,
            lat=lat,
        )
        out[:, :, ch] = np.where(np.isfinite(block), block, NO_DATA_VALUE)


def process_municipality(
    code: str,
    tiff_paths: list[Path],
    year_range: str,
    output_dir: Path,
    season_start: date,
    xavier_pr_work: Path | None = None,
    climate_work: dict[str, Path | None] | None = None,
) -> int:
    """Load daily TIFFs for one municipality, build [N, T, 11|13|17], save .npy."""
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

    climate_work = climate_work or {}
    has_rain = xavier_pr_work is not None
    has_cum = any(climate_work.get(k) is not None for k, _ in CUM_CLIMATE_SPECS)
    n_extra = (2 if has_rain or has_cum else 0) + (4 if has_cum else 0)
    n_out_ch = 11 + n_extra

    # Append DOY: [num_pixels, num_days, 11..]
    doy_broadcast = np.broadcast_to(doys, (len(stack), num_days))
    out = np.zeros((len(stack), num_days, n_out_ch), dtype=np.float32)
    out[:, :, :nbands] = stack
    out[:, :, 10] = doy_broadcast

    if has_rain or has_cum:
        rows = (idx_final // width).astype(np.int64)
        cols = (idx_final % width).astype(np.int64)
        _fill_climate_channels(
            out,
            year_range=year_range,
            rows=rows,
            cols=cols,
            ref_transform=ref_transform,
            ref_crs=ref_crs,
            dates=list(dates),
            xavier_pr_work=xavier_pr_work,
            climate_work=climate_work,
        )

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
    climate_work_str: dict[str, str | None] | None,
) -> tuple[str, int, str | None]:
    try:
        xp = Path(xavier_pr_work) if xavier_pr_work else None
        cw: dict[str, Path | None] = {}
        for k, v in (climate_work_str or {}).items():
            cw[k] = Path(v) if v else None
        n = process_municipality(
            code,
            tiff_paths,
            year_range,
            output_dir,
            season_start,
            xavier_pr_work=xp,
            climate_work=cw,
        )
        return (code, n, None)
    except Exception as e:
        return (code, 0, str(e))


def _default_workers() -> int:
    n = os.cpu_count() or 4
    return max(1, min(12, n - 2))


def _init_pool_worker(climate_pack_dir: str | None) -> None:
    """Mmap shared climate cubes; pin GDAL/BLAS to 1 thread."""
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    os.environ["GDAL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    _ensure_pyproj_proj_data()
    if not climate_pack_dir:
        return
    from data_download.xavier_rain_for_daily_npy import load_climate_mmap_pack

    load_climate_mmap_pack(climate_pack_dir)


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
    workers = max(1, int(args.workers) if args.workers is not None else _default_workers())
    items = sorted(muni_to_paths.items(), key=lambda x: x[0])
    if args.limit_munis is not None:
        items = items[: max(0, int(args.limit_munis))]
    if args.skip_existing:
        before = len(items)
        items = [
            (code, paths)
            for code, paths in items
            if not (out_base / year_range / code / f"{code}.npy").is_file()
        ]
        print(
            f"Skip-existing: {before - len(items)} already on disk, {len(items)} remaining"
        )
    if not items:
        print("Nothing to process.")
        return

    def _optional_nc(path: Path | None, flag: str) -> Path | None:
        if path is None:
            return None
        if not Path(path).is_file():
            raise SystemExit(f"{flag} not a file: {path}")
        return Path(path)

    pr_src = _optional_nc(args.xavier_pr_nc, "--xavier-pr-nc")
    climate_src = {
        "tmax": _optional_nc(args.xavier_tmax_nc, "--xavier-tmax-nc"),
        "tmin": _optional_nc(args.xavier_tmin_nc, "--xavier-tmin-nc"),
        "rs": _optional_nc(args.xavier_rs_nc, "--xavier-rs-nc"),
        "eto": _optional_nc(args.xavier_eto_nc, "--xavier-eto-nc"),
    }

    work_copies: list[Path] = []
    xavier_work_path: Path | None = None
    climate_work: dict[str, Path | None] = {k: None for k, _ in CUM_CLIMATE_SPECS}
    try:
        if pr_src is not None:
            xavier_work_path = _copy_nc_to_work(pr_src, "xavier_pr")
            work_copies.append(xavier_work_path)
        for key, src in climate_src.items():
            if src is None:
                continue
            wp = _copy_nc_to_work(src, f"xavier_{key}")
            work_copies.append(wp)
            climate_work[key] = wp
    except Exception:
        for p in work_copies:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        raise

    n_out = 11
    if pr_src is not None or any(climate_work.values()):
        n_out = 13
    if any(climate_work.values()):
        n_out = 17

    print(
        f"Processing {len(items)} municipalities, year-range {year_range}, "
        f"season_start={season_start} (DOY 1), workers={workers}, n_channels={n_out}"
        + (f", pr={pr_src}" if pr_src else "")
        + "".join(
            f", {k}={climate_src[k]}" for k in climate_work if climate_src[k] is not None
        )
    )
    total_pixels = 0
    failed: list[tuple[str, str]] = []
    xavier_work_str = os.fspath(xavier_work_path) if xavier_work_path else None
    climate_work_str = {
        k: (os.fspath(v) if v is not None else None) for k, v in climate_work.items()
    }

    climate_pairs: list[tuple[str, str]] = []
    if xavier_work_str:
        climate_pairs.append((xavier_work_str, "pr"))
    for key, var_name in CUM_CLIMATE_SPECS:
        pth = climate_work_str.get(key)
        if pth:
            climate_pairs.append((pth, var_name))

    pack_dir: Path | None = None
    try:
        pack_dir_str: str | None = None
        if climate_pairs:
            from data_download.xavier_rain_for_daily_npy import write_climate_mmap_pack

            pack_dir = Path(tempfile.mkdtemp(prefix="xavier_cubes_"))
            write_climate_mmap_pack(climate_pairs, pack_dir)
            pack_dir_str = os.fspath(pack_dir)

        if workers <= 1:
            _init_pool_worker(pack_dir_str)
            for code, paths in tqdm(items, desc="Municipalities", unit="muni"):
                try:
                    n = process_municipality(
                        code,
                        paths,
                        year_range,
                        out_base,
                        season_start,
                        xavier_pr_work=xavier_work_path,
                        climate_work=climate_work,
                    )
                    total_pixels += n
                except Exception as e:
                    failed.append((code, str(e)))
                    print(f"[ERROR] {code}: {e}")
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_pool_worker,
                initargs=(pack_dir_str,),
            ) as executor:
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
        for p in work_copies:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        if pack_dir is not None:
            shutil.rmtree(pack_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
