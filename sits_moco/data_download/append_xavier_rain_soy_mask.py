#!/usr/bin/env python3
"""
Append Xavier rain to municipal mega-pixel .npy, masked to MapBiomas soy.

No Earth Engine. Soy locations come from the public MapBiomas Collection 10
COG (windowed HTTP read of each municipality shapefile). Rain is Xavier daily
precipitation, averaged over soy pixels (Xavier cells weighted by soy count).

Sentinel-2 dates drive the time axis — especially consecutive dry days, which
are the longest run of pr < 1 mm between overpasses, not over the whole
season. Dates are taken from, in order:
  1. files/zonal_mean/{season}/{code}/{code}_zonal.csv
  2. daily TIFF stems under --tiff-root
  3. season DOY already stored in the .npy

Channels 11:13 = cumulative mm from Oct 1, and max dry streak between S2
dates. A sidecar {code}_soy_rain.csv is written next to the .npy with the
soy-average mm on each S2 day.

The bundled NetCDF (files/climate/pr_PR_daily_2020_2024.nc) only has values
over Paraná and starts 2020-01-01 (2019-2020 misses Oct–Dec 2019).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data_download.bdc_zonal import mapbiomas_url, soy_pixel_lonlat
from data_download.xavier_climate_for_daily_npy import copy_nc_to_temp
from data_download.xavier_rain_for_daily_npy import (
    N_RAIN_CHANNELS,
    load_pr_grid,
    rain_channels_mean_at_lonlat,
)
from datasets.pixel_transform import DOY_CHANNEL, NO_DATA_VALUE
from preprocessing.preprocess_daily_to_npy import _save_npy_atomic, parse_date_from_stem
from preprocessing.preprocess_tiff_to_npy import season_start_from_year_range

RAIN_SLICE = (DOY_CHANNEL + 1, DOY_CHANNEL + 1 + N_RAIN_CHANNELS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--npy-root",
        type=Path,
        default=Path("~/sits_moco_data/npy_muni_mean"),
        help="Municipal mega-pixel root {YYYY-YYYY}/{code}/{code}.npy",
    )
    p.add_argument(
        "--shapefile-dir",
        type=Path,
        default=_REPO / "files" / "shapefiles",
    )
    p.add_argument(
        "--xavier-pr-nc",
        type=Path,
        default=_REPO / "files" / "climate" / "pr_PR_daily_2020_2024.nc",
    )
    p.add_argument(
        "--year-range",
        type=str,
        default=None,
        metavar="YYYY-YYYY",
        help="One season. Default: every YYYY-YYYY folder under --npy-root",
    )
    p.add_argument(
        "--codes",
        type=str,
        default=None,
        help="Comma-separated municipality codes (default: all .npy in the season)",
    )
    p.add_argument("--overwrite-rain", action="store_true")
    p.add_argument(
        "--zonal-root",
        type=Path,
        default=_REPO / "files" / "zonal_mean",
        help="Zonal CSVs with a date column (S2 overpasses)",
    )
    p.add_argument(
        "--tiff-root",
        type=Path,
        default=None,
        help="Optional daily TIFF root {YYYY-YYYY}/{code}/*.tiff (S2 dates from filenames)",
    )
    return p.parse_args()


def harvest_year(year_range: str) -> int:
    return int(year_range.split("-")[1])


def _parse_iso_date(value: object) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text[:10], fmt).date()
        except ValueError:
            continue
    return None


def dates_from_zonal_csv(zonal_root: Path, year_range: str, code: str) -> list[date]:
    path = zonal_root / year_range / code / f"{code}_zonal.csv"
    if not path.is_file():
        return []
    out: list[date] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = _parse_iso_date(row.get("date"))
            if d is not None:
                out.append(d)
    return sorted(set(out))


def dates_from_tiffs(tiff_root: Path, year_range: str, code: str) -> list[date]:
    folder = tiff_root / year_range / code
    if not folder.is_dir():
        return []
    out: list[date] = []
    for p in list(folder.glob("*.tiff")) + list(folder.glob("*.tif")):
        d = parse_date_from_stem(p.stem, code)
        if d is not None:
            out.append(d)
    return sorted(set(out))


def resolve_s2_dates(
    arr: np.ndarray,
    year_range: str,
    code: str,
    zonal_root: Path | None,
    tiff_root: Path | None,
) -> tuple[list[date], str]:
    """S2 overpass dates aligned to npy timesteps (length T)."""
    npy_dates = obs_dates_from_npy(arr, year_range)
    zonal = dates_from_zonal_csv(zonal_root, year_range, code) if zonal_root else []
    tiffs = dates_from_tiffs(tiff_root, year_range, code) if tiff_root else []
    for label, src in (("zonal_csv", zonal), ("tiff", tiffs)):
        if not src:
            continue
        if len(src) == len(npy_dates):
            return src, label
        by_doy = {date_to_season_doy(d, year_range): d for d in src}
        aligned: list[date] = []
        ok = True
        for nd in npy_dates:
            key = date_to_season_doy(nd, year_range)
            if key not in by_doy:
                ok = False
                break
            aligned.append(by_doy[key])
        if ok:
            return aligned, f"{label}_aligned"
    return npy_dates, "npy_doy"


def date_to_season_doy(d: date, year_range: str) -> int:
    start = season_start_from_year_range(year_range)
    return (d - start).days + 1


def write_soy_rain_csv(
    path: Path,
    dates: list[date],
    rain: np.ndarray,
    daily: np.ndarray,
    n_soy: int,
    n_cells: int,
    date_source: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "date",
                "soy_mean_pr_mm",
                "cum_pr_mm",
                "dry_streak_days",
                "n_soy_pixels",
                "n_xavier_cells",
                "s2_date_source",
            ]
        )
        for i, d in enumerate(dates):
            day = daily[i]
            cum = rain[i, 0]
            dry = rain[i, 1]
            w.writerow(
                [
                    d.isoformat(),
                    "" if not np.isfinite(day) else f"{float(day):.4f}",
                    "" if not np.isfinite(cum) else f"{float(cum):.4f}",
                    "" if not np.isfinite(dry) else f"{float(dry):.1f}",
                    n_soy,
                    n_cells,
                    date_source,
                ]
            )


def obs_dates_from_npy(arr: np.ndarray, year_range: str) -> list[date]:
    start = season_start_from_year_range(year_range)
    doys = np.asarray(arr[0, :, DOY_CHANNEL], dtype=np.float64)
    out: list[date] = []
    for raw in doys:
        if not np.isfinite(raw) or raw == NO_DATA_VALUE:
            out.append(start)
            continue
        out.append(start + timedelta(days=int(round(float(raw))) - 1))
    return out


def load_muni_geometry(shapefile_dir: Path, code: str):
    shp = shapefile_dir / code / f"{code}.shp"
    if not shp.is_file():
        matches = list(shapefile_dir.glob(f"{code}/*.shp"))
        if not matches:
            return None
        shp = matches[0]
    gdf = gpd.read_file(shp)
    if gdf.empty or gdf.geometry.is_empty.all():
        return None
    gdf = gdf.to_crs("EPSG:4326")
    if hasattr(gdf, "union_all"):
        return gdf.union_all()
    return gdf.geometry.unary_union


def list_jobs(npy_root: Path, year_range: str, codes: set[str] | None) -> list[tuple[str, Path]]:
    season = npy_root / year_range
    jobs: list[tuple[str, Path]] = []
    if not season.is_dir():
        return jobs
    for npy in sorted(season.glob("*/*.npy")):
        code = npy.parent.name
        if codes is not None and code not in codes:
            continue
        jobs.append((code, npy))
    return jobs


def attach_rain(
    arr: np.ndarray,
    rain: np.ndarray,
    overwrite_rain: bool,
) -> np.ndarray | None:
    """Return updated [1, T, C] or None if skipped."""
    t = arr.shape[1]
    if rain.shape != (t, N_RAIN_CHANNELS):
        raise ValueError(f"rain shape {rain.shape} != ({t}, {N_RAIN_CHANNELS})")
    c = arr.shape[2]
    lo, hi = RAIN_SLICE
    if c < hi:
        out = np.full((1, t, hi), NO_DATA_VALUE, dtype=np.float32)
        out[:, :, :c] = arr
        out[:, :, lo:hi] = rain
        return out
    if not overwrite_rain:
        return None
    out = np.array(arr, dtype=np.float32, copy=True)
    out[0, :, lo:hi] = rain
    return out


def process_one(
    code: str,
    npy_path: Path,
    shapefile_dir: Path,
    year_range: str,
    mapbiomas_path: str,
    pr_vals: np.ndarray,
    time_values: np.ndarray,
    x_coord: np.ndarray,
    y_coord: np.ndarray,
    overwrite_rain: bool,
    zonal_root: Path | None,
    tiff_root: Path | None,
) -> str:
    arr = np.load(npy_path)
    if arr.ndim != 3 or arr.shape[0] < 1 or arr.shape[2] <= DOY_CHANNEL:
        return "bad_npy"
    if arr.shape[2] >= RAIN_SLICE[1] and not overwrite_rain:
        return "skipped"
    geom = load_muni_geometry(shapefile_dir, code)
    if geom is None:
        return "no_shapefile"
    lon, lat = soy_pixel_lonlat(geom, mapbiomas_path)
    if lon.size == 0:
        return "no_soy"
    dates, date_source = resolve_s2_dates(arr, year_range, code, zonal_root, tiff_root)
    rain, daily, n_cells = rain_channels_mean_at_lonlat(
        pr_vals, time_values, x_coord, y_coord, year_range, lon, lat, dates
    )
    if not np.isfinite(rain).any():
        return "no_rain"
    updated = attach_rain(arr, rain, overwrite_rain)
    if updated is None:
        return "skipped"
    _save_npy_atomic(npy_path, updated)
    write_soy_rain_csv(
        npy_path.with_name(f"{code}_soy_rain.csv"),
        dates,
        rain,
        daily,
        int(lon.size),
        n_cells,
        date_source,
    )
    return f"C={updated.shape[2]},soy={int(lon.size)},dates={date_source}"


def main() -> None:
    args = parse_args()
    npy_root = args.npy_root.expanduser().resolve()
    shp_dir = args.shapefile_dir.expanduser().resolve()
    nc_src = args.xavier_pr_nc.expanduser().resolve()
    if not nc_src.is_file():
        raise SystemExit(f"Xavier pr NetCDF not found: {nc_src}")
    overwrite = bool(args.overwrite_rain)
    seasons = (
        [args.year_range]
        if args.year_range
        else sorted(
            p.name
            for p in npy_root.iterdir()
            if p.is_dir() and len(p.name) == 9 and p.name[4] == "-"
        )
    )
    codes = (
        {f"{int(x.strip()):07d}" for x in args.codes.split(",") if x.strip()}
        if args.codes
        else None
    )

    nc_work = copy_nc_to_temp(nc_src, prefix="xavier_pr")
    zonal_root = args.zonal_root.expanduser().resolve() if args.zonal_root else None
    tiff_root = args.tiff_root.expanduser().resolve() if args.tiff_root else None
    try:
        pr_vals, time_values, x_coord, y_coord = load_pr_grid(nc_work)
        print(f"Xavier pr {tuple(pr_vals.shape)} from {nc_src.name}")
        for year_range in seasons:
            mb = mapbiomas_url(harvest_year(year_range))
            jobs = list_jobs(npy_root, year_range, codes)
            print(f"\n=== {year_range}: {len(jobs)} cubes, MapBiomas {mb} ===")
            from collections import Counter

            counts: Counter[str] = Counter()
            for code, npy_path in tqdm(jobs, desc=year_range, unit="muni"):
                try:
                    status = process_one(
                        code,
                        npy_path,
                        shp_dir,
                        year_range,
                        mb,
                        pr_vals,
                        time_values,
                        x_coord,
                        y_coord,
                        overwrite,
                        zonal_root,
                        tiff_root,
                    )
                except Exception as exc:
                    counts["error"] += 1
                    print(f"[ERROR] {code}: {exc}")
                    continue
                key = status.split(",")[0]
                counts[key] += 1
            print(dict(counts))
    finally:
        try:
            os.unlink(nc_work)
        except OSError:
            pass


if __name__ == "__main__":
    main()
