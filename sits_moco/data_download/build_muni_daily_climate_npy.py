#!/usr/bin/env python3
"""
Build municipal daily Xavier climate sidecars for DualSTNet.

For each municipality, average Xavier cells whose centers fall in the
shapefile polygon (centroid cell if none). Writes raw daily values for the
inclusive Oct 1–Mar 31 window:

  {npy_root}/{year_range}/{code}/{code}_climate_daily.npy   shape [T, 5]
  channels: pr, ETo, Rs, Tmax, Tmin  (daily, not cumulative)

Example:
  python data_download/build_muni_daily_climate_npy.py \\
      --npy-root ~/sits_moco_data/npy_muni_mean \\
      --year-range 2020-2021 \\
      --xavier-pr-nc files/climate/pr_PR_daily_2020_2024.nc \\
      --xavier-eto-nc files/climate/ETo_PR_daily_2020_2024.nc \\
      --xavier-rs-nc files/climate/Rs_PR_daily_2020_2024.nc \\
      --xavier-tmax-nc files/climate/Tmax_PR_daily_2020_2024.nc
  # Tmin: omit --xavier-tmin-nc to densify from cum Tmin in each .npy (ch 16)
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data_download.append_xavier_rain_soy_mask import load_muni_geometry  # noqa: E402
from data_download.xavier_climate_for_daily_npy import (  # noqa: E402
    CLIMATE_VAR_ORDER,
    copy_nc_to_temp,
    load_xavier_field_arrays,
)
from data_download.xavier_rain_for_daily_npy import (  # noqa: E402
    _time_index_to_utc_date,
    load_pr_grid,
    lonlat_to_pr_indices,
)
from datasets.constants import NO_DATA_VALUE  # noqa: E402
from datasets.daily_climate import (  # noqa: E402
    N_DAILY_CLIMATE,
    climate_sidecar_path,
    climate_window_dates,
    dates_inclusive,
)
from preprocessing.preprocess_daily_to_npy import _save_npy_atomic  # noqa: E402

CHANNEL_KEYS = ("pr",) + CLIMATE_VAR_ORDER  # pr, ETo, Rs, Tmax, Tmin
DOY_CHANNEL = 10
CUM_TMIN_CHANNEL = 16  # npy ch 13:17 = ETo, Rs, Tmax, Tmin → index 16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Write {code}_climate_daily.npy municipal Xavier daily averages."
    )
    p.add_argument(
        "--npy-root",
        type=Path,
        required=True,
        help="Base dir with {year_range}/{code}/{code}.npy",
    )
    p.add_argument(
        "--year-range",
        type=str,
        required=True,
        metavar="YYYY-YYYY",
        help="Season folder label, e.g. 2020-2021",
    )
    p.add_argument(
        "--shapefile-dir",
        type=Path,
        default=_REPO / "files" / "shapefiles_pr",
        help="Municipal shapefiles (default: files/shapefiles_pr)",
    )
    p.add_argument("--xavier-pr-nc", type=Path, required=True)
    p.add_argument("--xavier-eto-nc", type=Path, required=True)
    p.add_argument("--xavier-rs-nc", type=Path, required=True)
    p.add_argument("--xavier-tmax-nc", type=Path, required=True)
    p.add_argument(
        "--xavier-tmin-nc",
        type=Path,
        default=None,
        help=(
            "Optional Xavier Tmin NetCDF. If omitted, daily Tmin is recovered from "
            "cumulative Tmin already stored in each municipality .npy (channel 16), "
            "the same values written by append_xavier_climate_to_npy.py."
        ),
    )
    p.add_argument(
        "--codes",
        type=str,
        default=None,
        help="Optional comma-separated municipality codes",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rewrite sidecars that already exist",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List municipalities without writing",
    )
    return p.parse_args()


def _list_muni_npy(npy_root: Path, year_range: str, codes: set[str] | None):
    season = npy_root / year_range
    if not season.is_dir():
        return []
    jobs = []
    for child in sorted(season.iterdir()):
        if not child.is_dir():
            continue
        code = child.name
        if codes is not None and code not in codes:
            continue
        npy = child / f"{code}.npy"
        if not npy.is_file():
            continue
        jobs.append((code, npy))
    return jobs


def _time_index_map(time_values: np.ndarray, start, end) -> dict:
    out = {}
    for ti in range(len(time_values)):
        d = _time_index_to_utc_date(time_values, ti)
        if start <= d <= end:
            out[d] = ti
    return out


def _cells_in_geom(geom, x_coord: np.ndarray, y_coord: np.ndarray):
    try:
        from shapely.vectorized import contains as shp_contains
    except Exception:
        shp_contains = None

    xx, yy = np.meshgrid(x_coord, y_coord)  # (ny, nx)
    if shp_contains is not None:
        mask = shp_contains(geom, xx, yy)
    else:
        from shapely.geometry import Point

        mask = np.zeros(xx.shape, dtype=bool)
        for iy in range(xx.shape[0]):
            for ix in range(xx.shape[1]):
                mask[iy, ix] = bool(geom.contains(Point(float(xx[iy, ix]), float(yy[iy, ix]))))
    iy, ix = np.where(mask)
    if iy.size:
        return ix.astype(np.int64), iy.astype(np.int64)
    c = geom.centroid
    jx, jy = lonlat_to_pr_indices(float(c.x), float(c.y), x_coord, y_coord)
    return np.array([jx], dtype=np.int64), np.array([jy], dtype=np.int64)


def _mean_daily_for_cells(
    vals: np.ndarray,
    date_to_ti: dict,
    days: list,
    ix: np.ndarray,
    iy: np.ndarray,
) -> np.ndarray:
    out = np.full((len(days),), np.nan, dtype=np.float32)
    ny, nx = vals.shape[1], vals.shape[2]
    keep = (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
    if not np.any(keep):
        return out
    ix_k = ix[keep]
    iy_k = iy[keep]
    cells = vals[:, iy_k, ix_k]  # (ntime, n_cells)
    with np.errstate(all="ignore"):
        series = np.nanmean(cells, axis=1)
    for t, d in enumerate(days):
        ti = date_to_ti.get(d)
        if ti is None:
            continue
        v = series[ti]
        if np.isfinite(v):
            out[t] = np.float32(v)
    return out


def _densify_cum_tmin_from_npy(
    npy_path: Path,
    days: list[date],
    season_start: date,
) -> np.ndarray:
    """
    Recover a daily Tmin series from cumulative Tmin (npy channel 16).

    Cum Tmin was built as the sum of daily Tmin from Oct 1 through each S2
    date (see xavier_climate_for_daily_npy.cumulative_channels_for_observations).
    Interval means Δcum / Δdays fill calendar days between observations.
    """
    out = np.full((len(days),), np.nan, dtype=np.float32)
    try:
        arr = np.load(npy_path, mmap_mode="r")
    except OSError:
        return out
    if arr.ndim != 3 or arr.shape[2] <= CUM_TMIN_CHANNEL:
        return out
    with np.errstate(all="ignore"):
        doy = np.nanmean(
            np.where(
                (arr[:, :, DOY_CHANNEL] == NO_DATA_VALUE)
                | ~np.isfinite(arr[:, :, DOY_CHANNEL]),
                np.nan,
                arr[:, :, DOY_CHANNEL],
            ),
            axis=0,
        )
        cum = np.nanmean(
            np.where(
                (arr[:, :, CUM_TMIN_CHANNEL] == NO_DATA_VALUE)
                | ~np.isfinite(arr[:, :, CUM_TMIN_CHANNEL]),
                np.nan,
                arr[:, :, CUM_TMIN_CHANNEL],
            ),
            axis=0,
        )
    day_index = {d: i for i, d in enumerate(days)}
    by_date: dict[date, float] = {}
    for t in range(doy.shape[0]):
        if not np.isfinite(doy[t]) or not np.isfinite(cum[t]):
            continue
        d = season_start + timedelta(days=int(doy[t]) - 1)
        if d < days[0] or d > days[-1]:
            continue
        by_date[d] = float(cum[t])
    if not by_date:
        return out
    obs = sorted(by_date.items(), key=lambda x: x[0])

    def _fill(d0: date, d1: date, mean: float) -> None:
        d = d0
        while d <= d1:
            idx = day_index.get(d)
            if idx is not None and np.isfinite(mean):
                out[idx] = np.float32(mean)
            d = d + timedelta(days=1)

    d_first, c_first = obs[0]
    n0 = (d_first - season_start).days + 1
    if n0 > 0:
        _fill(season_start, d_first, c_first / float(n0))
    for (d0, c0), (d1, c1) in zip(obs, obs[1:]):
        n = (d1 - d0).days
        if n <= 0:
            continue
        mean = (c1 - c0) / float(n)
        _fill(d0 + timedelta(days=1), d1, mean)
    return out


def main() -> None:
    args = parse_args()
    year_range = args.year_range.strip()
    start, end = climate_window_dates(year_range)
    days = dates_inclusive(start, end)
    season_start = start
    npy_root = Path(args.npy_root).expanduser()
    shp_dir = Path(args.shapefile_dir).expanduser()
    if not shp_dir.is_absolute():
        shp_dir = (_REPO / shp_dir).resolve()

    codes = None
    if args.codes:
        codes = {c.strip() for c in args.codes.split(",") if c.strip()}
    jobs = _list_muni_npy(npy_root, year_range, codes)
    if not jobs:
        raise SystemExit(f"No {{code}}/{{code}}.npy under {npy_root / year_range}")

    tmin_from_npy = args.xavier_tmin_nc is None
    print(
        f"Daily climate sidecars for {len(jobs)} municipalities, "
        f"{year_range} ({days[0]}–{days[-1]}, T={len(days)})"
    )
    if tmin_from_npy:
        print(
            "  Tmin: densified from cumulative channel 16 in each .npy "
            "(no --xavier-tmin-nc)"
        )

    tmp_paths: list[Path] = []
    try:
        pr_tmp = copy_nc_to_temp(Path(args.xavier_pr_nc), prefix="xavier_pr")
        tmp_paths.append(pr_tmp)
        pr_vals, pr_time, x_coord, y_coord = load_pr_grid(pr_tmp)
        cubes: list[tuple[str, np.ndarray, dict]] = [
            ("pr", pr_vals, _time_index_map(pr_time, start, end))
        ]
        nc_map = {
            "ETo": args.xavier_eto_nc,
            "Rs": args.xavier_rs_nc,
            "Tmax": args.xavier_tmax_nc,
        }
        if not tmin_from_npy:
            nc_map["Tmin"] = args.xavier_tmin_nc
        for key in ("ETo", "Rs", "Tmax") + (("Tmin",) if not tmin_from_npy else ()):
            tmp = copy_nc_to_temp(Path(nc_map[key]), prefix=f"xavier_{key.lower()}")
            tmp_paths.append(tmp)
            vals, time_values, _x, _y = load_xavier_field_arrays(tmp, key)
            cubes.append((key, vals, _time_index_map(time_values, start, end)))

        # Channel order in output: pr, ETo, Rs, Tmax, Tmin
        name_to_cube = {name: (vals, dmap) for name, vals, dmap in cubes}

        counts: dict[str, int] = {}
        for code, npy_path in tqdm(jobs, desc="Climate sidecars", unit="muni"):
            dest = climate_sidecar_path(npy_path)
            if dest.is_file() and not args.force:
                counts["skip"] = counts.get("skip", 0) + 1
                continue
            geom = load_muni_geometry(shp_dir, code)
            if geom is None:
                counts["no_shapefile"] = counts.get("no_shapefile", 0) + 1
                continue
            if args.dry_run:
                counts["dry"] = counts.get("dry", 0) + 1
                continue
            ix, iy = _cells_in_geom(geom, x_coord, y_coord)
            block = np.full((len(days), N_DAILY_CLIMATE), NO_DATA_VALUE, dtype=np.float32)
            for ch, name in enumerate(CHANNEL_KEYS):
                if name == "Tmin" and tmin_from_npy:
                    series = _densify_cum_tmin_from_npy(npy_path, days, season_start)
                else:
                    vals, dmap = name_to_cube[name]
                    series = _mean_daily_for_cells(vals, dmap, days, ix, iy)
                block[:, ch] = np.where(np.isfinite(series), series, NO_DATA_VALUE)
            dest.parent.mkdir(parents=True, exist_ok=True)
            _save_npy_atomic(dest, block)
            counts["ok"] = counts.get("ok", 0) + 1
        print("Done:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        print(f"Channels {CHANNEL_KEYS} → [T={len(days)}, {N_DAILY_CLIMATE}]")
    finally:
        for p in tmp_paths:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass


if __name__ == "__main__":
    main()
