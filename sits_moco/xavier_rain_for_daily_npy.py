"""
Xavier daily precipitation (NetCDF) aligned to daily municipal .npy time series.

Spec (see project discussion):
- Spatial: rain cell that contains each Sentinel pixel centroid (WGS84 lon/lat).
- Calendar: UTC calendar days consistent with GEE export / TIFF filenames.
- Climate window: [Oct 1 of year_range first year, Mar 31 of second year], both inclusive.
- Per S2 timestep k: channel 0 = cumulative mm from Oct 1 through observation date t_k
  (inclusive, clipped to climate window; NaN pr days omitted from sum, no propagation).
- Per timestep k: channel 1 = max consecutive dry days in (t_{k-1}, t_k]; k=0 uses
  Oct 1 .. t_0 inclusive as first interval. Dry = pr < 1 mm; NaN breaks a dry run.
- No rain metrics after the last S2 date (not needed for trailing tail).
"""

from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


def _ensure_pyproj_proj_data() -> None:
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
import pandas as pd
import pyproj
import rasterio
import xarray as xr


EPOCH_1961 = datetime(1961, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
DRY_MM = 1.0
N_RAIN_CHANNELS = 2


def parse_year_range_bounds(year_range: str) -> tuple[int, int]:
    """Return (y1, y2) from 'YYYY-YYYY'."""
    m = re.match(r"^(\d{4})-(\d{4})$", year_range.strip())
    if not m:
        raise ValueError(f"Expected year-range like 2020-2021, got {year_range!r}")
    return int(m.group(1)), int(m.group(2))


def climate_window_dates(year_range: str) -> tuple[date, date]:
    """Inclusive [Oct 1 y1, Mar 31 y2] for folder label y1-y2."""
    y1, y2 = parse_year_range_bounds(year_range)
    return date(y1, 10, 1), date(y2, 3, 31)


def _utc_date_to_hours_since_1961(d: date) -> float:
    dt = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
    return (dt - EPOCH_1961).total_seconds() / 3600.0


def _grid_index_containing_1d(cell_centers: np.ndarray, v: float) -> int:
    """
    cell_centers: 1D strictly monotonic cell-center coordinates.
    Return index of the cell that contains scalar v (edges from midpoints between centers).
    """
    c = np.asarray(cell_centers, dtype=np.float64)
    n = len(c)
    if n == 0:
        raise ValueError("empty coordinate axis")
    if n == 1:
        return 0
    if c[1] < c[0]:
        # Decreasing (e.g. latitude north-to-south): search on reversed axis.
        return (n - 1) - _grid_index_containing_1d(c[::-1], v)
    left = float(c[0] - (c[1] - c[0]) / 2.0)
    mids = (c[:-1] + c[1:]) / 2.0
    last_r = float(c[-1] + (c[-1] - c[-2]) / 2.0)
    edges = np.concatenate([[left], mids, [last_r]])
    j = int(np.searchsorted(edges, v, side="right") - 1)
    return int(np.clip(j, 0, n - 1))


def lonlat_to_pr_indices(lon: float, lat: float, x_coord: np.ndarray, y_coord: np.ndarray) -> tuple[int, int]:
    """Return (ix, iy) for pr[time, iy, ix] with dims (time, y, x) as in Xavier files."""
    ix = _grid_index_containing_1d(x_coord, lon)
    iy = _grid_index_containing_1d(y_coord, lat)
    return ix, iy


def pixel_centroids_lonlat(
    rows: np.ndarray,
    cols: np.ndarray,
    transform: rasterio.Affine,
    src_crs,
) -> tuple[np.ndarray, np.ndarray]:
    """Centers (lon, lat) WGS84 for raster integer (row, col) indices."""
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    xs = np.empty_like(rows, dtype=np.float64)
    ys = np.empty_like(rows, dtype=np.float64)
    for i in range(len(rows)):
        x, y = transform * (cols[i] + 0.5, rows[i] + 0.5)
        xs[i], ys[i] = x, y
    if src_crs is None or str(src_crs).upper() == "EPSG:4326":
        return xs, ys
    t = pyproj.Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    lon, lat = t.transform(xs, ys)
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def _dates_inclusive(start: date, end: date) -> list[date]:
    out: list[date] = []
    d = start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def _max_consecutive_dry(pr_values: np.ndarray, dry_mm: float = DRY_MM) -> int:
    """
    pr_values: 1D aligned to consecutive calendar days in an interval.
    NaN: breaks dry run (does not count as dry). Returns max run length.
    """
    best = 0
    cur = 0
    for v in pr_values:
        if not np.isfinite(v):
            cur = 0
            continue
        if v < dry_mm:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _cumulative_to_date(pr_by_date: dict[date, float], oct1: date, through: date) -> float:
    s = 0.0
    for d in _dates_inclusive(oct1, through):
        v = pr_by_date.get(d)
        if v is None or not np.isfinite(v):
            continue
        s += float(v)
    return s


def _interval_dates_exclusive_start_closed_end(prev_d: date, curr_d: date) -> list[date]:
    """(prev_d, curr_d] as calendar days: strictly after prev_d up to curr_d inclusive."""
    out: list[date] = []
    d = prev_d + timedelta(days=1)
    while d <= curr_d:
        out.append(d)
        d += timedelta(days=1)
    return out


def _time_index_to_utc_date(time_values: np.ndarray, index: int) -> date:
    """One timestep -> UTC calendar date."""
    tv = time_values[index]
    if isinstance(tv, np.datetime64):
        ts = np.datetime_as_string(tv, "D")
        y, m, d = (int(x) for x in ts.split("-"))
        return date(y, m, d)
    try:
        return pd.Timestamp(tv).date()
    except Exception:
        pass
    # numeric offset: assume hours since 1961-01-01 00:00 UTC (Xavier convention)
    h = float(tv)
    return (EPOCH_1961 + timedelta(hours=h)).date()


def build_pr_lookup_for_cell(
    pr_time_yx: np.ndarray,
    time_values: np.ndarray,
    ix: int,
    iy: int,
    window_start: date,
    window_end: date,
) -> dict[date, float]:
    """
    pr_time_yx: (n_time, ny, nx)
    time_values: raw time coordinate values (datetime64 or hours offset).
    Map each UTC calendar day in [window_start, window_end] to pr (last wins if duplicates).
    """
    series = pr_time_yx[:, iy, ix].astype(np.float64)
    out: dict[date, float] = {}
    for ti in range(len(time_values)):
        d = _time_index_to_utc_date(time_values, ti)
        if d < window_start or d > window_end:
            continue
        v = series[ti]
        out[d] = float(v) if np.isfinite(v) else np.nan
    return out


def rain_channels_for_observations(
    obs_dates: list[date],
    oct1: date,
    pr_by_date: dict[date, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (running_cumulative_mm, max_dry_streak) each shape (T,) for T = len(obs_dates).
    Cumulative sums only days present in pr_by_date within each [oct1, t_k] span.
    """
    t = len(obs_dates)
    cum = np.zeros(t, dtype=np.float32)
    dry_streak = np.zeros(t, dtype=np.float32)
    if t == 0:
        return cum, dry_streak

    # First interval: calendar days from Oct1 through obs_dates[0] inclusive
    # Equivalent to (day before Oct1, obs_dates[0]]
    first_days = _dates_inclusive(oct1, obs_dates[0])
    pr_first = np.array([pr_by_date.get(d, np.nan) for d in first_days], dtype=np.float64)
    dry_streak[0] = float(_max_consecutive_dry(pr_first))
    cum[0] = float(_cumulative_to_date(pr_by_date, oct1, obs_dates[0]))

    for k in range(1, t):
        prev_d = obs_dates[k - 1]
        curr_d = obs_dates[k]
        interval = _interval_dates_exclusive_start_closed_end(prev_d, curr_d)
        pr_iv = np.array([pr_by_date.get(d, np.nan) for d in interval], dtype=np.float64)
        dry_streak[k] = float(_max_consecutive_dry(pr_iv))
        cum[k] = float(_cumulative_to_date(pr_by_date, oct1, curr_d))

    return cum, dry_streak


def compute_rain_block_for_municipality(
    nc_path: Path,
    year_range: str,
    rows: np.ndarray,
    cols: np.ndarray,
    transform,
    src_crs,
    obs_dates: list[date],
) -> np.ndarray:
    """
    Build [N, T, 2] rain features aligned to final .npy pixel order.
    Missing cell or load error: fills with NaN.

    ``nc_path`` should be a **local ASCII path** (e.g. temp file under %%TEMP%%). The parent
    copies the user’s NetCDF there because OneDrive + Unicode paths often make ``is_file()``
    succeed while the NetCDF C library returns errno 2 on open.
    """
    n = len(rows)
    t = len(obs_dates)
    out = np.full((n, t, N_RAIN_CHANNELS), np.nan, dtype=np.float32)
    if n == 0 or t == 0:
        return out

    window_start, window_end = climate_window_dates(year_range)
    oct1 = window_start

    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    p = Path(nc_path)
    if not p.is_file():
        raise FileNotFoundError(f"Xavier work copy missing: {p}")

    with xr.open_dataset(
        os.fspath(p), decode_times=True, engine="netcdf4"
    ) as ds:
        pr = ds["pr"]
        if pr.ndim != 3:
            raise ValueError(f"Expected 3D pr, got shape {pr.shape}")
        da = pr.squeeze()
        if "time" not in da.dims:
            raise ValueError(f"pr dims {da.dims}: expected a 'time' dimension")
        dim_names = list(da.dims)
        if dim_names[0] != "time":
            da = da.transpose("time", *[d for d in dim_names if d != "time"])
        pr_vals = np.asarray(da.values, dtype=np.float32)
        time_values = np.asarray(ds["time"].values)

        x_da = ds["x"] if "x" in ds.coords else ds["longitude"]
        y_da = ds["y"] if "y" in ds.coords else ds["latitude"]
        x_coord = np.asarray(x_da.values, dtype=np.float64)
        y_coord = np.asarray(y_da.values, dtype=np.float64)

    lon, lat = pixel_centroids_lonlat(rows, cols, transform, src_crs)

    # Unique cells -> compute once
    cell_to_indices: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        try:
            ix, iy = lonlat_to_pr_indices(float(lon[i]), float(lat[i]), x_coord, y_coord)
        except Exception:
            continue
        key = (ix, iy)
        cell_to_indices.setdefault(key, []).append(i)

    for (ix, iy), idx_list in cell_to_indices.items():
        if iy < 0 or iy >= pr_vals.shape[1] or ix < 0 or ix >= pr_vals.shape[2]:
            continue
        try:
            pr_by_date = build_pr_lookup_for_cell(
                pr_vals, time_values, ix, iy, window_start, window_end
            )
            cum, dsk = rain_channels_for_observations(obs_dates, oct1, pr_by_date)
            for i in idx_list:
                out[i, :, 0] = cum.astype(np.float32)
                out[i, :, 1] = dsk.astype(np.float32)
        except Exception:
            continue

    return out
