"""
Xavier climate fields (ETo, Rs, Tmax, Tmin) as cumulative channels aligned to daily .npy.

Builds on data_download/xavier_rain_for_daily_npy.py (same grid, climate window, NaN rules).

Per S2 timestep k, each field is the running sum from Oct 1 through observation date t_k
(inclusive; NaN days omitted from the sum).

Channel order when stacked after rain (npy 13:17):
  0 = cum ETo (mm)
  1 = cum Rs (MJ/m^2)
  2 = cum Tmax (°C·day)
  3 = cum Tmin (°C·day)
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr

from data_download.xavier_rain_for_daily_npy import (
    _cumulative_to_date,
    build_pr_lookup_for_cell,
    climate_window_dates,
    lonlat_to_pr_indices,
    pixel_centroids_lonlat,
)

N_CLIMATE_CHANNELS = 4
CLIMATE_VAR_ORDER = ("ETo", "Rs", "Tmax", "Tmin")

# NetCDF variable-name aliases (Xavier / BR-DWGD naming varies slightly).
_VAR_ALIASES: dict[str, tuple[str, ...]] = {
    "ETo": ("ETo", "eto", "Eto", "ET0", "et0"),
    "Rs": ("Rs", "RS", "rs"),
    "Tmax": ("Tmax", "tmax", "TMAX"),
    "Tmin": ("Tmin", "tmin", "TMIN"),
}


def resolve_var_name(ds: xr.Dataset, canonical: str) -> str:
    aliases = _VAR_ALIASES.get(canonical, (canonical,))
    for name in aliases:
        if name in ds.data_vars or name in ds.variables:
            return name
    available = list(ds.data_vars)
    raise KeyError(
        f"Xavier dataset missing {canonical!r} (tried {aliases}). "
        f"Available data_vars: {available}"
    )


def load_xavier_field_arrays(
    nc_path: Path,
    canonical_var: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Open one Xavier NetCDF and return (values[time,y,x], time_values, x_coord, y_coord).

    ``nc_path`` should be a local ASCII path (temp copy) when the source lives on
    OneDrive / Unicode paths — same constraint as rain.
    """
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    p = Path(nc_path)
    if not p.is_file():
        raise FileNotFoundError(f"Xavier work copy missing: {p}")

    with xr.open_dataset(os.fspath(p), decode_times=True, engine="netcdf4") as ds:
        var_name = resolve_var_name(ds, canonical_var)
        da = ds[var_name]
        if da.ndim != 3:
            raise ValueError(f"Expected 3D {var_name}, got shape {da.shape}")
        da = da.squeeze()
        if "time" not in da.dims:
            raise ValueError(f"{var_name} dims {da.dims}: expected a 'time' dimension")
        dim_names = list(da.dims)
        if dim_names[0] != "time":
            da = da.transpose("time", *[d for d in dim_names if d != "time"])
        values = np.asarray(da.values, dtype=np.float32)
        time_values = np.asarray(ds["time"].values)
        x_da = ds["x"] if "x" in ds.coords else ds["longitude"]
        y_da = ds["y"] if "y" in ds.coords else ds["latitude"]
        x_coord = np.asarray(x_da.values, dtype=np.float64)
        y_coord = np.asarray(y_da.values, dtype=np.float64)
    return values, time_values, x_coord, y_coord


def cumulative_channels_for_observations(
    obs_dates: list[date],
    oct1: date,
    field_by_date: dict[date, float],
) -> np.ndarray:
    """Running cumulative from oct1 through each obs date. Shape (T,)."""
    t = len(obs_dates)
    cum = np.zeros(t, dtype=np.float32)
    if t == 0:
        return cum
    for k, d in enumerate(obs_dates):
        cum[k] = float(_cumulative_to_date(field_by_date, oct1, d))
    return cum


def compute_climate_block_for_municipality(
    nc_paths: dict[str, Path],
    year_range: str,
    rows: np.ndarray,
    cols: np.ndarray,
    transform,
    src_crs,
    obs_dates: list[date],
) -> np.ndarray:
    """
    Build [N, T, 4] climate features: cum ETo, cum Rs, cum Tmax, cum Tmin.

    ``nc_paths`` maps canonical names (ETo, Rs, Tmax, Tmin) to local NetCDF paths.
    Missing cell / load error → NaN for that pixel.
    """
    n = len(rows)
    t = len(obs_dates)
    out = np.full((n, t, N_CLIMATE_CHANNELS), np.nan, dtype=np.float32)
    if n == 0 or t == 0:
        return out

    for key in CLIMATE_VAR_ORDER:
        if key not in nc_paths or nc_paths[key] is None:
            raise ValueError(f"nc_paths missing required key {key!r}")

    window_start, window_end = climate_window_dates(year_range)
    oct1 = window_start
    lon, lat = pixel_centroids_lonlat(rows, cols, transform, src_crs)

    for ch, canonical in enumerate(CLIMATE_VAR_ORDER):
        values, time_values, x_coord, y_coord = load_xavier_field_arrays(
            Path(nc_paths[canonical]), canonical
        )

        cell_to_indices: dict[tuple[int, int], list[int]] = {}
        for i in range(n):
            try:
                ix, iy = lonlat_to_pr_indices(
                    float(lon[i]), float(lat[i]), x_coord, y_coord
                )
            except Exception:
                continue
            cell_to_indices.setdefault((ix, iy), []).append(i)

        ny, nx = values.shape[1], values.shape[2]
        for (ix, iy), idx_list in cell_to_indices.items():
            if iy < 0 or iy >= ny or ix < 0 or ix >= nx:
                continue
            try:
                by_date = build_pr_lookup_for_cell(
                    values, time_values, ix, iy, window_start, window_end
                )
                cum = cumulative_channels_for_observations(obs_dates, oct1, by_date)
                for i in idx_list:
                    out[i, :, ch] = cum
            except Exception:
                continue

    return out


def copy_nc_to_temp(src: Path, *, prefix: str) -> Path:
    """Copy NetCDF to a local temp file (ASCII path) for netCDF4 / OneDrive safety."""
    import tempfile
    import uuid

    src = Path(src).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Not a file: {src}")
    blob = src.read_bytes()
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
