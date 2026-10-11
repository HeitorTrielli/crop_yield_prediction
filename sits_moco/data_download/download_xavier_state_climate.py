#!/usr/bin/env python3
"""
Clip BR-DWGD Xavier climate NetCDFs to one or more Brazilian states.

Writes files/climate/{var}_{UF}_daily_{Y1}_{Y2}.nc for each requested variable:
  pr, ETo, Rs, Tmax, Tmin

Packing / attrs mirror the existing Paraná files under files/climate/ when present;
otherwise float32 zlib is used. Spatial grid is the state bbox clip (not PR grid).

Example (Rio Grande do Sul, DualSTNet window):
  python data_download/download_xavier_state_climate.py \\
      --states rs --year-range 2019-2024 --download --skip-existing
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

_DD = Path(__file__).resolve().parent
_REPO = _DD.parent
if str(_DD) not in sys.path:
    sys.path.insert(0, str(_DD))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from download_pam_soy_sidra import UF_BY_CODE, parse_state_codes  # noqa: E402
from download_xavier_state_rain import (  # noqa: E402
    BR_DWGD_DRIVE_FOLDER_ID,
    BBOX_PAD_DEG,
    FOLDER_MIME,
    clip_bbox,
    list_drive_tree,
    state_bboxes,
)
from xavier_rain_for_daily_npy import parse_year_range_bounds  # noqa: E402

CLIMATE_VARS = ("pr", "ETo", "Rs", "Tmax", "Tmin")
_FILE_STEM_RE = re.compile(
    r"^(?P<var>pr|ETo|Rs|Tmax|Tmin|RH|u2)_(?P<start>\d{8})_(?P<end>\d{8})",
    re.IGNORECASE,
)
_VAR_ALIASES: dict[str, tuple[str, ...]] = {
    "pr": ("pr", "PR", "precip"),
    "ETo": ("ETo", "eto", "ETO", "ET0", "et0"),
    "Rs": ("Rs", "rs", "RS", "Rad", "rad"),
    "Tmax": ("Tmax", "tmax", "TMAX"),
    "Tmin": ("Tmin", "tmin", "TMIN"),
}
# Fallback packing when no PR template exists for that var
_FALLBACK_ENC: dict[str, dict] = {
    "pr": {
        "dtype": "int16",
        "scale_factor": 0.006866664632099367,
        "add_offset": 224.99999999999997,
        "_FillValue": np.int16(-32768),
    },
    "ETo": {
        "dtype": "uint8",
        "scale_factor": 0.05118110236220472,
        "add_offset": -0.05118110236220327,
        "_FillValue": np.uint8(0),
    },
    "Rs": {
        "dtype": "uint8",
        "scale_factor": 0.15708661417322836,
        "add_offset": -0.05708661417322806,
        "_FillValue": np.uint8(0),
    },
    "Tmax": {
        "dtype": "int16",
        "scale_factor": 0.0010681478316599017,
        "add_offset": 15.0,
        "_FillValue": np.int16(-32768),
    },
    "Tmin": {
        "dtype": "int16",
        "scale_factor": 0.0010681478316599017,
        "add_offset": 15.0,
        "_FillValue": np.int16(-32768),
    },
}
_UNITS = {
    "pr": "mm",
    "ETo": "mm",
    "Rs": "MJ m-2 day-1",
    "Tmax": "Celcius degrees",
    "Tmin": "Celcius degrees",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--states",
        type=str,
        default="rs",
        help="UF list, e.g. rs or mt,go (default: rs)",
    )
    p.add_argument(
        "--vars",
        type=str,
        default=",".join(CLIMATE_VARS),
        help=f"Comma-separated vars (default: {','.join(CLIMATE_VARS)})",
    )
    p.add_argument(
        "--year-range",
        type=str,
        default="2019-2024",
        metavar="YYYY-YYYY",
        help="Calendar years in output filename (default 2019-2024)",
    )
    p.add_argument(
        "--shapefile-dir",
        type=Path,
        default=_REPO / "files" / "shapefiles",
    )
    p.add_argument(
        "--nc-dir",
        type=Path,
        default=_REPO / "files" / "xavier_nc",
        help="Cache for national BR-DWGD NetCDFs / zips",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO / "files" / "climate",
    )
    p.add_argument(
        "--template-dir",
        type=Path,
        default=_REPO / "files" / "climate",
        help="Directory with {var}_PR_daily_*.nc packing templates",
    )
    p.add_argument("--download", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--credentials-dir",
        type=Path,
        default=_REPO / "data",
    )
    return p.parse_args()


def _parse_stem(path: Path) -> tuple[str, date, date] | None:
    m = _FILE_STEM_RE.match(path.stem)
    if not m:
        return None
    return (
        m.group("var"),
        date(int(m.group("start")[:4]), int(m.group("start")[4:6]), int(m.group("start")[6:8])),
        date(int(m.group("end")[:4]), int(m.group("end")[4:6]), int(m.group("end")[6:8])),
    )


def find_local_national(nc_dir: Path, var: str, start: date, end: date) -> Path | None:
    ranked: list[tuple[int, Path]] = []
    aliases = {a.lower() for a in _VAR_ALIASES[var]}
    for path in nc_dir.rglob("*"):
        if path.suffix.lower() not in {".nc", ".nc4"}:
            continue
        parsed = _parse_stem(path)
        if parsed is None:
            if path.name.lower().startswith(var.lower()):
                ranked.append((10**9, path))
            continue
        v, lo, hi = parsed
        if v.lower() not in aliases:
            continue
        if lo <= start and hi >= end:
            ranked.append(((hi - lo).days, path))
    if not ranked:
        return None
    ranked.sort(key=lambda x: x[0])
    return ranked[0][1]


def pick_drive_item(files: list[dict], var: str, start: date, end: date) -> dict | None:
    aliases = {a.lower() for a in _VAR_ALIASES[var]}
    ranked: list[tuple[int, dict]] = []
    zip_fallback: dict | None = None
    for item in files:
        name = str(item.get("name") or "")
        path = Path(name)
        suffix = path.suffix.lower()
        parsed = _parse_stem(path)
        if parsed is not None and suffix in {".nc", ".nc4"}:
            v, lo, hi = parsed
            if v.lower() in aliases and lo <= start and hi >= end:
                ranked.append(((hi - lo).days, item))
            continue
        stem = path.stem.lower()
        if suffix == ".zip":
            # pr_Tmax_Tmin zip covers pr/Tmax/Tmin; other zips named by var
            if var in ("pr", "Tmax", "Tmin") and stem.startswith("pr") and "control" not in stem:
                zip_fallback = item
            elif var.lower() in stem:
                zip_fallback = item
    if ranked:
        ranked.sort(key=lambda x: (x[0], x[1].get("name") or ""))
        return ranked[0][1]
    return zip_fallback


def download_file(service, item: dict, dest: Path) -> Path:
    from googleapiclient.http import MediaIoBaseDownload

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = service.files().get_media(fileId=item["id"])
    size = int(item["size"]) if str(item.get("size") or "").isdigit() else None
    print(f"Downloading {item.get('name')} ({item.get('size', '?')} bytes)")
    with open(tmp, "wb") as f:
        downloader = MediaIoBaseDownload(f, request, chunksize=16 * 1024 * 1024)
        done = False
        pbar = tqdm(
            total=size, unit="B", unit_scale=True, desc=str(item.get("name"))[:40]
        )
        progress = 0
        while not done:
            status, done = downloader.next_chunk()
            if status is not None:
                nxt = int(status.resumable_progress)
                pbar.update(max(0, nxt - progress))
                progress = nxt
        pbar.close()
    tmp.replace(dest)
    return dest


def extract_var_from_zip(
    zip_path: Path, dest_dir: Path, var: str, start: date, end: date
) -> Path | None:
    aliases = {a.lower() for a in _VAR_ALIASES[var]}
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            n
            for n in zf.namelist()
            if Path(n).suffix.lower() in {".nc", ".nc4"}
            and any(Path(n).name.lower().startswith(a) for a in aliases)
        ]
        if not members:
            print(f"  zip {zip_path.name} has no {var}_*.nc")
            return None
        chosen: list[str] = []
        for name in members:
            parsed = _parse_stem(Path(name))
            if parsed is None:
                chosen.append(name)
                continue
            v, lo, hi = parsed
            if v.lower() in aliases and lo <= start and hi >= end:
                chosen.append(name)
        if not chosen:
            chosen = members

        def span_key(name: str) -> int:
            parsed = _parse_stem(Path(name))
            if parsed is None:
                return 10**9
            _v, lo, hi = parsed
            return (hi - lo).days

        chosen.sort(key=span_key)
        member = chosen[0]
        out_name = Path(member).name
        dest = dest_dir / out_name
        if dest.is_file():
            print(f"  already extracted {out_name}")
            return dest
        print(f"  extracting {member} from {zip_path.name}")
        with zf.open(member) as src, open(dest, "wb") as dst:
            while True:
                chunk = src.read(16 * 1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
        return dest


def ensure_national_nc(
    var: str,
    nc_dir: Path,
    start: date,
    end: date,
    credentials_dir: Path,
    download: bool,
    drive_files: list[dict] | None,
) -> tuple[Path, list[dict] | None]:
    local = find_local_national(nc_dir, var, start, end)
    if local is not None:
        print(f"National {var} on disk: {local}")
        return local, drive_files
    if not download:
        raise SystemExit(
            f"No national {var} NetCDF covering {start}–{end} in {nc_dir}. "
            "Re-run with --download."
        )
    from drive_api import build_drive_service

    if drive_files is None:
        service = build_drive_service(credentials_dir)
        print(f"Listing public BR-DWGD Drive folder {BR_DWGD_DRIVE_FOLDER_ID}...")
        drive_files = list_drive_tree(service, BR_DWGD_DRIVE_FOLDER_ID)
        for item in drive_files:
            print(f"  {item.get('name')} ({item.get('size', '?')} bytes)")
    else:
        service = build_drive_service(credentials_dir)

    item = pick_drive_item(drive_files, var, start, end)
    if item is None:
        raise SystemExit(f"No Drive item for {var} covering {start}–{end}")
    name = str(item.get("name") or "download")
    dest = nc_dir / Path(name).name
    if not dest.is_file():
        download_file(service, item, dest)
    if dest.suffix.lower() == ".zip":
        extracted = extract_var_from_zip(dest, nc_dir, var, start, end)
        if extracted is None:
            raise SystemExit(f"Downloaded {dest.name} but found no {var}_*.nc")
        return extracted, drive_files
    return dest, drive_files


def load_var_window(nc_path: Path, var: str, start: date, end: date) -> xr.DataArray:
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    aliases = _VAR_ALIASES[var]
    with xr.open_dataset(os.fspath(nc_path), decode_times=True, engine="netcdf4") as ds:
        var_name = next(
            (n for n in aliases if n in ds.data_vars or n in ds.variables),
            None,
        )
        if var_name is None:
            raise KeyError(f"{nc_path} has no {var}; data_vars={list(ds.data_vars)}")
        da = ds[var_name]
        rename = {}
        for src, dst in (
            ("longitude", "x"),
            ("lon", "x"),
            ("latitude", "y"),
            ("lat", "y"),
        ):
            if src in da.coords or src in da.dims:
                rename[src] = dst
        if rename:
            da = da.rename(rename)
        if "x" not in da.dims or "y" not in da.dims:
            raise ValueError(f"{nc_path}: expected x/y dims, got {da.dims}")
        if "time" not in da.dims:
            raise ValueError(f"{nc_path}: expected time dim, got {da.dims}")
        da = da.transpose("time", "y", "x")
        sliced = da.sel(time=slice(np.datetime64(start), np.datetime64(end)))
        loaded = sliced.load()
    if loaded.sizes.get("time", 0) == 0:
        raise SystemExit(f"{nc_path}: no timesteps in {start}–{end}")
    loaded = loaded.astype(np.float32)
    loaded.name = var
    return loaded


def find_template(template_dir: Path, var: str) -> Path | None:
    matches = sorted(template_dir.glob(f"{var}_PR_daily_*.nc"))
    return matches[0] if matches else None


def save_clipped(
    da: xr.DataArray, dest: Path, var: str, template: Path | None
) -> None:
    y = da["y"]
    if float(y[0]) > float(y[-1]):
        da = da.isel(y=slice(None, None, -1))
    ds = da.to_dataset(name=var)
    enc_base = dict(_FALLBACK_ENC[var])
    spatial = None
    spatial_attrs: dict = {}
    if template is not None and template.is_file():
        with xr.open_dataset(os.fspath(template)) as tmpl:
            if var in tmpl:
                src = tmpl[var]
                for k in ("dtype", "scale_factor", "add_offset", "_FillValue"):
                    if k in src.encoding:
                        enc_base[k] = src.encoding[k]
            if "spatial_ref" in tmpl:
                spatial = tmpl["spatial_ref"]
                spatial_attrs = dict(tmpl["spatial_ref"].attrs)
    ds[var].attrs.update(
        units=_UNITS.get(var, ""),
        standard_name=var,
        grid_mapping="spatial_ref",
    )
    if spatial is not None:
        ds["spatial_ref"] = spatial
        ds["spatial_ref"].attrs.update(spatial_attrs)
    encoding = {
        var: {
            **enc_base,
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
        }
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    ds.to_netcdf(tmp, encoding=encoding)
    tmp.replace(dest)


def climate_out_path(output_dir: Path, var: str, abbr: str, y1: int, y2: int) -> Path:
    return output_dir / f"{var}_{abbr}_daily_{y1}_{y2}.nc"


def main() -> None:
    args = parse_args()
    y1, y2 = parse_year_range_bounds(args.year_range.strip())
    start, end = date(y1, 1, 1), date(y2, 12, 31)
    uf_codes = parse_state_codes(args.states)
    vars_req = [v.strip() for v in args.vars.split(",") if v.strip()]
    for v in vars_req:
        if v not in CLIMATE_VARS:
            raise SystemExit(f"Unknown var {v!r}; choose from {CLIMATE_VARS}")

    nc_dir = args.nc_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    template_dir = args.template_dir.expanduser().resolve()
    creds = args.credentials_dir.expanduser().resolve()
    nc_dir.mkdir(parents=True, exist_ok=True)

    labels = ", ".join(f"{UF_BY_CODE[u]}({u})" for u in uf_codes)
    print(f"States: {labels}")
    print(f"Vars: {', '.join(vars_req)}")
    print(f"Window: {start}–{end} → {output_dir}/{{var}}_{{UF}}_daily_{y1}_{y2}.nc")

    bboxes = state_bboxes(uf_codes, args.shapefile_dir.expanduser().resolve())
    if args.dry_run:
        for uf in uf_codes:
            abbr = UF_BY_CODE[uf]
            bbox = bboxes[uf]
            for var in vars_req:
                dest = climate_out_path(output_dir, var, abbr, y1, y2)
                print(
                    f"  {abbr}/{var}: bbox lon[{bbox[0]:.2f},{bbox[2]:.2f}] "
                    f"lat[{bbox[1]:.2f},{bbox[3]:.2f}] → {dest.name}"
                )
        return

    drive_files: list[dict] | None = None
    n_ok = 0
    n_skip = 0
    for var in vars_req:
        nc_path, drive_files = ensure_national_nc(
            var, nc_dir, start, end, creds, bool(args.download), drive_files
        )
        print(f"Loading {var} {start}–{end} from {nc_path.name}...")
        da = load_var_window(nc_path, var, start, end)
        print(f"  national window shape {tuple(da.shape)}")
        template = find_template(template_dir, var)
        for uf in uf_codes:
            abbr = UF_BY_CODE[uf]
            dest = climate_out_path(output_dir, var, abbr, y1, y2)
            if args.skip_existing and dest.is_file():
                print(f"  {abbr}/{var}: skip existing {dest.name}")
                n_skip += 1
                continue
            lon_min, lat_min, lon_max, lat_max = bboxes[uf]
            clipped = clip_bbox(da, lon_min, lat_min, lon_max, lat_max)
            if clipped.sizes.get("x", 0) == 0 or clipped.sizes.get("y", 0) == 0:
                raise SystemExit(f"{abbr}/{var}: empty clip for bbox {bboxes[uf]}")
            save_clipped(clipped, dest, var, template)
            print(
                f"  {abbr}/{var}: wrote {dest.name} "
                f"{tuple(clipped.shape)} y×x="
                f"{clipped.sizes['y']}×{clipped.sizes['x']}"
            )
            n_ok += 1

    print(f"Done: wrote={n_ok} skipped={n_skip}")


if __name__ == "__main__":
    main()
