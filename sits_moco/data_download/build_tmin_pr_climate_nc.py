#!/usr/bin/env python3
"""
Build files/climate/Tmin_PR_daily_2020_2024.nc to match Tmax_PR_daily_2020_2024.nc.

Source: same public BR-DWGD Drive folder used for national pr
(pr_Tmax_Tmin_NetCDF_Files.zip → Tmin_*.nc), clipped/aligned to the existing
Tmax grid and calendar window (2020-01-01 … 2024-12-31).

Example:
  python data_download/build_tmin_pr_climate_nc.py --download
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

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_DD = Path(__file__).resolve().parent
if str(_DD) not in sys.path:
    sys.path.insert(0, str(_DD))

from download_xavier_state_rain import (  # noqa: E402
    BR_DWGD_DRIVE_FOLDER_ID,
    list_drive_tree,
)
from drive_api import build_drive_service  # noqa: E402

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

_FILE_STEM_RE = re.compile(
    r"^(?P<var>pr|ETo|Rs|Tmax|Tmin|RH|u2)_(?P<start>\d{8})_(?P<end>\d{8})",
    re.IGNORECASE,
)
TEMPLATE = _REPO / "files" / "climate" / "Tmax_PR_daily_2020_2024.nc"
OUT = _REPO / "files" / "climate" / "Tmin_PR_daily_2020_2024.nc"
NC_DIR = _REPO / "files" / "xavier_nc"
START = date(2020, 1, 1)
END = date(2024, 12, 31)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--download",
        action="store_true",
        help="Fetch BR-DWGD zip from Drive if Tmin is not already cached",
    )
    p.add_argument(
        "--nc-dir",
        type=Path,
        default=NC_DIR,
        help="Cache dir for national Xavier NetCDFs / zip",
    )
    p.add_argument(
        "--template",
        type=Path,
        default=TEMPLATE,
        help="Existing Tmax_PR_daily_*.nc to match grid/time/encoding",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=OUT,
        help="Destination Tmin_PR_daily_*.nc",
    )
    p.add_argument(
        "--credentials-dir",
        type=Path,
        default=_REPO / "data",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output if it already exists",
    )
    return p.parse_args()


def _parse_stem(path: Path) -> tuple[str, date, date] | None:
    m = _FILE_STEM_RE.match(path.stem)
    if not m:
        return None
    var = m.group("var")
    lo = datetime_from_yyyymmdd(m.group("start"))
    hi = datetime_from_yyyymmdd(m.group("end"))
    return var, lo, hi


def datetime_from_yyyymmdd(s: str) -> date:
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def find_local_tmin(nc_dir: Path) -> Path | None:
    ranked: list[tuple[int, Path]] = []
    for path in nc_dir.rglob("*"):
        if path.suffix.lower() not in {".nc", ".nc4"}:
            continue
        parsed = _parse_stem(path)
        if parsed is None:
            # Accept names like Tmin_*.nc without dates if var is clear
            if path.name.lower().startswith("tmin"):
                ranked.append((10**9, path))
            continue
        var, lo, hi = parsed
        if var.lower() != "tmin":
            continue
        if lo <= START and hi >= END:
            ranked.append(((hi - lo).days, path))
    if not ranked:
        return None
    ranked.sort(key=lambda x: x[0])
    return ranked[0][1]


def pick_drive_tmin_zip(files: list[dict]) -> dict | None:
    """Prefer pr_Tmax_Tmin zip; else any zip mentioning Tmin."""
    preferred = None
    fallback = None
    for item in files:
        name = str(item.get("name") or "")
        low = name.lower()
        if not low.endswith(".zip"):
            continue
        if "tmin" in low or ("tmax" in low and "pr" in low):
            if low.startswith("pr") and "tmax" in low and "tmin" in low:
                preferred = item
            elif fallback is None:
                fallback = item
    return preferred or fallback


def pick_drive_tmin_nc(files: list[dict]) -> dict | None:
    ranked: list[tuple[int, dict]] = []
    for item in files:
        name = str(item.get("name") or "")
        path = Path(name)
        if path.suffix.lower() not in {".nc", ".nc4"}:
            continue
        parsed = _parse_stem(path)
        if parsed is None:
            continue
        var, lo, hi = parsed
        if var.lower() != "tmin":
            continue
        if lo <= START and hi >= END:
            ranked.append(((hi - lo).days, item))
    if not ranked:
        return None
    ranked.sort(key=lambda x: x[0])
    return ranked[0][1]


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
        pbar = tqdm(total=size, unit="B", unit_scale=True, desc=str(item.get("name"))[:40])
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


def extract_tmin_from_zip(zip_path: Path, dest_dir: Path) -> Path | None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            n
            for n in zf.namelist()
            if Path(n).name.lower().startswith("tmin")
            and Path(n).suffix.lower() in {".nc", ".nc4"}
        ]
        if not members:
            print(f"  zip has no Tmin_*.nc; sample: {zf.namelist()[:12]}")
            return None
        chosen: list[str] = []
        for name in members:
            parsed = _parse_stem(Path(name))
            if parsed is None:
                chosen.append(name)
                continue
            var, lo, hi = parsed
            if var.lower() == "tmin" and lo <= START and hi >= END:
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


def ensure_tmin_nc(nc_dir: Path, credentials_dir: Path, download: bool) -> Path:
    local = find_local_tmin(nc_dir)
    if local is not None:
        print(f"National Tmin already on disk: {local}")
        return local
    if not download:
        raise SystemExit(
            f"No Tmin NetCDF under {nc_dir}. Re-run with --download "
            "(BR-DWGD Drive folder with pr_Tmax_Tmin_NetCDF_Files.zip)."
        )

    service = build_drive_service(credentials_dir)
    print(f"Listing public BR-DWGD Drive folder {BR_DWGD_DRIVE_FOLDER_ID}...")
    files = list_drive_tree(service, BR_DWGD_DRIVE_FOLDER_ID)
    for item in files:
        print(f"  {item.get('name')} ({item.get('size', '?')} bytes)")

    nc_item = pick_drive_tmin_nc(files)
    if nc_item is not None:
        dest = nc_dir / Path(str(nc_item["name"])).name
        if not dest.is_file():
            download_file(service, nc_item, dest)
        return dest

    zip_item = pick_drive_tmin_zip(files)
    if zip_item is None:
        raise SystemExit(
            "No Tmin NetCDF / pr_Tmax_Tmin zip in Drive folder "
            f"{BR_DWGD_DRIVE_FOLDER_ID}"
        )
    zip_dest = nc_dir / Path(str(zip_item["name"])).name
    if not zip_dest.is_file():
        download_file(service, zip_item, zip_dest)
    extracted = extract_tmin_from_zip(zip_dest, nc_dir)
    if extracted is None:
        raise SystemExit(f"Downloaded {zip_dest.name} but found no Tmin_*.nc inside")
    return extracted


def resolve_tmin_var(ds: xr.Dataset) -> str:
    for name in ("Tmin", "tmin", "TMIN"):
        if name in ds.data_vars or name in ds.variables:
            return name
    raise KeyError(f"No Tmin variable in dataset; have {list(ds.data_vars)}")


def align_to_template(tmin_path: Path, template_path: Path) -> xr.DataArray:
    with xr.open_dataset(os.fspath(template_path), decode_times=True) as tmpl:
        t_ref = tmpl["Tmax"]
        x_ref = np.asarray(t_ref["x"].values)
        y_ref = np.asarray(t_ref["y"].values)
        time_ref = t_ref["time"]

    with xr.open_dataset(os.fspath(tmin_path), decode_times=True) as ds:
        var = resolve_tmin_var(ds)
        da = ds[var]
        # Normalize coord names
        rename = {}
        for src, dst in (("longitude", "x"), ("lon", "x"), ("latitude", "y"), ("lat", "y")):
            if src in da.coords or src in da.dims:
                rename[src] = dst
        if rename:
            da = da.rename(rename)
        if "time" not in da.dims:
            raise ValueError(f"Tmin dims {da.dims}: expected time")
        da = da.transpose("time", "y", "x")
        # Select time window then interpolate onto template grid if needed
        da = da.sel(time=slice(np.datetime64(START), np.datetime64(END)))
        da = da.load()

    # Match template time exactly when possible
    try:
        da = da.sel(time=time_ref.values)
    except Exception:
        da = da.interp(time=time_ref)

    # Match spatial grid (nearest — Xavier grids should already coincide)
    try:
        da = da.sel(x=x_ref, y=y_ref, method="nearest")
    except Exception:
        da = da.interp(x=x_ref, y=y_ref)

    da = da.assign_coords(x=("x", x_ref), y=("y", y_ref), time=time_ref)
    da = da.astype(np.float32)
    da.name = "Tmin"
    return da


def save_like_tmax(da: xr.DataArray, dest: Path, template_path: Path) -> None:
    with xr.open_dataset(os.fspath(template_path)) as tmpl:
        tmax = tmpl["Tmax"]
        scale = float(tmax.encoding.get("scale_factor", 0.001))
        offset = float(tmax.encoding.get("add_offset", 0.0))
        fill = np.int16(tmax.encoding.get("_FillValue", -32768))
        chunks = tmax.encoding.get("chunksizes")
        spatial = tmpl["spatial_ref"] if "spatial_ref" in tmpl else None
        spatial_attrs = dict(tmpl["spatial_ref"].attrs) if spatial is not None else {}

    ds = da.to_dataset(name="Tmin")
    ds["Tmin"].attrs.update(
        units="Celcius degrees",
        standard_name="Tmin",
        valid_min=np.int16(-32767),
        valid_max=np.int16(32767),
        FillValue=np.int64(-32768),
        grid_mapping="spatial_ref",
    )
    if spatial is not None:
        ds["spatial_ref"] = spatial
        ds["spatial_ref"].attrs.update(spatial_attrs)

    enc: dict = {
        "Tmin": {
            "dtype": "int16",
            "scale_factor": scale,
            "add_offset": offset,
            "_FillValue": fill,
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
        }
    }
    if chunks is not None:
        enc["Tmin"]["chunksizes"] = chunks

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    ds.to_netcdf(tmp, encoding=enc)
    tmp.replace(dest)
    print(f"Wrote {dest} shape={tuple(da.shape)}")


def main() -> None:
    args = parse_args()
    template = args.template.expanduser().resolve()
    output = args.output.expanduser().resolve()
    nc_dir = args.nc_dir.expanduser().resolve()
    if not template.is_file():
        raise SystemExit(f"Template missing: {template}")
    if output.is_file() and not args.force:
        print(f"Already exists: {output} (pass --force to overwrite)")
        return

    tmin_src = ensure_tmin_nc(
        nc_dir,
        args.credentials_dir.expanduser().resolve(),
        download=bool(args.download),
    )
    print(f"Aligning {tmin_src.name} → template {template.name}")
    da = align_to_template(tmin_src, template)
    save_like_tmax(da, output, template)


if __name__ == "__main__":
    main()
