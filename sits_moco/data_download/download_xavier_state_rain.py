#!/usr/bin/env python3
"""
Write daily Xavier rain NetCDFs in the same layout as files/climate/.

Existing Paraná file:
  files/climate/pr_PR_daily_2020_2024.nc
  dims (time, y, x), packed int16 pr, SIRGAS 2000 spatial_ref,
  calendar years 2020-01-01 … 2024-12-31.

This script does the same for other UFs / the missing 2019–2020 years:
  files/climate/pr_{UF}_daily_{Y1}_{Y2}.nc

It does not compute cum-pr or municipal means (that is the next step, same
as the original pixel cubes).

Example (new 2020 zonal-mean states, calendar 2019–2020):
  python data_download/download_xavier_state_rain.py --download

  python data_download/download_xavier_state_rain.py --states mt,go,to --download
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import zipfile
from datetime import date, datetime
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

from download_pam_soy_sidra import UF_BY_CODE, parse_state_codes
from xavier_rain_for_daily_npy import parse_year_range_bounds

BR_DWGD_DRIVE_FOLDER_ID = "11-qnvwojirAtaQxSE03N0_SUrbcsz44N"
VAR = "pr"
_VAR_ALIASES = ("pr", "PR", "precip")
_FILE_STEM_RE = re.compile(
    r"^(?P<var>pr|ETo|Rs|Tmax|Tmin|RH|u2)_(?P<start>\d{8})_(?P<end>\d{8})",
    re.IGNORECASE,
)
_CLIMATE_STEM_RE = re.compile(
    r"^(pr|ETo|Rs|Tmax|Tmin|RH|u2)_[A-Za-z]{2}_daily_\d{4}_\d{4}$",
    re.IGNORECASE,
)
FOLDER_MIME = "application/vnd.google-apps.folder"
BBOX_PAD_DEG = 0.2
UF_ABBR = {v: k for k, v in UF_BY_CODE.items()}
TEMPLATE_PR = Path("files/climate/pr_PR_daily_2020_2024.nc")
# Same packing as files/climate/pr_PR_daily_2020_2024.nc
PR_SCALE = 0.006866664632099367
PR_OFFSET = 224.99999999999997
PR_FILL = np.int16(-32768)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--keep-list",
        type=Path,
        default=Path("files/zonal_mean_keep_0.8_1.2.csv"),
        help="Infer UFs from municipality_code (harvest year = year-range end)",
    )
    p.add_argument(
        "--states",
        type=str,
        default=None,
        help="Override UFs, e.g. mt,go,to. Default: keep-list states missing rain.",
    )
    p.add_argument(
        "--npy-root",
        type=Path,
        default=Path("~/sits_moco_data/npy_muni_mean"),
    )
    p.add_argument(
        "--all-keep",
        action="store_true",
        help="Include UFs even if every muni already has Xavier rain channels",
    )
    p.add_argument(
        "--shapefile-dir",
        type=Path,
        default=Path("files/shapefiles"),
        help="Municipal shapefiles; used for state bbox if geobr is unavailable",
    )
    p.add_argument(
        "--nc-dir",
        type=Path,
        default=Path("files/xavier_nc"),
        help="Cache for the national BR-DWGD pr download",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("files/climate"),
        help="Same folder as pr_PR_daily_2020_2024.nc",
    )
    p.add_argument(
        "--year-range",
        type=str,
        default="2019-2020",
        metavar="YYYY-YYYY",
        help="Calendar years in the filename (default 2019-2020 → 2019-01-01…2020-12-31)",
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="Fetch the covering national pr NetCDF from the public Drive folder",
    )
    p.add_argument(
        "--credentials-dir",
        type=Path,
        default=Path("data"),
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not overwrite an existing files/climate/pr_{UF}_daily_*.nc",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
    )
    return p.parse_args()


def code7(value: object) -> str:
    return f"{int(float(value)):07d}"


def uf_from_muni(code: str) -> int:
    return int(code[:2])


def npy_path(root: Path, year_range: str, code: str) -> Path | None:
    nested = root / year_range / code / f"{code}.npy"
    if nested.is_file():
        return nested
    flat = root / year_range / f"{code}.npy"
    return flat if flat.is_file() else None


def npy_channels(path: Path) -> int | None:
    try:
        arr = np.load(path, mmap_mode="r")
    except OSError:
        return None
    if arr.ndim < 3:
        return None
    return int(arr.shape[-1])


def keep_codes_for_harvest(path: Path, harvest_year: int) -> list[str]:
    codes: list[str] = []
    seen: set[str] = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            year_raw = row.get("year") or row.get("season_year")
            if year_raw is None or str(year_raw).strip() == "":
                continue
            if int(float(year_raw)) != harvest_year:
                continue
            try:
                code = code7(row["municipality_code"])
            except (KeyError, TypeError, ValueError):
                continue
            if code not in seen:
                seen.add(code)
                codes.append(code)
    return codes


def codes_missing_rain(
    codes: list[str], npy_root: Path, year_range: str
) -> list[str]:
    kept: list[str] = []
    skipped = 0
    for code in codes:
        path = npy_path(npy_root, year_range, code)
        if path is None:
            kept.append(code)
            continue
        n_ch = npy_channels(path)
        if n_ch is None or n_ch < 13:
            kept.append(code)
        else:
            skipped += 1
    print(
        f"Need rain extras: {len(kept)} munis "
        f"({skipped} already have ≥13 channels)"
    )
    return kept


def infer_state_codes(muni_codes: list[str]) -> list[int]:
    ufs: list[int] = []
    seen: set[int] = set()
    for code in muni_codes:
        uf = uf_from_muni(code)
        if uf not in UF_BY_CODE:
            continue
        if uf not in seen:
            seen.add(uf)
            ufs.append(uf)
    return sorted(ufs)


def ymd_to_date(text: str) -> date:
    return datetime.strptime(text, "%Y%m%d").date()


def parse_nc_filename(path: Path) -> tuple[str, date, date] | None:
    m = _FILE_STEM_RE.match(path.stem)
    if not m:
        return None
    var = "pr" if m.group("var").lower() == "pr" else m.group("var")
    return var, ymd_to_date(m.group("start")), ymd_to_date(m.group("end"))


def is_climate_layout(path: Path) -> bool:
    return bool(_CLIMATE_STEM_RE.match(path.stem))


def find_national_pr(nc_dir: Path, start: date, end: date) -> Path | None:
    aliases = {a.lower() for a in _VAR_ALIASES}
    candidates: list[tuple[int, Path]] = []
    for path in nc_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".nc", ".nc4"}:
            continue
        if is_climate_layout(path):
            continue
        parsed = parse_nc_filename(path)
        if parsed is not None:
            name, lo, hi = parsed
            if name != VAR:
                continue
            if lo <= start and hi >= end:
                candidates.append(((hi - lo).days, path))
            continue
        if path.stem.lower() in aliases:
            candidates.append((10**9, path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1].name))
    return candidates[0][1]


def list_drive_tree(service, folder_id: str) -> list[dict]:
    out: list[dict] = []
    page_token = None
    while True:
        resp = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                pageSize=100,
                fields="nextPageToken, files(id, name, mimeType, size)",
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        for item in resp.get("files", []):
            if item.get("mimeType") == FOLDER_MIME:
                out.extend(list_drive_tree(service, item["id"]))
            else:
                out.append(item)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return out


def pick_drive_pr(files: list[dict], start: date, end: date) -> dict | None:
    """Prefer a dated pr_*.nc; else the BR-DWGD zip that contains pr (Tmax/Tmin)."""
    ranked: list[tuple[int, dict]] = []
    zip_fallback: dict | None = None
    for item in files:
        name = str(item.get("name") or "")
        path = Path(name)
        suffix = path.suffix.lower()
        parsed = parse_nc_filename(path)
        if parsed is not None and suffix in {".nc", ".nc4", ".zip"}:
            var, lo, hi = parsed
            if var == VAR and lo <= start and hi >= end:
                ranked.append(((hi - lo).days, item))
            continue
        # Official folder ships one zip: pr_Tmax_Tmin_NetCDF_Files.zip
        stem = path.stem.lower()
        if suffix == ".zip" and stem.startswith("pr") and "control" not in stem:
            zip_fallback = item
    if ranked:
        ranked.sort(key=lambda x: (x[0], x[1].get("name") or ""))
        return ranked[0][1]
    return zip_fallback


def extract_pr_from_zip(zip_path: Path, dest_dir: Path, start: date, end: date) -> Path | None:
    """Extract only pr_*.nc members that cover [start, end]."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if Path(n).name.lower().startswith("pr")]
        members = [
            n
            for n in members
            if Path(n).suffix.lower() in {".nc", ".nc4"}
        ]
        if not members:
            print(f"  zip has no pr_*.nc; members like {zf.namelist()[:8]}")
            return None
        chosen: list[str] = []
        for name in members:
            parsed = parse_nc_filename(Path(name))
            if parsed is None:
                chosen.append(name)
                continue
            var, lo, hi = parsed
            if var == VAR and lo <= start and hi >= end:
                chosen.append(name)
        if not chosen:
            chosen = members
        # Prefer the shortest date span that still covers the window.
        def span_key(name: str) -> int:
            parsed = parse_nc_filename(Path(name))
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


def download_national_pr(
    nc_dir: Path, start: date, end: date, credentials_dir: Path
) -> Path:
    from drive_api import build_drive_service
    from googleapiclient.http import MediaIoBaseDownload

    existing = find_national_pr(nc_dir, start, end)
    if existing is not None:
        print(f"National pr already on disk: {existing}")
        return existing

    service = build_drive_service(credentials_dir)
    print(f"Listing public BR-DWGD Drive folder {BR_DWGD_DRIVE_FOLDER_ID}...")
    files = list_drive_tree(service, BR_DWGD_DRIVE_FOLDER_ID)
    print(f"  {len(files)} files in folder tree:")
    for item in files:
        print(f"    {item.get('name')} ({item.get('size', '?')} bytes)")
    item = pick_drive_pr(files, start, end)
    if item is None:
        raise SystemExit(
            f"No Drive pr archive covering {start}–{end}. "
            "Expected pr_Tmax_Tmin_NetCDF_Files.zip in "
            "https://drive.google.com/drive/folders/11-qnvwojirAtaQxSE03N0_SUrbcsz44N"
        )
    nc_dir.mkdir(parents=True, exist_ok=True)
    name = item["name"]
    dest = nc_dir / Path(name).name
    print(f"Downloading {name} ({item.get('size', '?')} bytes)")
    request = service.files().get_media(fileId=item["id"])
    tmp = dest.with_suffix(dest.suffix + ".part")
    size = int(item["size"]) if str(item.get("size") or "").isdigit() else None
    with open(tmp, "wb") as f:
        downloader = MediaIoBaseDownload(f, request, chunksize=16 * 1024 * 1024)
        done = False
        pbar = tqdm(total=size, unit="B", unit_scale=True, desc=name[:40])
        progress = 0
        while not done:
            status, done = downloader.next_chunk()
            if status is not None:
                nxt = int(status.resumable_progress)
                pbar.update(max(0, nxt - progress))
                progress = nxt
        pbar.close()
    tmp.replace(dest)
    if dest.suffix.lower() == ".zip":
        extracted = extract_pr_from_zip(dest, nc_dir, start, end)
        if extracted is None:
            raise SystemExit(f"Downloaded {name} but it has no pr_*.nc inside")
        return extracted
    found = find_national_pr(nc_dir, start, end)
    if found is None:
        raise SystemExit(f"Downloaded {name} but could not find a pr NetCDF in {nc_dir}")
    return found


def _coord(ds: xr.Dataset, names: tuple[str, ...]) -> xr.DataArray:
    for name in names:
        if name in ds.coords or name in ds.variables:
            return ds[name]
    raise KeyError(f"Dataset missing coords {names}; have {list(ds.coords)}")


def load_pr_window(nc_path: Path, start: date, end: date) -> xr.DataArray:
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    with xr.open_dataset(os.fspath(nc_path), decode_times=True, engine="netcdf4") as ds:
        var_name = next(
            (n for n in _VAR_ALIASES if n in ds.data_vars or n in ds.variables),
            None,
        )
        if var_name is None:
            raise KeyError(f"{nc_path} has no pr; data_vars={list(ds.data_vars)}")
        da = ds[var_name]
        x = _coord(ds, ("x", "longitude", "lon"))
        y = _coord(ds, ("y", "latitude", "lat"))
        da = da.rename({x.name: "x", y.name: "y"})
        if "time" not in da.dims:
            raise ValueError(f"pr dims {da.dims}: expected time")
        da = da.transpose("time", "y", "x")
        sliced = da.sel(time=slice(np.datetime64(start), np.datetime64(end)))
        loaded = sliced.load()
    if loaded.sizes.get("time", 0) == 0:
        raise SystemExit(f"{nc_path}: no timesteps in {start}–{end}")
    return loaded.astype(np.float32)


def clip_bbox(
    da: xr.DataArray, lon_min: float, lat_min: float, lon_max: float, lat_max: float
) -> xr.DataArray:
    x = da["x"]
    y = da["y"]
    x_slice = (
        slice(lon_min, lon_max) if float(x[0]) < float(x[-1]) else slice(lon_max, lon_min)
    )
    y_slice = (
        slice(lat_min, lat_max) if float(y[0]) < float(y[-1]) else slice(lat_max, lat_min)
    )
    return da.sel(x=x_slice, y=y_slice)


def state_bboxes(
    uf_codes: list[int], shapefile_dir: Path
) -> dict[int, tuple[float, float, float, float]]:
    """Return {uf: (lon_min, lat_min, lon_max, lat_max)} padded in degrees."""
    out: dict[int, tuple[float, float, float, float]] = {}
    try:
        from geobr import read_state

        gdf = read_state(code_state="all", year=2020, simplified=True)
        col = "code_state" if "code_state" in gdf.columns else "code_muni"
        for uf in uf_codes:
            sub = gdf[gdf[col].astype(int) == uf]
            if sub.empty:
                continue
            minx, miny, maxx, maxy = sub.total_bounds
            out[uf] = (
                float(minx) - BBOX_PAD_DEG,
                float(miny) - BBOX_PAD_DEG,
                float(maxx) + BBOX_PAD_DEG,
                float(maxy) + BBOX_PAD_DEG,
            )
        missing = [u for u in uf_codes if u not in out]
        if not missing:
            return out
        print(f"geobr missed UFs {missing}; falling back to municipal shapefiles")
    except Exception as exc:
        print(f"geobr state bounds unavailable ({exc}); using municipal shapefiles")

    import geopandas as gpd

    shapefile_dir = shapefile_dir.expanduser().resolve()
    for uf in uf_codes:
        if uf in out:
            continue
        prefix = f"{uf:02d}"
        xs: list[float] = []
        ys: list[float] = []
        for sub in sorted(shapefile_dir.glob(f"{prefix}*")):
            shps = list(sub.glob("*.shp")) if sub.is_dir() else []
            if not shps:
                continue
            gdf = gpd.read_file(shps[0])
            if gdf.empty:
                continue
            if gdf.crs is None or gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            minx, miny, maxx, maxy = gdf.total_bounds
            xs.extend([minx, maxx])
            ys.extend([miny, maxy])
        if not xs:
            raise SystemExit(
                f"No bbox for {UF_BY_CODE.get(uf, uf)} ({uf}): "
                f"install geobr or add shapefiles under {shapefile_dir}/{prefix}*"
            )
        out[uf] = (
            min(xs) - BBOX_PAD_DEG,
            min(ys) - BBOX_PAD_DEG,
            max(xs) + BBOX_PAD_DEG,
            max(ys) + BBOX_PAD_DEG,
        )
    return out


def climate_out_path(output_dir: Path, abbr: str, y1: int, y2: int) -> Path:
    return output_dir / f"pr_{abbr}_daily_{y1}_{y2}.nc"


def save_like_climate_folder(da: xr.DataArray, dest: Path, template: Path) -> None:
    """Write packed pr (time, y, x) like pr_PR_daily_2020_2024.nc."""
    y = da["y"]
    if float(y[0]) > float(y[-1]):
        da = da.isel(y=slice(None, None, -1))
    ds = da.to_dataset(name="pr")
    ds["pr"].attrs.update(
        units="mm",
        standard_name="pr",
        valid_min=np.int16(-32767),
        valid_max=np.int16(32767),
        FillValue=np.int64(-32768),
        grid_mapping="spatial_ref",
    )
    if template.is_file():
        with xr.open_dataset(os.fspath(template)) as tmpl:
            if "spatial_ref" in tmpl:
                ds["spatial_ref"] = tmpl["spatial_ref"]
                ds["spatial_ref"].attrs.update(dict(tmpl["spatial_ref"].attrs))
    encoding = {
        "pr": {
            "dtype": "int16",
            "scale_factor": PR_SCALE,
            "add_offset": PR_OFFSET,
            "_FillValue": PR_FILL,
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
        }
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(dest, encoding=encoding)


def main() -> None:
    args = parse_args()
    y1, y2 = parse_year_range_bounds(args.year_range.strip())
    start, end = date(y1, 1, 1), date(y2, 12, 31)
    harvest = y2
    year_range = f"{y1}-{y2}"
    nc_dir = args.nc_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    npy_root = args.npy_root.expanduser().resolve()
    template = TEMPLATE_PR
    if not template.is_file():
        template = output_dir / "pr_PR_daily_2020_2024.nc"

    if args.states:
        uf_codes = parse_state_codes(args.states)
    else:
        keep_path = args.keep_list.expanduser().resolve()
        if not keep_path.is_file():
            raise SystemExit(f"Keep list not found: {keep_path}")
        codes = keep_codes_for_harvest(keep_path, harvest)
        print(f"{keep_path.name}: {len(codes)} municipalities in harvest {harvest}")
        if not args.all_keep:
            codes = codes_missing_rain(codes, npy_root, year_range)
        uf_codes = infer_state_codes(codes)

    if not uf_codes:
        raise SystemExit("No states to download")
    labels = ", ".join(f"{UF_BY_CODE[u]}({u})" for u in uf_codes)
    print(f"States: {labels}")
    print(
        f"Emulating {template.name if template.is_file() else 'pr_UF_daily_Y1_Y2.nc'}: "
        f"{start}–{end} → {output_dir}/pr_{{UF}}_daily_{y1}_{y2}.nc"
    )

    bboxes = state_bboxes(uf_codes, args.shapefile_dir)
    if args.dry_run:
        for uf in uf_codes:
            abbr = UF_BY_CODE[uf]
            dest = climate_out_path(output_dir, abbr, y1, y2)
            bbox = bboxes[uf]
            print(
                f"  {abbr}: bbox lon[{bbox[0]:.2f},{bbox[2]:.2f}] "
                f"lat[{bbox[1]:.2f},{bbox[3]:.2f}] → {dest.name}"
            )
        return

    if args.download:
        nc_path = download_national_pr(
            nc_dir, start, end, args.credentials_dir.expanduser().resolve()
        )
    else:
        nc_path = find_national_pr(nc_dir, start, end)
        if nc_path is None:
            raise SystemExit(
                f"No national pr NetCDF covering {start}–{end} in {nc_dir}. "
                "Re-run with --download."
            )
        print(f"National pr: {nc_path}")

    print(f"Loading pr {start}–{end}...")
    da = load_pr_window(nc_path, start, end)
    print(f"  national window shape {tuple(da.shape)} {dict(da.sizes)}")

    n_ok = 0
    for uf in uf_codes:
        abbr = UF_BY_CODE[uf]
        dest = climate_out_path(output_dir, abbr, y1, y2)
        if args.skip_existing and dest.is_file():
            print(f"  {abbr}: skip existing {dest.name}")
            n_ok += 1
            continue
        lon_min, lat_min, lon_max, lat_max = bboxes[uf]
        clipped = clip_bbox(da, lon_min, lat_min, lon_max, lat_max)
        if clipped.sizes.get("x", 0) == 0 or clipped.sizes.get("y", 0) == 0:
            raise SystemExit(f"{abbr}: empty clip for bbox {bboxes[uf]}")
        save_like_climate_folder(clipped, dest, template)
        print(
            f"  {abbr}: wrote {dest.name} "
            f"{tuple(clipped.shape)} y×x="
            f"{clipped.sizes['y']}×{clipped.sizes['x']}"
        )
        n_ok += 1

    print(f"Done: {n_ok}/{len(uf_codes)} files in {output_dir}")


if __name__ == "__main__":
    main()
