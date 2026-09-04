#!/usr/bin/env python3
"""
Download MapBiomas-soy zonal MEAN Sentinel-2 time series per municipality from GEE.

For each season:
  1. MapBiomas soybean inventory (class 39, harvest year)
  2. Chunk municipalities by UF and run one S2 reduceRegions export per chunk
     (one daily mosaic, many polygons — far cheaper than one GEE task per muni)
  3. Split the Drive table into one CSV per municipality (same schema as before)
     (GEE COMPLETED can precede Drive listing; wait --drive-wait before giving up,
     and do not start a new export just because the CSV is late)
  4. Quality columns: valid_pixel_count, clear_fraction, mean_cloud_score_soy
  5. Sidecar area file: mapbiomas_soy_area_ha vs IBGE PAM (filter after download)

Usage:
  python data_download/download_soy_gee_zonal_mean.py \\
      --shapefile-dir files/shapefiles --season-year 2024 \\
      --gee-project crop-yield-pred-506711 --gee-account heigthtrielli@gmail.com

  # Rio Grande do Sul only, skip municipalities already on disk:
  python data_download/download_soy_gee_zonal_mean.py \\
      --shapefile-dir files/shapefiles --season-years 2020-2024 \\
      --pam-csv files/pam_soy_br_2020_2024.csv --states rs \\
      --skip-existing --workers 10 --export-chunk-size 40

  # Soy presence only (writes inventory CSVs, no S2 export):
  python data_download/download_soy_gee_zonal_mean.py \\
      --shapefile-dir files/shapefiles --season-years 2020-2024 --inventory-only

  # Pull chunk CSVs already on Drive (no new GEE exports); skip munis on disk:
  python data_download/download_soy_gee_zonal_mean.py \\
      --shapefile-dir files/shapefiles --ingest-drive --skip-existing

  # Same, from a local folder of Drive downloads:
  python data_download/download_soy_gee_zonal_mean.py \\
      --shapefile-dir files/shapefiles --ingest-dir ~/Downloads/GEE_Soy_Zonal_Export --skip-existing

Output layout:
  {output_dir}/{YYYY-YYYY}/{municipality_code}/
    {code}_zonal.csv          daily band means + quality
    {code}_area.csv             mapbiomas soy area (one row)
    mapbiomas_soy_inventory.csv all munis × this season (soy pixels / ha)
  {output_dir}/mapbiomas_no_soy.csv
    (season_year, municipality_code) with zero MapBiomas soy — skipped for S2
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import ee
import geopandas as gpd

from drive_api import (
    build_drive_service,
    delete_file_from_drive,
    download_file_from_drive,
    get_folder_id_by_name,
    list_files_in_folder,
)
from gee_export import DEFAULT_GEE_ACCOUNT, DEFAULT_GEE_PROJECT, initialize_gee
from gee_zonal import (
    chunk_zonal_feature_collection,
    mapbiomas_soy_areas_for_collection,
    start_table_export_task,
    wait_for_gee_task,
)

ZONAL_SELECTORS = [
    "B11",
    "B12",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "clear_fraction",
    "date",
    "doy",
    "mean_cloud_score_soy",
    "municipality_code",
    "season_year",
    "soy_pixel_count",
    "valid_pixel_count",
]
ZONAL_CSV_FIELDS = ["system:index", *ZONAL_SELECTORS, ".geo"]
EMPTY_GEO = '{"type":"MultiPoint","coordinates":[]}'

from download_pam_soy_sidra import (
    UF_BY_CODE,
    default_output_path,
    fetch_states,
    parse_state_codes,
)

DEFAULT_CREDENTIALS_DIR = Path("data")
DEFAULT_DRIVE_WAIT_SECONDS = 900
_LOG_LOCK = threading.Lock()
_THREAD_LOCAL = threading.local()


class DriveLagError(RuntimeError):
    """GEE export finished but the CSV is not listed on Drive yet (or still missing)."""


INVENTORY_FIELDS = [
    "municipality_code",
    "season_year",
    "year_range",
    "mapbiomas_soy_pixel_count",
    "mapbiomas_soy_area_ha",
]


def log(msg: str) -> None:
    with _LOG_LOCK:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    shape_group = p.add_mutually_exclusive_group(required=False)
    shape_group.add_argument(
        "--shapefile",
        type=Path,
        action="append",
        dest="shapefiles",
        help="Municipality .shp (repeatable)",
    )
    shape_group.add_argument(
        "--shapefile-dir",
        type=Path,
        dest="shapefile_dir",
        help="Directory of municipality subfolders each containing a .shp",
    )
    p.add_argument(
        "--season-year",
        type=int,
        default=None,
        metavar="X",
        help="Single harvest season X: Oct (X-1)–Mar (X)",
    )
    p.add_argument(
        "--season-years",
        type=str,
        default=None,
        metavar="SPEC",
        help="Multiple seasons: '2020-2024' or '2020,2021,2022' (loops all munis each year)",
    )
    p.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Custom start YYYY-MM-DD (with --end-date; single range, not multi-year)",
    )
    p.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Custom end YYYY-MM-DD",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("files/zonal_mean"),
        help="Base output directory (default: files/zonal_mean)",
    )
    p.add_argument(
        "--drive-folder",
        type=str,
        default="GEE_Soy_Zonal_Export",
        help="Temporary Drive folder for CSV exports",
    )
    p.add_argument(
        "--gee-project",
        type=str,
        default=DEFAULT_GEE_PROJECT,
        help=f"GEE Cloud project ID (default: {DEFAULT_GEE_PROJECT})",
    )
    p.add_argument(
        "--gee-account",
        type=str,
        default=DEFAULT_GEE_ACCOUNT,
        help=f"Google account for Earth Engine auth (default: {DEFAULT_GEE_ACCOUNT})",
    )
    p.add_argument(
        "--scale",
        type=int,
        default=30,
        help="reduceRegion scale in meters (default: 30)",
    )
    p.add_argument(
        "--cloud-pct",
        type=float,
        default=20.0,
        help="Max scene CLOUDY_PIXEL_PERCENTAGE metadata filter (default: 20)",
    )
    p.add_argument(
        "--cloud-score-threshold",
        type=float,
        default=0.60,
        help="Cloud Score+ clear pixel threshold (default: 0.60)",
    )
    p.add_argument(
        "--max-download-workers",
        type=int,
        default=None,
        help="Deprecated alias for --workers",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Parallel chunk export+download jobs (default: 4). "
        "Each job is many municipalities; 2–4 is usually enough",
    )
    p.add_argument(
        "--soy-chunk-size",
        type=int,
        default=200,
        help="Municipalities per MapBiomas reduceRegions getInfo (default: 200)",
    )
    p.add_argument(
        "--export-chunk-size",
        type=int,
        default=40,
        help="Municipalities per S2 reduceRegions export, grouped by UF (default: 40)",
    )
    p.add_argument(
        "--inventory-only",
        action="store_true",
        help="Build soy inventories and stop (no Sentinel-2 export)",
    )
    p.add_argument(
        "--rebuild-inventory",
        action="store_true",
        help="Ignore cached mapbiomas_soy_inventory.csv and re-query GEE",
    )
    p.add_argument(
        "--pam-csv",
        type=Path,
        default=None,
        help="IBGE PAM soy CSV; municipalities with no planted area that "
        "harvest year are skipped (SIDRA is fetched if --download-pam)",
    )
    p.add_argument(
        "--download-pam",
        action="store_true",
        help="Download PAM from SIDRA into --pam-csv (UFs inferred from shapefiles)",
    )
    p.add_argument(
        "--pam-states",
        type=str,
        default=None,
        help="UFs for --download-pam (default: infer from shapefile codes, or all)",
    )
    p.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between GEE task polls",
    )
    p.add_argument(
        "--drive-wait",
        type=int,
        default=DEFAULT_DRIVE_WAIT_SECONDS,
        metavar="SEC",
        help=(
            "After GEE COMPLETED, keep polling Drive for the CSV before giving up "
            f"(default: {DEFAULT_DRIVE_WAIT_SECONDS}s). Does not start a new export."
        ),
    )
    p.add_argument(
        "--credentials-dir",
        type=Path,
        default=DEFAULT_CREDENTIALS_DIR,
        help="Drive OAuth credentials directory",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip municipality if {code}_zonal.csv already exists",
    )
    p.add_argument(
        "--ingest-drive",
        action="store_true",
        help="Download chunk CSVs already in --drive-folder and split them; no GEE export",
    )
    p.add_argument(
        "--ingest-dir",
        type=Path,
        default=None,
        help="Local folder of zonal_*.csv chunk tables to split (no Drive, no GEE)",
    )
    p.add_argument(
        "--delete-drive-after",
        action="store_true",
        help="With --ingest-drive, delete each Drive CSV after it is split",
    )
    p.add_argument(
        "--states",
        type=str,
        default=None,
        help="Restrict shapefiles to these UFs (e.g. rs, pr,sc, 43). "
        "Default: all municipalities under --shapefile-dir",
    )
    args = p.parse_args()
    args.ingest_only = bool(args.ingest_drive) or args.ingest_dir is not None
    if args.ingest_drive and args.ingest_dir is not None:
        p.error("Use only one of --ingest-drive or --ingest-dir")

    if args.shapefile_dir is not None:
        d = Path(args.shapefile_dir).resolve()
        if not d.is_dir():
            p.error(f"--shapefile-dir is not a directory: {d}")
        args.shapefiles = []
        for subdir in sorted(d.iterdir()):
            if subdir.is_dir():
                shps = list(subdir.glob("*.shp"))
                if shps:
                    args.shapefiles.append(shps[0])
        if not args.shapefiles and not args.ingest_only:
            p.error(f"No .shp files found under {d}")
    else:
        args.shapefiles = [Path(p).resolve() for p in (args.shapefiles or [])]

    if not args.shapefiles and not args.ingest_only:
        p.error("Set --shapefile or --shapefile-dir")

    if args.states:
        try:
            state_codes = parse_state_codes(args.states)
        except ValueError as e:
            p.error(str(e))
        kept = filter_shapefiles_by_states(args.shapefiles, state_codes)
        labels = ",".join(UF_BY_CODE[c] for c in state_codes)
        if not kept and not args.ingest_only:
            p.error(
                f"--states {args.states} matched 0 shapefiles "
                f"(looked for IBGE prefixes {state_codes} among "
                f"{len(args.shapefiles)} files)"
            )
        args.shapefiles = kept
        args.states_codes = state_codes
        args.states_label = labels
    else:
        args.states_codes = None
        args.states_label = None

    if args.start_date or args.end_date:
        if not (args.start_date and args.end_date):
            p.error("Both --start-date and --end-date are required together")
        if args.season_years:
            p.error("Do not combine --season-years with --start-date/--end-date")
        args.season_years_list = []
    else:
        args.season_years_list = parse_season_years_spec(
            args.season_years, args.season_year
        )
        if not args.season_years_list and not args.ingest_only:
            p.error("Set --season-year, --season-years, or --start-date/--end-date")

    if args.workers is None:
        args.workers = args.max_download_workers if args.max_download_workers else 4
    args.workers = max(1, int(args.workers))
    args.soy_chunk_size = max(20, int(args.soy_chunk_size))
    args.export_chunk_size = max(5, int(args.export_chunk_size))

    return args


def parse_season_years_spec(
    seasons_spec: str | None,
    single: int | None,
) -> list[int]:
    """Parse '2020-2024' / '2020,2021' / single --season-year into sorted unique years."""
    years: list[int] = []
    if seasons_spec:
        for part in seasons_spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                lo, hi = int(a), int(b)
                if hi < lo:
                    lo, hi = hi, lo
                years.extend(range(lo, hi + 1))
            else:
                years.append(int(part))
    if single is not None:
        years.append(int(single))
    # unique preserve order
    out: list[int] = []
    seen: set[int] = set()
    for y in years:
        if y not in seen:
            seen.add(y)
            out.append(y)
    return out


def get_season_dates(season_year: int) -> list[str]:
    start = datetime(season_year - 1, 10, 1)
    end = datetime(season_year, 3, 31)
    out = []
    d = start
    while d <= end:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def get_date_list_for_season(season_year: int) -> tuple[list[str], int, int]:
    dates = get_season_dates(season_year)
    years = [int(d[:4]) for d in dates]
    return dates, min(years), max(years)


def get_date_list_from_args(args: argparse.Namespace) -> tuple[list[str], int, int]:
    if args.start_date and args.end_date:
        start = datetime.strptime(args.start_date, "%Y-%m-%d")
        end = datetime.strptime(args.end_date, "%Y-%m-%d")
        out = []
        d = start
        while d <= end:
            out.append(d.strftime("%Y-%m-%d"))
            d += timedelta(days=1)
        years = [int(x[:4]) for x in out]
        return out, min(years), max(years)
    # single-season path (multi-year handled in main)
    return get_date_list_for_season(args.season_year)


def write_area_sidecar(
    out_dir: Path,
    municipality_code: str,
    season_year: int,
    year_start: int,
    year_end: int,
    area_info: dict[str, float | int],
) -> Path:
    path = out_dir / f"{municipality_code}_area.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "municipality_code",
                "season_year",
                "year_range",
                "mapbiomas_soy_pixel_count",
                "mapbiomas_soy_area_ha",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "municipality_code": municipality_code,
                "season_year": season_year,
                "year_range": f"{year_start}-{year_end}",
                "mapbiomas_soy_pixel_count": area_info["mapbiomas_soy_pixel_count"],
                "mapbiomas_soy_area_ha": f"{area_info['mapbiomas_soy_area_ha']:.4f}",
            }
        )
    return path


def season_dir(output_dir: Path, year_start: int, year_end: int) -> Path:
    return output_dir / f"{year_start}-{year_end}"


def uf_codes_from_shapefiles(shapefiles: list[Path]) -> list[int]:
    codes: list[int] = []
    seen: set[int] = set()
    for shp in shapefiles:
        stem = shp.stem
        if len(stem) < 2 or not stem[:2].isdigit():
            continue
        uf = int(stem[:2])
        if uf in UF_BY_CODE and uf not in seen:
            seen.add(uf)
            codes.append(uf)
    return sorted(codes)


def filter_shapefiles_by_states(
    shapefiles: list[Path], state_codes: list[int]
) -> list[Path]:
    keep = set(state_codes)
    out: list[Path] = []
    for shp in shapefiles:
        stem = shp.stem
        if len(stem) >= 2 and stem[:2].isdigit() and int(stem[:2]) in keep:
            out.append(shp)
    return out


def pam_years_spec(args: argparse.Namespace) -> str:
    years = list(args.season_years_list or [])
    if not years and args.season_year is not None:
        years = [int(args.season_year)]
    if not years and args.start_date and args.end_date:
        years = [int(args.end_date[:4])]
    if not years:
        years = [2020, 2021, 2022, 2023, 2024]
    lo, hi = min(years), max(years)
    if hi == lo:
        return str(lo)
    if years == list(range(lo, hi + 1)):
        return f"{lo}-{hi}"
    return ",".join(str(y) for y in years)


def ensure_pam_csv(args: argparse.Namespace, shapefiles: list[Path]) -> Path | None:
    """Return PAM CSV path, downloading from SIDRA when requested or missing."""
    if args.pam_csv is None and not args.download_pam:
        return None
    years = pam_years_spec(args)
    if args.pam_states:
        state_codes = parse_state_codes(args.pam_states)
    else:
        state_codes = uf_codes_from_shapefiles(shapefiles) or parse_state_codes("all")
    dest = args.pam_csv or default_output_path(state_codes, years)
    dest = Path(dest)
    if dest.is_file() and not args.download_pam:
        return dest
    if dest.is_file() and args.download_pam:
        log(f"Re-downloading PAM soy → {dest}")
    else:
        log(f"Downloading PAM soy {years} for {len(state_codes)} UF(s) → {dest}")
    df = fetch_states(years, state_codes)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    n_muni = df["municipality_code"].nunique() if not df.empty else 0
    log(f"PAM: {len(df)} rows, {n_muni} municipalities → {dest}")
    args.pam_csv = dest
    return dest


def load_pam_soy_codes_by_year(path: Path) -> dict[int, set[str]]:
    """Harvest year → municipality codes with planted area or production > 0."""
    out: dict[int, set[str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                year = int(float(row["year"]))
                code = f"{int(float(row['municipality_code'])):07d}"
            except (KeyError, TypeError, ValueError):
                continue
            planted = 0.0
            produced = 0.0
            try:
                planted = float(row.get("area_planted_ha") or 0)
            except (TypeError, ValueError):
                pass
            try:
                produced = float(row.get("production_t") or 0)
            except (TypeError, ValueError):
                pass
            if planted <= 0 and produced <= 0:
                continue
            out.setdefault(year, set()).add(code)
    return out
    return output_dir / f"{year_start}-{year_end}"


def inventory_csv_path(output_dir: Path, year_start: int, year_end: int) -> Path:
    return season_dir(output_dir, year_start, year_end) / "mapbiomas_soy_inventory.csv"


def no_soy_csv_path(output_dir: Path) -> Path:
    return output_dir / "mapbiomas_no_soy.csv"


def _parse_area_row(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    if not rows:
        return None
    row = rows[0]
    code = str(row.get("municipality_code") or path.parent.name).strip()
    try:
        pixels = int(float(row.get("mapbiomas_soy_pixel_count") or 0))
    except (TypeError, ValueError):
        return None
    try:
        area = float(row.get("mapbiomas_soy_area_ha") or 0.0)
    except (TypeError, ValueError):
        area = float(pixels * 30 * 30 / 10000.0)
    return {
        "municipality_code": code,
        "mapbiomas_soy_pixel_count": pixels,
        "mapbiomas_soy_area_ha": area,
    }


def load_inventory_csv(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = str(row.get("municipality_code") or "").strip()
            if not code:
                continue
            try:
                pixels = int(float(row.get("mapbiomas_soy_pixel_count") or 0))
            except (TypeError, ValueError):
                continue
            try:
                area = float(row.get("mapbiomas_soy_area_ha") or 0.0)
            except (TypeError, ValueError):
                area = float(pixels * 30 * 30 / 10000.0)
            out[code] = {
                "municipality_code": code,
                "mapbiomas_soy_pixel_count": pixels,
                "mapbiomas_soy_area_ha": area,
            }
    return out


def harvest_existing_area_sidecars(
    output_dir: Path, year_start: int, year_end: int
) -> dict[str, dict[str, Any]]:
    root = season_dir(output_dir, year_start, year_end)
    out: dict[str, dict[str, Any]] = {}
    if not root.is_dir():
        return out
    for path in root.glob("*/*_area.csv"):
        parsed = _parse_area_row(path)
        if parsed:
            out[parsed["municipality_code"]] = parsed
    return out


def write_inventory_csv(
    path: Path,
    inventory: dict[str, dict[str, Any]],
    season_year: int,
    year_start: int,
    year_end: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    codes = sorted(inventory)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INVENTORY_FIELDS)
        w.writeheader()
        for code in codes:
            info = inventory[code]
            w.writerow(
                {
                    "municipality_code": code,
                    "season_year": season_year,
                    "year_range": f"{year_start}-{year_end}",
                    "mapbiomas_soy_pixel_count": info["mapbiomas_soy_pixel_count"],
                    "mapbiomas_soy_area_ha": f"{float(info['mapbiomas_soy_area_ha']):.4f}",
                }
            )


def append_no_soy_index(
    output_dir: Path,
    season_year: int,
    inventory: dict[str, dict[str, Any]],
) -> int:
    """Rewrite the combined no-soy index for this season's zeros; merge other years."""
    path = no_soy_csv_path(output_dir)
    keep: list[dict[str, str]] = []
    if path.is_file():
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if str(row.get("season_year")) != str(season_year):
                    keep.append(
                        {
                            "season_year": str(row.get("season_year", "")),
                            "municipality_code": str(row.get("municipality_code", "")),
                        }
                    )
    n_zero = 0
    for code, info in sorted(inventory.items()):
        if int(info["mapbiomas_soy_pixel_count"]) <= 0:
            keep.append(
                {"season_year": str(season_year), "municipality_code": code}
            )
            n_zero += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["season_year", "municipality_code"])
        w.writeheader()
        w.writerows(keep)
    return n_zero


def shapefile_to_ee_feature(
    shapefile: Path,
    extra_properties: dict[str, Any] | None = None,
) -> ee.Feature | None:
    """Build an EE feature, or None if the shapefile has no usable geometry."""
    try:
        gdf = gpd.read_file(shapefile)
    except Exception as exc:
        log(f"{shapefile.stem}: cannot read shapefile ({exc})")
        return None
    if gdf.empty:
        log(f"{shapefile.stem}: empty shapefile — skipped")
        return None
    geom_col = gdf.geometry
    valid = geom_col.notna() & ~geom_col.is_empty
    if not bool(valid.any()):
        log(f"{shapefile.stem}: empty geometry — skipped")
        return None
    gdf = gdf.loc[valid]
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    geom = gdf.geometry.iloc[0]
    if geom is None or geom.is_empty:
        log(f"{shapefile.stem}: empty geometry — skipped")
        return None
    try:
        ee_geometry = ee.Geometry(geom.__geo_interface__)
    except Exception as exc:
        log(f"{shapefile.stem}: invalid geometry for GEE ({exc})")
        return None
    return ee.Feature(
        ee_geometry,
        {"municipality_code": shapefile.stem, **(extra_properties or {})},
    )


def _gee_retry(fn, *, attempts: int = 5, what: str):
    delay = 4.0
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last = exc
            msg = str(exc).lower()
            retryable = any(
                s in msg
                for s in ("429", "rate", "quota", "concurrent", "timeout", "reset")
            )
            if not retryable or i == attempts - 1:
                raise
            log(f"{what}: retry {i + 1}/{attempts} after {delay:.0f}s ({exc})")
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise last  # pragma: no cover


def soy_areas_for_shapefiles(
    shapefiles: list[Path],
    season_year: int,
    scale: int,
    chunk_size: int,
) -> dict[str, dict[str, Any]]:
    """Chunked reduceRegions getInfo for missing inventory rows."""
    out: dict[str, dict[str, Any]] = {}
    n = len(shapefiles)
    if n == 0:
        return out

    def run_chunk(chunk: list[Path]) -> list[dict[str, Any]]:
        feats: list[ee.Feature] = []
        for p in chunk:
            feat = shapefile_to_ee_feature(p)
            if feat is None:
                out[p.stem] = {
                    "municipality_code": p.stem,
                    "mapbiomas_soy_pixel_count": 0,
                    "mapbiomas_soy_area_ha": 0.0,
                }
                continue
            feats.append(feat)
        if not feats:
            return []
        fc = ee.FeatureCollection(feats)
        return _gee_retry(
            lambda: mapbiomas_soy_areas_for_collection(fc, season_year, scale),
            what=f"MapBiomas soy chunk n={len(feats)}",
        )

    def split_query(chunk: list[Path]) -> None:
        if not chunk:
            return
        try:
            rows = run_chunk(chunk)
        except Exception as exc:
            if len(chunk) == 1:
                p = chunk[0]
                log(
                    f"{p.stem}: MapBiomas query failed ({exc}) — treating as no soy"
                )
                out[p.stem] = {
                    "municipality_code": p.stem,
                    "mapbiomas_soy_pixel_count": 0,
                    "mapbiomas_soy_area_ha": 0.0,
                }
                return
            mid = max(1, len(chunk) // 2)
            split_query(chunk[:mid])
            split_query(chunk[mid:])
            return
        for row in rows:
            code = str(row.get("municipality_code") or "").strip()
            if code:
                out[code] = row
        missing = [p.stem for p in chunk if p.stem not in out]
        if missing and len(chunk) > 1:
            by_code = {p.stem: p for p in chunk}
            split_query([by_code[c] for c in missing if c in by_code])

    for start in range(0, n, chunk_size):
        chunk = shapefiles[start : start + chunk_size]
        log(
            f"MapBiomas soy inventory {season_year}: "
            f"munis {start + 1}–{start + len(chunk)} / {n}"
        )
        split_query(chunk)
    return out


def build_season_soy_inventory(
    shapefiles: list[Path],
    args: argparse.Namespace,
    season_year: int,
    year_start: int,
    year_end: int,
) -> dict[str, dict[str, Any]]:
    path = inventory_csv_path(args.output_dir, year_start, year_end)
    inventory: dict[str, dict[str, Any]] = {}
    if not args.rebuild_inventory:
        inventory.update(harvest_existing_area_sidecars(args.output_dir, year_start, year_end))
        inventory.update(load_inventory_csv(path))

    wanted = {p.stem for p in shapefiles}
    missing = [p for p in shapefiles if p.stem not in inventory]
    log(
        f"Soy inventory {year_start}-{year_end}: "
        f"{len([c for c in wanted if c in inventory])} cached, {len(missing)} to query on GEE"
    )
    if missing:
        queried = soy_areas_for_shapefiles(
            missing, season_year, args.scale, args.soy_chunk_size
        )
        inventory.update(queried)
        still = [p.stem for p in missing if p.stem not in inventory]
        if still:
            log(f"WARNING: {len(still)} munis missing from soy inventory, e.g. {still[:5]}")

    season_inv = {c: inventory[c] for c in wanted if c in inventory}
    write_inventory_csv(path, season_inv, season_year, year_start, year_end)
    n_zero = append_no_soy_index(args.output_dir, season_year, season_inv)
    n_soy = sum(1 for v in season_inv.values() if int(v["mapbiomas_soy_pixel_count"]) > 0)
    log(
        f"Soy inventory {year_start}-{year_end}: {n_soy} with soy, "
        f"{n_zero} without → {path}"
    )
    return season_inv


def thread_drive_service(credentials_dir: Path) -> Any:
    svc = getattr(_THREAD_LOCAL, "drive", None)
    if svc is None:
        svc = build_drive_service(credentials_dir)
        _THREAD_LOCAL.drive = svc
    return svc


def uf_key(shapefile: Path) -> str:
    stem = shapefile.stem
    return stem[:2] if len(stem) >= 2 else "xx"


def export_chunks(shapefiles: list[Path], chunk_size: int) -> list[list[Path]]:
    """Group by UF so each reduceRegions job stays geographically compact."""
    by_uf: dict[str, list[Path]] = {}
    for p in shapefiles:
        by_uf.setdefault(uf_key(p), []).append(p)
    chunks: list[list[Path]] = []
    for uf in sorted(by_uf):
        items = sorted(by_uf[uf], key=lambda x: x.stem)
        for i in range(0, len(items), chunk_size):
            chunks.append(items[i : i + chunk_size])
    return chunks


def _muni_code_aliases(stem: str) -> list[str]:
    aliases = [stem]
    try:
        n = int(float(stem))
        aliases.extend([str(n), f"{n:07d}"])
    except (TypeError, ValueError):
        pass
    return aliases


def _row_muni_stem(raw: str, lookup: dict[str, str]) -> str | None:
    text = str(raw or "").strip()
    if text in lookup:
        return lookup[text]
    if text.endswith(".0"):
        text = text[:-2]
        if text in lookup:
            return lookup[text]
    try:
        key = str(int(float(text)))
    except (TypeError, ValueError):
        return None
    return lookup.get(key) or lookup.get(f"{int(key):07d}")


def split_chunk_csv(
    chunk_csv: Path,
    shapefiles: list[Path],
    season_root: Path,
    skip_existing: bool = False,
) -> dict[str, int]:
    """Write per-municipality zonal CSVs. Returns stem → row count (-1 if skipped)."""
    lookup: dict[str, str] = {}
    for p in shapefiles:
        for alias in _muni_code_aliases(p.stem):
            lookup[alias] = p.stem

    grouped: dict[str, list[dict[str, str]]] = {}
    with open(chunk_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stem = _row_muni_stem(row.get("municipality_code", ""), lookup)
            if stem is None:
                raw = str(row.get("municipality_code") or "").strip()
                try:
                    stem = f"{int(float(raw)):07d}"
                except (TypeError, ValueError):
                    continue
            grouped.setdefault(stem, []).append(row)

    counts: dict[str, int] = {p.stem: 0 for p in shapefiles}
    for stem, rows in grouped.items():
        out_dir = season_root / stem
        path = out_dir / f"{stem}_zonal.csv"
        if skip_existing and path.is_file():
            counts[stem] = -1
            continue
        rows.sort(key=lambda r: str(r.get("date") or ""))
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ZONAL_CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            for i, row in enumerate(rows):
                out = {k: row.get(k, "") for k in ZONAL_CSV_FIELDS}
                out["system:index"] = str(i)
                out["municipality_code"] = stem
                if not str(out.get(".geo") or "").strip():
                    out[".geo"] = EMPTY_GEO
                w.writerow(out)
        counts[stem] = len(rows)
    return counts


def _drive_csv_candidates(files: list[dict], file_prefix: str) -> list[dict]:
    named = [
        f
        for f in files
        if f.get("name", "").startswith(file_prefix)
        and str(f["name"]).endswith(".csv")
    ]
    if named:
        return named
    return [f for f in files if file_prefix in f.get("name", "")]


def download_drive_csv(
    drive_service: Any,
    drive_folder: str,
    file_prefix: str,
    dest_dir: Path,
    wait_seconds: int = DEFAULT_DRIVE_WAIT_SECONDS,
    poll_interval: int = 30,
) -> Path:
    """Poll Drive until the export CSV appears. GEE COMPLETED often precedes Drive listing."""
    wait_seconds = max(0, int(wait_seconds))
    poll_interval = max(5, int(poll_interval))
    deadline = time.monotonic() + wait_seconds
    fid: str | None = None
    candidates: list[dict] = []
    while True:
        remaining = deadline - time.monotonic()
        if fid is None:
            fid = get_folder_id_by_name(drive_service, drive_folder)
        if fid:
            files = list_files_in_folder(drive_service, fid)
            candidates = _drive_csv_candidates(files, file_prefix)
        if candidates:
            break
        if remaining <= 0:
            if fid is None:
                raise DriveLagError("Drive folder not found after export")
            raise DriveLagError("CSV not found on Drive after export")
        sleep_s = min(poll_interval, remaining)
        where = "folder" if fid is None else "CSV"
        log(
            f"{file_prefix}: {where} not on Drive yet; "
            f"waiting {sleep_s:.0f}s ({remaining:.0f}s left, no new GEE task)"
        )
        time.sleep(sleep_s)
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp_name = candidates[0]["name"]
    download_file_from_drive(drive_service, candidates[0]["id"], tmp_name, dest_dir)
    try:
        delete_file_from_drive(drive_service, candidates[0]["id"])
    except Exception as e:
        log(f"{file_prefix}: Drive cleanup warning: {e}")
    return dest_dir / tmp_name


def year_range_from_chunk_name(name: str) -> tuple[int, int] | None:
    m = re.match(r"^zonal_(\d{4})-(\d{4})", Path(name).name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def year_range_from_chunk_csv(path: Path) -> tuple[int, int] | None:
    parsed = year_range_from_chunk_name(path.name)
    if parsed:
        return parsed
    with open(path, newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f), None)
    if not row:
        return None
    raw_range = str(row.get("year_range") or "").strip()
    if "-" in raw_range:
        a, b = raw_range.split("-", 1)
        try:
            return int(a), int(b)
        except ValueError:
            pass
    sy = row.get("season_year")
    try:
        y = int(float(sy))
        return y - 1, y
    except (TypeError, ValueError):
        return None


def ingest_chunk_files(
    chunk_csvs: list[Path],
    output_dir: Path,
    shapefiles: list[Path],
    skip_existing: bool,
    harvest_years: set[int] | None,
) -> tuple[int, int, int]:
    """Split chunk tables into per-muni CSVs. Returns (wrote, skipped, files)."""
    n_wrote = n_skip = n_files = 0
    for chunk_csv in chunk_csvs:
        yr = year_range_from_chunk_csv(chunk_csv)
        if yr is None:
            log(f"ingest skip (no season in name/rows): {chunk_csv.name}")
            continue
        y1, y2 = yr
        if harvest_years and y2 not in harvest_years:
            continue
        n_files += 1
        season_root = output_dir / f"{y1}-{y2}"
        counts = split_chunk_csv(
            chunk_csv, shapefiles, season_root, skip_existing=skip_existing
        )
        wrote = sum(1 for n in counts.values() if n > 0)
        skipped = sum(1 for n in counts.values() if n < 0)
        n_wrote += wrote
        n_skip += skipped
        log(
            f"ingest {chunk_csv.name}: {wrote} new munis, {skipped} already on disk "
            f"→ {season_root}"
        )
    return n_wrote, n_skip, n_files


def ingest_from_drive(args: argparse.Namespace) -> None:
    drive_service = build_drive_service(args.credentials_dir)
    fid = get_folder_id_by_name(drive_service, args.drive_folder)
    if not fid:
        raise SystemExit(f"Drive folder not found: {args.drive_folder}")
    files = [
        f
        for f in list_files_in_folder(drive_service, fid)
        if str(f.get("name") or "").startswith("zonal_")
        and str(f.get("name") or "").lower().endswith(".csv")
    ]
    harvest = set(args.season_years_list) if args.season_years_list else None
    if harvest:
        kept = []
        for f in files:
            yr = year_range_from_chunk_name(f["name"])
            if yr is None or yr[1] in harvest:
                kept.append(f)
        files = kept
    files.sort(key=lambda f: str(f.get("name") or ""))
    if not files:
        log(f"No zonal_*.csv in Drive folder {args.drive_folder}")
        return
    tmp_dir = Path(args.output_dir).expanduser().resolve() / "_drive_ingest"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    shapefiles = [Path(p).resolve() for p in (args.shapefiles or [])]
    skip = bool(args.skip_existing)
    n_wrote = n_skip = 0
    log(f"Ingest {len(files)} Drive CSVs from {args.drive_folder} (no GEE export)")
    for meta in files:
        name = meta["name"]
        dest = tmp_dir / name
        log(f"download {name}")
        download_file_from_drive(drive_service, meta["id"], name, tmp_dir)
        w, s, _ = ingest_chunk_files(
            [dest],
            Path(args.output_dir).expanduser().resolve(),
            shapefiles,
            skip,
            harvest,
        )
        n_wrote += w
        n_skip += s
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass
        if args.delete_drive_after:
            try:
                delete_file_from_drive(drive_service, meta["id"])
            except Exception as exc:
                log(f"{name}: Drive cleanup warning: {exc}")
    log(f"Ingest done: {n_wrote} new municipality CSVs, {n_skip} already existed")


def ingest_from_dir(args: argparse.Namespace) -> None:
    folder = Path(args.ingest_dir).expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"--ingest-dir is not a directory: {folder}")
    chunks = sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.name.startswith("zonal_") and p.suffix.lower() == ".csv"
    )
    harvest = set(args.season_years_list) if args.season_years_list else None
    shapefiles = [Path(p).resolve() for p in (args.shapefiles or [])]
    log(f"Ingest {len(chunks)} local chunk CSVs from {folder} (no GEE, no Drive)")
    n_wrote, n_skip, n_files = ingest_chunk_files(
        chunks,
        Path(args.output_dir).expanduser().resolve(),
        shapefiles,
        bool(args.skip_existing),
        harvest,
    )
    log(
        f"Ingest done: {n_files} chunk files → {n_wrote} new municipality CSVs, "
        f"{n_skip} already existed"
    )


def run_one_chunk(
    shapefiles: list[Path],
    args: argparse.Namespace,
    date_list: list[str],
    year_start: int,
    year_end: int,
    mapbiomas_year: int,
    inventory: dict[str, dict[str, Any]],
    chunk_index: int,
    total_chunks: int,
) -> list[dict[str, str]]:
    """Export one UF chunk, split into per-muni CSVs. Returns summary rows."""
    if not shapefiles:
        return []
    label = (
        f"chunk {chunk_index + 1}/{total_chunks} uf={uf_key(shapefiles[0])} "
        f"n={len(shapefiles)} ({shapefiles[0].stem}–{shapefiles[-1].stem})"
    )
    try:
        return _export_and_split_chunk(
            shapefiles,
            args,
            date_list,
            year_start,
            year_end,
            mapbiomas_year,
            inventory,
            label,
        )
    except DriveLagError as exc:
        log(
            f"{label}: {exc}. GEE already completed — not starting another export. "
            "Pull later with --ingest-drive --skip-existing."
        )
        return [
            {
                "municipality_code": p.stem,
                "status": "error",
                "note": str(exc),
            }
            for p in shapefiles
        ]
    except Exception as exc:
        if len(shapefiles) > 1:
            log(f"{label}: failed ({exc}); splitting into two smaller chunks")
            mid = max(1, len(shapefiles) // 2)
            left = run_one_chunk(
                shapefiles[:mid],
                args,
                date_list,
                year_start,
                year_end,
                mapbiomas_year,
                inventory,
                chunk_index,
                total_chunks,
            )
            right = run_one_chunk(
                shapefiles[mid:],
                args,
                date_list,
                year_start,
                year_end,
                mapbiomas_year,
                inventory,
                chunk_index,
                total_chunks,
            )
            return left + right
        log(f"[ERROR] {shapefiles[0].stem}: {exc}")
        return [
            {
                "municipality_code": shapefiles[0].stem,
                "status": "error",
                "note": str(exc),
            }
        ]


def _export_and_split_chunk(
    shapefiles: list[Path],
    args: argparse.Namespace,
    date_list: list[str],
    year_start: int,
    year_end: int,
    mapbiomas_year: int,
    inventory: dict[str, dict[str, Any]],
    label: str,
) -> list[dict[str, str]]:
    feats: list[ee.Feature] = []
    skipped_geom: list[dict[str, str]] = []
    season_root = season_dir(args.output_dir, year_start, year_end)

    for shp in shapefiles:
        info = inventory.get(shp.stem) or {
            "mapbiomas_soy_pixel_count": 0,
            "mapbiomas_soy_area_ha": 0.0,
        }
        write_area_sidecar(
            season_root / shp.stem,
            shp.stem,
            mapbiomas_year,
            year_start,
            year_end,
            info,
        )
        soy_px = int(info["mapbiomas_soy_pixel_count"])
        feat = shapefile_to_ee_feature(shp, {"soy_pixel_count": soy_px})
        if feat is None:
            skipped_geom.append(
                {
                    "municipality_code": shp.stem,
                    "status": "no_soy",
                    "note": "empty shapefile geometry",
                }
            )
            continue
        feats.append(feat)

    if not feats:
        return skipped_geom

    fc_munis = ee.FeatureCollection(feats)
    log(f"{label}: starting reduceRegions table export...")
    fc = chunk_zonal_feature_collection(
        fc_munis,
        date_list[0],
        date_list[-1],
        mapbiomas_year,
        args.cloud_pct,
        args.cloud_score_threshold,
        args.scale,
    )
    file_prefix = (
        f"zonal_{year_start}-{year_end}_{shapefiles[0].stem}_{shapefiles[-1].stem}"
        f"_n{len(shapefiles)}"
    )
    description = file_prefix[:100]
    task_id = _gee_retry(
        lambda: start_table_export_task(
            fc,
            description,
            args.drive_folder,
            file_prefix,
            selectors=ZONAL_SELECTORS,
        ),
        what=f"{label} export start",
    )
    wait_for_gee_task(task_id, poll_interval=args.poll_interval)

    tmp_dir = season_root / "_chunks"
    drive_service = thread_drive_service(args.credentials_dir)
    chunk_csv = download_drive_csv(
        drive_service,
        args.drive_folder,
        file_prefix,
        tmp_dir,
        wait_seconds=getattr(args, "drive_wait", DEFAULT_DRIVE_WAIT_SECONDS),
        poll_interval=args.poll_interval,
    )
    counts = split_chunk_csv(chunk_csv, shapefiles, season_root)
    try:
        chunk_csv.unlink(missing_ok=True)
    except Exception:
        pass

    rows: list[dict[str, str]] = list(skipped_geom)
    n_ok = 0
    n_empty = 0
    for shp in shapefiles:
        if any(r["municipality_code"] == shp.stem for r in skipped_geom):
            continue
        n_rows = int(counts.get(shp.stem, 0))
        zonal_path = season_root / shp.stem / f"{shp.stem}_zonal.csv"
        if n_rows <= 0:
            if zonal_path.is_file():
                zonal_path.unlink()
            n_empty += 1
            rows.append(
                {
                    "municipality_code": shp.stem,
                    "status": "no_imagery",
                    "note": "no clear soy S2 dates",
                }
            )
            continue
        n_ok += 1
        rows.append(
            {
                "municipality_code": shp.stem,
                "status": "success",
                "note": f"{n_rows} dates",
            }
        )
    log(f"{label}: saved {n_ok} CSVs ({n_empty} munis with no clear soy dates)")
    return rows


def run_season_batch(
    args: argparse.Namespace,
    shapefiles: list[Path],
    season_year: int | None,
    date_list: list[str],
    year_start: int,
    year_end: int,
) -> tuple[int, int]:
    """Run all municipalities for one season. Returns (ok, total)."""
    mapbiomas_year = season_year if season_year is not None else year_end
    print(
        f"\n=== Season {year_start}-{year_end} "
        f"(MapBiomas={mapbiomas_year}): {len(shapefiles)} municipalities ==="
    )

    rows: list[dict[str, str]] = []
    pam_by_year = getattr(args, "pam_codes_by_year", None)
    if pam_by_year is not None:
        pam_keep = pam_by_year.get(int(mapbiomas_year), set())

        def _pam_code(stem: str) -> str:
            try:
                return f"{int(stem):07d}"
            except ValueError:
                return stem

        with_pam: list[Path] = []
        no_pam: list[Path] = []
        for p in shapefiles:
            if _pam_code(p.stem) in pam_keep:
                with_pam.append(p)
            else:
                no_pam.append(p)
        log(
            f"PAM {mapbiomas_year}: {len(with_pam)}/{len(shapefiles)} municipalities "
            f"have planted area or production; skipping {len(no_pam)} without PAM soy"
        )
        for shp in no_pam:
            rows.append(
                {
                    "municipality_code": shp.stem,
                    "status": "no_pam",
                    "note": f"no PAM soy in {mapbiomas_year}",
                }
            )
        shapefiles = with_pam

    inventory = build_season_soy_inventory(
        shapefiles, args, mapbiomas_year, year_start, year_end
    )
    if args.inventory_only:
        # Still write area sidecars for no-soy / cache completeness
        for shp in shapefiles:
            info = inventory.get(shp.stem)
            if info is None:
                continue
            out_dir = season_dir(args.output_dir, year_start, year_end) / shp.stem
            write_area_sidecar(
                out_dir, shp.stem, mapbiomas_year, year_start, year_end, info
            )
        return 0, len(shapefiles)

    soy_shapefiles: list[Path] = []
    no_soy: list[Path] = []
    for p in shapefiles:
        info = inventory.get(p.stem)
        if info is not None and int(info["mapbiomas_soy_pixel_count"]) <= 0:
            no_soy.append(p)
        else:
            soy_shapefiles.append(p)
    log(
        f"Season {year_start}-{year_end}: skipping {len(no_soy)} municipalities "
        f"with no MapBiomas soy; {len(soy_shapefiles)} with soy"
    )

    summary_path = season_dir(args.output_dir, year_start, year_end) / "download_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    for shp in no_soy:
        info = inventory.get(shp.stem) or {
            "mapbiomas_soy_pixel_count": 0,
            "mapbiomas_soy_area_ha": 0.0,
        }
        out_dir = season_dir(args.output_dir, year_start, year_end) / shp.stem
        write_area_sidecar(
            out_dir, shp.stem, mapbiomas_year, year_start, year_end, info
        )
        rows.append(
            {
                "municipality_code": shp.stem,
                "status": "no_soy",
                "note": "mapbiomas_soy_area_ha=0",
            }
        )

    todo: list[Path] = []
    n_skip = 0
    for shp in soy_shapefiles:
        zonal_path = (
            season_dir(args.output_dir, year_start, year_end)
            / shp.stem
            / f"{shp.stem}_zonal.csv"
        )
        if args.skip_existing and zonal_path.is_file():
            n_skip += 1
            rows.append(
                {
                    "municipality_code": shp.stem,
                    "status": "skipped",
                    "note": "exists",
                }
            )
            continue
        todo.append(shp)

    chunks = export_chunks(todo, args.export_chunk_size)
    log(
        f"Season {year_start}-{year_end}: {n_skip} existing CSVs skipped; "
        f"{len(todo)} munis in {len(chunks)} reduceRegions chunks "
        f"(size={args.export_chunk_size}, workers={args.workers})"
    )

    n_chunks = len(chunks)

    def _job(item: tuple[int, list[Path]]) -> list[dict[str, str]]:
        i, chunk = item
        try:
            return run_one_chunk(
                chunk,
                args,
                date_list,
                year_start,
                year_end,
                mapbiomas_year,
                inventory,
                i,
                n_chunks,
            )
        except Exception as e:
            log(f"[ERROR] chunk {i + 1}/{n_chunks}: {e}")
            return [
                {
                    "municipality_code": p.stem,
                    "status": "error",
                    "note": str(e),
                }
                for p in chunk
            ]

    if n_chunks == 0:
        pass
    elif args.workers <= 1:
        for i, chunk in enumerate(chunks):
            rows.extend(_job((i, chunk)))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_job, (i, chunk)) for i, chunk in enumerate(chunks)]
            for fut in as_completed(futs):
                rows.extend(fut.result())

    rows.sort(key=lambda r: r["municipality_code"])
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["municipality_code", "status", "note"])
        w.writeheader()
        w.writerows(rows)

    ok = sum(1 for r in rows if r["status"] == "success")
    n_pam = sum(1 for r in rows if r["status"] == "no_pam")
    print(
        f"Season {year_start}-{year_end}: {ok}/{len(soy_shapefiles)} soy munis succeeded "
        f"({len(no_soy)} MapBiomas no-soy, {n_pam} no PAM soy skipped) → {summary_path}"
    )
    return ok, len(rows)


def main() -> None:
    args = parse_args()

    if args.ingest_dir is not None:
        ingest_from_dir(args)
        return
    if args.ingest_drive:
        ingest_from_drive(args)
        return

    shapefiles = [Path(p).resolve() for p in args.shapefiles]
    for shp in shapefiles:
        if not shp.is_file():
            sys.exit(f"Shapefile not found: {shp}")
    if args.states_label:
        log(
            f"Restricted to UF {args.states_label}: "
            f"{len(shapefiles)} municipality shapefiles"
        )

    pam_path = ensure_pam_csv(args, shapefiles)
    args.pam_codes_by_year = None
    if pam_path is not None:
        args.pam_codes_by_year = load_pam_soy_codes_by_year(pam_path)
        n_pairs = sum(len(v) for v in args.pam_codes_by_year.values())
        log(
            f"PAM filter {pam_path}: {n_pairs} municipality-years "
            f"across {sorted(args.pam_codes_by_year)}"
        )

    initialize_gee(args.gee_project, args.gee_account)

    drive_service = build_drive_service(args.credentials_dir)
    _THREAD_LOCAL.drive = drive_service

    if args.start_date and args.end_date:
        date_list, year_start, year_end = get_date_list_from_args(args)
        print(
            f"Zonal mean export: {len(shapefiles)} municipalities, "
            f"custom {date_list[0]}–{date_list[-1]}  "
            f"chunks={args.export_chunk_size} workers={args.workers}"
        )
        ok, total = run_season_batch(
            args, shapefiles, None, date_list, year_start, year_end
        )
        print(f"\nDone: {ok}/{total} succeeded.")
    else:
        seasons = args.season_years_list
        print(
            f"Zonal mean export: {len(shapefiles)} municipalities × "
            f"{len(seasons)} seasons {seasons}  chunks={args.export_chunk_size} "
            f"workers={args.workers}"
            + ("  [inventory-only]" if args.inventory_only else "")
        )
        grand_ok = grand_total = 0
        for sy in seasons:
            date_list, year_start, year_end = get_date_list_for_season(sy)
            ok, total = run_season_batch(
                args, shapefiles, sy, date_list, year_start, year_end
            )
            grand_ok += ok
            grand_total += total
        print(f"\nDone: {grand_ok}/{grand_total} municipality-seasons succeeded.")

    pam_ref = args.pam_csv or "files/pam_soy_pr_2019_2025.csv"
    print(
        "Next: merge area sidecars with IBGE PAM and filter:\n"
        "  python data_download/filter_zonal_area_coverage.py \\\n"
        f"    --zonal-root {args.output_dir} \\\n"
        f"    --pam-csv {pam_ref}"
    )


if __name__ == "__main__":
    main()
