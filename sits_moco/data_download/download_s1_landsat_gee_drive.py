#!/usr/bin/env python3
"""
Download Brazilian soybean Sentinel-1 or Landsat daily imagery from Google Earth Engine.

Same pipeline as download_soy_gee_drive.py (GEE → Drive → local raw/), but for:
  - Sentinel-1 IW GRD (VV/VH, dB), optional ASCENDING/DESCENDING filter
  - Landsat 8/9 Collection 2 Tier-1 Level-2 surface reflectance (scaled)

By default exports on a shared 30 m EPSG:4326 pixel grid (Projection.atScale) so
S1/Landsat pixels align with S2 downloads that use the same alignment. Pass
--s2-reference path/to/s2.tif to lock exactly to an existing S2 GeoTIFF grid.

Uses MapBiomas year-X soy classification to mask pixels. By default fetches the
soybean growth cycle Oct (X-1)--Mar (X); optional --start-date/--end-date override.

Usage:
  python data_download/download_s1_landsat_gee_drive.py --sensor s1 --shapefile path/to/mun.shp --season-year 2024
  python data_download/download_s1_landsat_gee_drive.py --sensor s1 --shapefile-dir files/shapefiles --season-year 2024 --resolution 30
  python data_download/download_s1_landsat_gee_drive.py --sensor s1 --shapefile a.shp --season-year 2024 --s2-reference path/to/s2.tif
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import ee
import geopandas as gpd
from drive_api import (
    build_drive_service,
    delete_file_from_drive,
    download_file_from_drive,
    get_folder_id_by_name,
    list_files_in_folder,
)
from gee_export import (
    DEFAULT_GEE_ACCOUNT,
    DEFAULT_GEE_PROJECT,
    build_daily_landsat_image,
    build_daily_s1_image,
    crs_transform_from_reference_tiff,
    get_aligned_crs_transform,
    get_dates_with_landsat_scenes,
    get_dates_with_s1_scenes,
    initialize_gee,
    start_export_task,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
DEFAULT_CREDENTIALS_DIR = Path("data")
SENSOR_DEFAULTS = {
    "s1": {
        "resolution": 30,
        "output_dir": Path("files/raw_tiff_s1"),
        "drive_folder": "GEE_Soy_Export_S1",
    },
    "landsat": {
        "resolution": 30,
        "output_dir": Path("files/raw_tiff_landsat"),
        "drive_folder": "GEE_Soy_Export_Landsat",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Brazilian soy daily Sentinel-1 or Landsat imagery "
            "via GEE → Drive → local."
        )
    )
    parser.add_argument(
        "--sensor",
        type=str,
        required=True,
        choices=["s1", "landsat"],
        help="Imagery source: s1 (Sentinel-1 GRD) or landsat (L8+L9 C2 L2)",
    )
    shape_group = parser.add_mutually_exclusive_group(required=True)
    shape_group.add_argument(
        "--shapefile",
        type=Path,
        action="append",
        dest="shapefiles",
        help="Path to municipality shapefile (.shp); can be repeated for batch",
    )
    shape_group.add_argument(
        "--shapefile-dir",
        type=Path,
        dest="shapefile_dir",
        help=(
            "Directory of municipality folders; each subfolder is a municipality "
            "code with a .shp"
        ),
    )
    parser.add_argument(
        "--season-year",
        type=int,
        default=None,
        metavar="X",
        help="Season year X: Oct (X-1)–Mar (X). Required if --start-date/--end-date not set",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Start date (inclusive). Use with --end-date instead of --season-year",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="End date (inclusive). Use with --start-date instead of --season-year",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Export scale in meters (default: 30 for s1 and landsat). Used for aligned grid.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base output directory (default: files/raw_tiff_s1 or files/raw_tiff_landsat)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help=(
            "Directory to save raw GEE files "
            "(default: {output_dir}/{year_start}-{year_end}/{municipality}/raw)."
        ),
    )
    parser.add_argument(
        "--drive-folder",
        type=str,
        default=None,
        help="Google Drive folder for temporary exports (sensor-specific default)",
    )
    parser.add_argument(
        "--s2-reference",
        type=Path,
        default=None,
        help=(
            "Optional Sentinel-2 GeoTIFF whose CRS/transform/dimensions pin the export "
            "grid (exact pixel match to that file). If omitted, uses a global "
            "EPSG:4326 grid at --resolution via Projection.atScale (same grid for all "
            "municipalities/sensors at that scale)."
        ),
    )
    parser.add_argument(
        "--no-align-grid",
        action="store_true",
        help="Disable fixed crsTransform (legacy scale+region export; pixels may not match S2)",
    )
    parser.add_argument(
        "--gee-project",
        type=str,
        default=DEFAULT_GEE_PROJECT,
        help=f"GEE project ID (default: {DEFAULT_GEE_PROJECT})",
    )
    parser.add_argument(
        "--gee-account",
        type=str,
        default=DEFAULT_GEE_ACCOUNT,
        help=f"Google account for Earth Engine auth (default: {DEFAULT_GEE_ACCOUNT})",
    )
    parser.add_argument(
        "--cloud-pct",
        type=float,
        default=20.0,
        help="Max scene cloud cover %% for Landsat metadata filter (default: 20; ignored for s1)",
    )
    parser.add_argument(
        "--orbit-pass",
        type=str,
        default=None,
        choices=["ASCENDING", "DESCENDING"],
        help="Optional Sentinel-1 orbit pass filter (ignored for landsat)",
    )
    parser.add_argument(
        "--max-download-workers",
        type=int,
        default=20,
        help="Parallel workers for downloading from Drive",
    )
    parser.add_argument(
        "--credentials-dir",
        type=Path,
        default=DEFAULT_CREDENTIALS_DIR,
        help="Directory with credentials.json and token.json for Drive (default: data)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between GEE task polls (default: 30)",
    )
    args = parser.parse_args()

    defaults = SENSOR_DEFAULTS[args.sensor]
    if args.resolution is None:
        args.resolution = defaults["resolution"]
    if args.output_dir is None:
        args.output_dir = defaults["output_dir"]
    if args.drive_folder is None:
        args.drive_folder = defaults["drive_folder"]

    if args.s2_reference is not None:
        args.s2_reference = Path(args.s2_reference).resolve()
        if not args.s2_reference.is_file():
            parser.error(f"--s2-reference not found: {args.s2_reference}")

    if args.shapefile_dir is not None:
        d = Path(args.shapefile_dir).resolve()
        if not d.is_dir():
            parser.error(f"--shapefile-dir is not a directory: {d}")
        args.shapefiles = []
        for subdir in sorted(d.iterdir()):
            if subdir.is_dir():
                shps = list(subdir.glob("*.shp"))
                if shps:
                    args.shapefiles.append(shps[0])
        if not args.shapefiles:
            parser.error(f"No .shp files found in subfolders of {d}")
    else:
        args.shapefiles = args.shapefiles or []
        args.shapefiles = [Path(p).resolve() for p in args.shapefiles]

    if args.start_date is not None or args.end_date is not None:
        if args.start_date is None or args.end_date is None:
            parser.error("Both --start-date and --end-date must be set together")
        try:
            datetime.strptime(args.start_date, "%Y-%m-%d")
            datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError as e:
            parser.error(f"Dates must be YYYY-MM-DD: {e}")
        if args.start_date > args.end_date:
            parser.error("--start-date must be <= --end-date")
    else:
        if args.season_year is None:
            parser.error("Set either --season-year or both --start-date and --end-date")

    return args


def output_path_for_date(
    output_dir: Path,
    municipality_code: str,
    year_start: int,
    year_end: int,
    date_str: str,
) -> Path:
    """Path for one day under {output_dir}/{year_start}-{year_end}/{municipality}/."""
    y, m, d = date_str.split("-")
    return (
        output_dir
        / f"{year_start}-{year_end}"
        / municipality_code
        / f"{municipality_code}_{y}_{m}_{d}.tiff"
    )


def raw_dir_for_date(
    output_dir: Path,
    municipality_code: str,
    year_start: int,
    year_end: int,
) -> Path:
    """Directory for raw GEE tiles."""
    return output_dir / f"{year_start}-{year_end}" / municipality_code / "raw"


def download_raw_only(
    service,
    file_list: list[tuple[str, str]],
    raw_dir: Path,
) -> None:
    """Download Drive files with exact GEE filenames into raw_dir."""
    raw_dir = Path(raw_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    for fid, fname in file_list:
        download_file_from_drive(service, fid, fname, raw_dir)


def _date_from_raw_filename(municipality_code: str, filename: str) -> str | None:
    """Extract YYYY-MM-DD from GEE raw filename."""
    stem = Path(filename).stem
    if not stem.startswith(municipality_code + "_"):
        return None
    suffix = stem[len(municipality_code) + 1 :]
    parts = suffix.rsplit("-", 2)
    if (
        len(parts) == 3
        and len(parts[1]) == 10
        and parts[1].isdigit()
        and len(parts[2]) == 10
        and parts[2].isdigit()
    ):
        return parts[0]
    if len(suffix) == 10 and suffix[4] == "-" and suffix[7] == "-":
        return suffix
    return None


def raw_has_date(raw_dir: Path, municipality_code: str, date_str: str) -> bool:
    """True if raw_dir contains at least one file for that date."""
    if not raw_dir.exists():
        return False
    prefix = f"{municipality_code}_{date_str}"
    return any(f.name.startswith(prefix) for f in raw_dir.iterdir() if f.is_file())


def get_season_dates(season_year: int) -> list[str]:
    """Return list of YYYY-MM-DD from (season_year-1)-10-01 to season_year-03-31."""
    start = datetime(season_year - 1, 10, 1)
    end = datetime(season_year, 3, 31)
    out = []
    d = start
    while d <= end:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def get_date_list_from_args(args: argparse.Namespace) -> tuple[list[str], int, int]:
    """Return (list of YYYY-MM-DD, year_start, year_end) from args."""
    if args.start_date and args.end_date:
        start = datetime.strptime(args.start_date, "%Y-%m-%d")
        end = datetime.strptime(args.end_date, "%Y-%m-%d")
        out = []
        d = start
        while d <= end:
            out.append(d.strftime("%Y-%m-%d"))
            d += timedelta(days=1)
        years = [int(d[:4]) for d in out]
        return out, min(years), max(years)
    dates = get_season_dates(args.season_year)
    years = [int(d[:4]) for d in dates]
    return dates, min(years), max(years)


def _dates_with_coverage(ee_geometry: ee.Geometry, args: argparse.Namespace, start_date: str, end_date: str) -> list[str]:
    """Query GEE for dates that have at least one scene for the selected sensor."""
    if args.sensor == "s1":
        return get_dates_with_s1_scenes(
            ee_geometry, start_date, end_date, orbit_pass=args.orbit_pass
        )
    return get_dates_with_landsat_scenes(
        ee_geometry, start_date, end_date, cloud_pct=args.cloud_pct
    )


def _build_daily_image(
    ee_geometry: ee.Geometry,
    date_str: str,
    mapbiomas_year: int,
    args: argparse.Namespace,
) -> Optional[ee.Image]:
    """Build one daily image for the selected sensor."""
    if args.sensor == "s1":
        return build_daily_s1_image(
            ee_geometry,
            date_str,
            mapbiomas_year,
            orbit_pass=args.orbit_pass,
        )
    return build_daily_landsat_image(
        ee_geometry,
        date_str,
        mapbiomas_year,
        cloud_pct=args.cloud_pct,
    )


def _resolve_export_grid(
    args: argparse.Namespace,
) -> tuple[Optional[str], Optional[list[float]], Optional[tuple[int, int]]]:
    """
    Return (crs, crs_transform, dimensions) for aligned export.

    dimensions is only set when locking to a reference TIFF.
    """
    if args.no_align_grid:
        return None, None, None
    if args.s2_reference is not None:
        crs, crs_transform, dimensions = crs_transform_from_reference_tiff(
            args.s2_reference
        )
        return crs, crs_transform, dimensions
    crs = "EPSG:4326"
    crs_transform = get_aligned_crs_transform(args.resolution, crs=crs)
    return crs, crs_transform, None


def run_one_municipality(
    shapefile: Path,
    args: argparse.Namespace,
    date_list: list[str],
    year_start: int,
    year_end: int,
    drive_service: Any,
    run_start_time: float,
    total_municipalities: int,
    municipality_index: int,
    export_crs: Optional[str] = None,
    export_crs_transform: Optional[list[float]] = None,
    export_dimensions: Optional[tuple[int, int]] = None,
) -> tuple[list[str], list[dict]]:
    """Run export+download for one municipality. Returns (downloaded_dates, failed_entries)."""
    municipality_code = shapefile.stem
    output_mun_dir = args.output_dir / f"{year_start}-{year_end}" / municipality_code
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapbiomas_year = args.season_year if args.season_year is not None else year_end
    sensor_label = "S1" if args.sensor == "s1" else "Landsat"

    log = lambda msg: print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

    gdf = gpd.read_file(shapefile)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    geom = gdf.geometry.values[0]
    ee_geometry = ee.Geometry(geom.__geo_interface__)

    raw_d = (
        Path(args.raw_dir).resolve()
        if getattr(args, "raw_dir", None) is not None
        else raw_dir_for_date(args.output_dir, municipality_code, year_start, year_end)
    )
    if raw_d.exists():
        existing_dates = set()
        for f in raw_d.iterdir():
            if f.is_file():
                d = _date_from_raw_filename(municipality_code, f.name)
                if d:
                    existing_dates.add(d)
        already_count = len(existing_dates)
    else:
        already_count = 0
    total_days_in_range = len(date_list)

    start_date = date_list[0]
    end_date = date_list[-1]
    log(f"Municipality {municipality_code}: querying dates with {sensor_label} coverage...")
    dates_to_check = _dates_with_coverage(ee_geometry, args, start_date, end_date)
    log(
        f"Municipality {municipality_code} ({municipality_index + 1}/{total_municipalities}): "
        f"{len(dates_to_check)} dates with possible {sensor_label} coverage "
        f"(of {total_days_in_range} in range), "
        f"{already_count} already downloaded (elapsed: {time.time() - run_start_time:.0f}s)"
    )

    pending_tasks: dict[str, tuple[str, str]] = {}
    downloaded_dates: list[str] = []
    failed_entries: list[dict] = []
    failed_lock = Lock()
    max_workers = max(1, args.max_download_workers)

    def poll_and_download() -> None:
        """Process completed/failed tasks; download to raw/ then delete from Drive."""
        task_list = ee.data.getTaskList()
        to_remove_failed: list[str] = []
        download_jobs: list[tuple[str, list[tuple[str, str]], Path]] = []
        for task in task_list:
            tid = task.get("id")
            if tid not in pending_tasks:
                continue
            state = task.get("state")
            date_str, file_prefix = pending_tasks[tid]
            if state in ("SUCCEEDED", "COMPLETED"):
                fid = get_folder_id_by_name(drive_service, args.drive_folder)
                if not fid:
                    log(
                        "Drive folder not found after task completed; skipping download"
                    )
                    to_remove_failed.append(tid)
                    continue
                files = list_files_in_folder(drive_service, fid)
                candidates = [
                    f
                    for f in files
                    if f["name"].startswith(file_prefix) or file_prefix in f["name"]
                ]
                if not candidates:
                    log(f"No file found for prefix {file_prefix}; task {tid}")
                    to_remove_failed.append(tid)
                    continue
                local_path = output_path_for_date(
                    args.output_dir,
                    municipality_code,
                    year_start,
                    year_end,
                    date_str,
                )
                file_list = [(c["id"], c["name"]) for c in candidates]
                download_jobs.append((tid, file_list, local_path))
            elif state == "FAILED":
                err = task.get("error_message", "unknown")
                log(f"Task failed for {date_str}: {err}")
                with failed_lock:
                    failed_entries.append({"date": date_str, "error": err})
                to_remove_failed.append(tid)

        successful_downloads: set[str] = set()
        download_failed_tids: set[str] = set()
        if download_jobs:

            def do_one(tid_file_list_path):
                tid, file_list, path = tid_file_list_path
                try:
                    raw_dir = path.parent / "raw"
                    download_raw_only(drive_service, file_list, raw_dir)
                    for fid, _ in file_list:
                        delete_file_from_drive(drive_service, fid)
                    return (tid, path, None)
                except Exception as e:
                    return (tid, path, str(e))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(do_one, j): j for j in download_jobs}
                for fut in as_completed(futures):
                    tid, path, err = fut.result()
                    if err:
                        date_str = pending_tasks[tid][0]
                        log(
                            f"Download failed for {path.name}: {err} "
                            "(logged; re-run script later to retry)"
                        )
                        with failed_lock:
                            failed_entries.append({"date": date_str, "error": str(err)})
                        download_failed_tids.add(tid)
                    else:
                        successful_downloads.add(tid)
                        log(f"Downloaded {path.name} -> raw/ (removed from Drive)")

        for tid in to_remove_failed:
            pending_tasks.pop(tid, None)
        for tid in download_failed_tids:
            pending_tasks.pop(tid, None)
        for tid in successful_downloads:
            pending_tasks.pop(tid, None)

    dates_to_check_set = set(dates_to_check)
    fid = get_folder_id_by_name(drive_service, args.drive_folder)
    if fid:
        prefix_to_files: dict[str, list[tuple[str, str]]] = {}
        for f in list_files_in_folder(drive_service, fid):
            name = f.get("name", "")
            if not name.startswith(municipality_code + "_") or not (
                name.endswith(".tif") or name.endswith(".tiff")
            ):
                continue
            stem = name.rsplit(".", 1)[0]
            parts = stem.rsplit("-", 2)
            if (
                len(parts) == 3
                and parts[1].isdigit()
                and len(parts[1]) == 10
                and parts[2].isdigit()
                and len(parts[2]) == 10
            ):
                prefix = parts[0]
            else:
                prefix = stem
            prefix_to_files.setdefault(prefix, []).append((f["id"], name))
        for prefix, file_list in prefix_to_files.items():
            suffix = prefix[len(municipality_code) + 1 :]
            if len(suffix) == 10 and suffix[4] == "-" and suffix[7] == "-":
                date_str = suffix
            elif suffix.count("_") == 2:
                parts = suffix.split("_")
                if len(parts) == 3:
                    date_str = f"{parts[0]}-{parts[1]}-{parts[2]}"
                else:
                    continue
            else:
                continue
            if date_str not in dates_to_check_set:
                continue
            if raw_d.exists() and raw_has_date(raw_d, municipality_code, date_str):
                log(f"Already have raw files for {date_str}; removing from Drive")
                for fid, _ in file_list:
                    try:
                        delete_file_from_drive(drive_service, fid)
                    except Exception as e:
                        log(f"Could not delete orphan from Drive: {e}")
                continue
            try:
                download_raw_only(drive_service, file_list, raw_d)
                for fid, _ in file_list:
                    delete_file_from_drive(drive_service, fid)
                log(
                    f"Resumed: downloaded {len(file_list)} raw tile(s) for {date_str} "
                    "(removed from Drive)"
                )
            except Exception as e:
                log(f"Resume download failed for {date_str}: {e}")

    for date_str in dates_to_check:
        if raw_d.exists() and raw_has_date(raw_d, municipality_code, date_str):
            continue

        poll_and_download()

        log(f"Building {sensor_label} image for {date_str}...")
        img = _build_daily_image(ee_geometry, date_str, mapbiomas_year, args)
        if img is None:
            log(f"No imagery for {date_str}; skipping")
            continue

        file_prefix = f"{municipality_code}_{date_str}"
        try:
            task_id = start_export_task(
                img,
                description=file_prefix,
                folder_name=args.drive_folder,
                region=ee_geometry,
                scale=args.resolution,
                crs=export_crs or "EPSG:4326",
                crs_transform=export_crs_transform,
                dimensions=export_dimensions,
            )
        except Exception as e:
            log(f"Export start failed for {date_str}: {e}")
            with failed_lock:
                failed_entries.append({"date": date_str, "error": str(e)})
            continue

        pending_tasks[task_id] = (date_str, file_prefix)
        log(f"Started export for {date_str} (task {task_id})")

    half = max(1, args.poll_interval // 2)
    try:
        while pending_tasks:
            n = len(pending_tasks)
            log(f"Waiting for remaining exports... ({n} task(s) pending)")
            time.sleep(half)
            if pending_tasks:
                log(f"Still waiting... ({len(pending_tasks)} task(s) pending)")
            time.sleep(max(0, args.poll_interval - half))
            try:
                poll_and_download()
            except Exception as e:
                log(f"Poll/download error (will retry): {e}")
    except KeyboardInterrupt:
        log(
            f"Interrupted. {len(pending_tasks)} export(s) still in progress. "
            "Re-run the same command to resume: the script will download any completed files "
            "already on Drive, then continue with missing dates."
        )
        raise SystemExit(1)

    downloaded_dates = []
    if raw_d.exists():
        for f in raw_d.iterdir():
            if f.is_file():
                d = _date_from_raw_filename(municipality_code, f.name)
                if d:
                    downloaded_dates.append(d)
        downloaded_dates = sorted(set(downloaded_dates))
    final_count = len(downloaded_dates)
    elapsed = time.time() - run_start_time
    log(
        f"Municipality {municipality_code} done: {final_count}/{total_days_in_range} "
        f"raw date(s) (elapsed: {elapsed:.0f}s)"
    )

    success_set = set(downloaded_dates)
    failed_by_date = {e["date"]: e.get("error", "") for e in failed_entries}
    log_path = output_mun_dir / "download_log.csv"
    output_mun_dir.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "status", "note"])
        for date_str in sorted(dates_to_check):
            if date_str in success_set:
                status, note = "success", ""
            elif date_str in failed_by_date:
                status, note = "failed", failed_by_date[date_str]
            else:
                status, note = "no_imagery", ""
            w.writerow([date_str, status, note])
    log(f"Wrote download log: {log_path}")

    return (downloaded_dates, failed_entries)


def main() -> None:
    args = parse_args()

    shapefiles = [Path(p).resolve() for p in args.shapefiles]
    for shp in shapefiles:
        if not shp.exists():
            sys.exit(f"Shapefile not found: {shp}")

    date_list, year_start, year_end = get_date_list_from_args(args)
    run_start_time = time.time()
    log = lambda msg: print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
    total_files_before = 0
    total_files_after = 0

    initialize_gee(args.gee_project, args.gee_account)

    drive_service = build_drive_service(args.credentials_dir)

    sensor_label = "Sentinel-1" if args.sensor == "s1" else "Landsat 8/9"
    export_crs, export_crs_transform, export_dimensions = _resolve_export_grid(args)
    if export_crs_transform is not None:
        align_note = (
            f"aligned grid crs={export_crs}, transform={export_crs_transform}"
            + (
                f", dimensions={export_dimensions[0]}x{export_dimensions[1]}"
                if export_dimensions
                else ""
            )
        )
    else:
        align_note = "legacy scale+region (no fixed crsTransform)"
    log(
        f"Batch ({sensor_label}): {len(shapefiles)} municipality(ies), "
        f"date range {date_list[0]}–{date_list[-1]} ({year_start}-{year_end}), "
        f"{len(date_list)} days, scale={args.resolution}m, drive={args.drive_folder}, "
        f"{align_note}"
    )

    for i, shapefile in enumerate(shapefiles):
        mun_code = shapefile.stem
        out_mun_dir = args.output_dir / f"{year_start}-{year_end}" / mun_code
        raw_dir = (
            Path(args.raw_dir).resolve()
            if getattr(args, "raw_dir", None) is not None
            else (out_mun_dir / "raw")
        )
        if raw_dir.exists():
            total_files_before += len(list(raw_dir.iterdir()))
        run_one_municipality(
            shapefile,
            args,
            date_list,
            year_start,
            year_end,
            drive_service,
            run_start_time,
            len(shapefiles),
            i,
            export_crs=export_crs,
            export_crs_transform=export_crs_transform,
            export_dimensions=export_dimensions,
        )
        if raw_dir.exists():
            total_files_after += len(list(raw_dir.iterdir()))

    elapsed = time.time() - run_start_time
    new_files = total_files_after - total_files_before
    log(f"Done. Total elapsed: {elapsed:.0f}s. New raw files this run: {new_files}.")


if __name__ == "__main__":
    main()
