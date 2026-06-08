#!/usr/bin/env python3
"""
Download Brazilian soybean Sentinel-2 daily imagery from Google Earth Engine.

Uses MapBiomas year-X soy classification to mask pixels. By default fetches the
soybean growth cycle Oct (X-1)--Mar (X); optional --start-date/--end-date override.
Exports to Google Drive, then downloads the exact GEE files to local .../raw/ and deletes them from Drive (no merge).
To merge raw tiles into .tiff later, use preprocessing/merge_gee_tiles_to_tiff.py (raw files are never deleted locally).
Output: raw_tiff/{year_start}-{year_end}/{municipality}/raw/<exact GEE filename>.
Per-municipality download_log.csv (date, status, note) for retries and progress.

No-data convention: masked pixels (no S2 coverage, clouds, non-soy) are written as
NO_DATA_VALUE (-9999) and the TIFF nodata tag is set so downstream (clip, training)
can exclude them and only use real observations.

Usage:
  python data_download/download_soy_gee_drive.py --shapefile path/to/municipality.shp --season-year 2024
  python data_download/download_soy_gee_drive.py --shapefile a.shp --shapefile b.shp --start-date 2023-10-01 --end-date 2024-03-31
  python data_download/download_soy_gee_drive.py --shapefile-dir path/to/shapefiles --season-year 2024 --max-download-workers 4
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
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
from gee_export import build_daily_image, get_dates_with_s2_scenes, start_export_task

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
DEFAULT_CREDENTIALS_DIR = Path("data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Brazilian soy daily S2 imagery via GEE → Drive → local."
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
        help="Directory of municipality folders (e.g. files/shapefiles/); each subfolder is a municipality code with a .shp (one run per shapefile found)",
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
        default=30,
        help="Export scale in meters (default: 30). Used in output path and GEE export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("files/raw_tiff"),
        help="Base output directory (default: files/raw_tiff)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Directory to save raw GEE files (default: {output_dir}/{year_start}-{year_end}/{municipality}/raw).",
    )
    parser.add_argument(
        "--drive-folder",
        type=str,
        default="GEE_Soy_Export",
        help="Google Drive folder name for temporary exports (default: GEE_Soy_Export)",
    )
    parser.add_argument(
        "--gee-project",
        type=str,
        default="soybean-yield-prediction",
        help="GEE project ID (default: from ee.Initialize or default project)",
    )
    parser.add_argument(
        "--cloud-pct",
        type=float,
        default=20.0,
        help="Max scene cloud percentage (metadata filter) for S2 scenes and for 'dates with coverage' query (default: 20)",
    )
    parser.add_argument(
        "--cloud-score-threshold",
        type=float,
        default=0.60,
        help="Cloud Score+ clear pixel threshold (default: 0.60)",
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

    # Resolve shapefile list
    if args.shapefile_dir is not None:
        d = Path(args.shapefile_dir).resolve()
        if not d.is_dir():
            parser.error(f"--shapefile-dir is not a directory: {d}")
        # One folder per municipality; each folder contains that municipality's .shp
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

    # Date range: either (start_date, end_date) or season_year
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


# -----------------------------------------------------------------------------
# Date range: soy cycle (Oct X-1 to Mar X) or custom start/end
# -----------------------------------------------------------------------------
def output_path_for_date(
    output_dir: Path,
    municipality_code: str,
    year_start: int,
    year_end: int,
    date_str: str,
) -> Path:
    """Path for one day: {output_dir}/{year_start}-{year_end}/{municipality}/{municipality}_{year}_{month}_{day}.tiff"""
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
    """Directory for raw GEE tiles: {output_dir}/{year_start}-{year_end}/{municipality}/raw/"""
    return output_dir / f"{year_start}-{year_end}" / municipality_code / "raw"


def download_raw_only(
    service,
    file_list: list[tuple[str, str]],
    raw_dir: Path,
) -> None:
    """
    Download all Drive files with their exact GEE filenames into raw_dir. No merge, no delete from Drive.
    file_list: [(file_id, file_name), ...].
    """
    raw_dir = Path(raw_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    for fid, fname in file_list:
        download_file_from_drive(service, fid, fname, raw_dir)


def _date_from_raw_filename(municipality_code: str, filename: str) -> str | None:
    """Extract YYYY-MM-DD from GEE raw filename (e.g. 4100103_2022-10-02.tif or 4100103_2022-10-02-0000000000-0000009984.tif)."""
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
        return parts[0]  # 2022-10-02
    if len(suffix) == 10 and suffix[4] == "-" and suffix[7] == "-":
        return suffix
    return None


def raw_has_date(raw_dir: Path, municipality_code: str, date_str: str) -> bool:
    """True if raw_dir contains at least one file for that date (GEE prefix = municipality_code_date_str)."""
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


# -----------------------------------------------------------------------------
# Main: single municipality run + batch entrypoint
# -----------------------------------------------------------------------------
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
) -> tuple[list[str], list[dict]]:
    """Run export+download for one municipality. Returns (downloaded_dates, failed_entries)."""
    municipality_code = shapefile.stem
    output_mun_dir = args.output_dir / f"{year_start}-{year_end}" / municipality_code
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapbiomas_year = args.season_year if args.season_year is not None else year_end

    log = lambda msg: print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

    gdf = gpd.read_file(shapefile)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    geom = gdf.geometry.values[0]
    ee_geometry = ee.Geometry(geom.__geo_interface__)

    # Progress: count already-downloaded days (raw/ only, no GEE)
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

    # Only check dates where Sentinel-2 has at least one scene (one GEE query for whole range)
    start_date = date_list[0]
    end_date = date_list[-1]
    log(f"Municipality {municipality_code}: querying dates with S2 coverage...")
    dates_to_check = get_dates_with_s2_scenes(
        ee_geometry, start_date, end_date, args.cloud_pct
    )
    log(
        f"Municipality {municipality_code} ({municipality_index + 1}/{total_municipalities}): "
        f"{len(dates_to_check)} dates with possible S2 coverage (of {total_days_in_range} in range), "
        f"{already_count} already downloaded (elapsed: {time.time() - run_start_time:.0f}s)"
    )

    pending_tasks: dict[str, tuple[str, str]] = {}
    downloaded_dates: list[str] = []
    failed_entries: list[dict] = []
    failed_lock = Lock()
    max_workers = max(1, args.max_download_workers)

    def poll_and_download() -> None:
        """Process completed/failed tasks; download to raw/ then delete from Drive.
        Only remove from pending_tasks on definitive GEE state (SUCCEEDED/COMPLETED/FAILED).
        On download failure we log and remove (no retry this run); re-run script later to retry via resume step.
        """
        task_list = ee.data.getTaskList()
        to_remove_failed: list[str] = (
            []
        )  # tids with definitive FAILED (or success but no file to download)
        download_jobs: list[tuple[str, list[tuple[str, str]], Path]] = (
            []
        )  # (tid, [(file_id, file_name), ...], local_path)
        for task in task_list:
            tid = task.get("id")
            if tid not in pending_tasks:
                continue
            state = task.get("state")
            date_str, file_prefix = pending_tasks[tid]
            # Only act on definitive terminal states from GEE; ignore RUNNING, READY, etc.
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
                # All tiles for this date: (tid, [(file_id, file_name), ...], output_path)
                file_list = [(c["id"], c["name"]) for c in candidates]
                download_jobs.append((tid, file_list, local_path))
            elif state == "FAILED":
                err = task.get("error_message", "unknown")
                log(f"Task failed for {date_str}: {err}")
                with failed_lock:
                    failed_entries.append({"date": date_str, "error": err})
                to_remove_failed.append(tid)
            # else: RUNNING, READY, etc. — do not remove; wait for definitive state

        # Download in parallel; remove from pending on success or on failure (log only, no retry this run)
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
                    # Do not delete from Drive on failure — keep file so we can retry later (logged in failed_entries)
                    return (tid, path, str(e))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(do_one, j): j for j in download_jobs}
                for fut in as_completed(futures):
                    tid, path, err = fut.result()
                    if err:
                        date_str = pending_tasks[tid][0]
                        log(
                            f"Download failed for {path.name}: {err} (logged; re-run script later to retry)"
                        )
                        with failed_lock:
                            failed_entries.append({"date": date_str, "error": str(err)})
                        download_failed_tids.add(tid)
                    else:
                        successful_downloads.add(tid)
                        log(f"Downloaded {path.name} -> raw/ (removed from Drive)")

        # Remove on definitive outcome: GEE failed, download failed (log only), or download succeeded
        for tid in to_remove_failed:
            pending_tasks.pop(tid, None)
        for tid in download_failed_tids:
            pending_tasks.pop(tid, None)
        for tid in successful_downloads:
            pending_tasks.pop(tid, None)

    # Resume: download any files already on Drive from a previous run (e.g. after interrupt).
    # Group by export prefix (GEE may have written multiple tiles per date: prefix-XXXXXXXXXX-YYYYYYYYYY.tif).
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
            # Strip GEE tile suffix -NNNNNNNNNN-NNNNNNNNNN if present
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
            # prefix is e.g. state_41_2022-10-02 -> date_str = 2022-10-02
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
                    f"Resumed: downloaded {len(file_list)} raw tile(s) for {date_str} (removed from Drive)"
                )
            except Exception as e:
                log(f"Resume download failed for {date_str}: {e}")
                # Do not delete from Drive — keep file so we can retry later

    for date_str in dates_to_check:
        if raw_d.exists() and raw_has_date(raw_d, municipality_code, date_str):
            continue

        poll_and_download()

        log(f"Building image for {date_str}...")
        img = build_daily_image(
            ee_geometry,
            date_str,
            mapbiomas_year,
            args.cloud_pct,
            args.cloud_score_threshold,
        )
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
            )
        except Exception as e:
            log(f"Export start failed for {date_str}: {e}")
            with failed_lock:
                failed_entries.append({"date": date_str, "error": str(e)})
            continue

        pending_tasks[task_id] = (date_str, file_prefix)
        log(f"Started export for {date_str} (task {task_id})")

    # Sleep in chunks so we don't sit silent for 60s (avoids being killed as "idle"); log once per cycle
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

    # All GEE tasks are done and all files have been downloaded from Drive at this point.

    # Build downloaded list from raw/ on disk (source of truth for status log)
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
        f"Municipality {municipality_code} done: {final_count}/{total_days_in_range} raw date(s) (elapsed: {elapsed:.0f}s)"
    )

    # Per-municipality log written only here, after all tasks ended and downloads completed
    # (every attempted date and its status: success / failed / no_imagery)
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

    try:
        if args.gee_project:
            ee.Initialize(project=args.gee_project)
        else:
            ee.Initialize()
    except ee.EEException as e:
        if "Please authenticate" in str(e) or "credentials" in str(e).lower():
            sys.exit(
                'GEE not authenticated. Run in a terminal: python -c "import ee; ee.Authenticate()"'
            )
        raise

    drive_service = build_drive_service(args.credentials_dir)

    log(
        f"Batch: {len(shapefiles)} municipality(ies), date range {date_list[0]}–{date_list[-1]} ({year_start}-{year_end}), {len(date_list)} days"
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
        )
        if raw_dir.exists():
            total_files_after += len(list(raw_dir.iterdir()))

    elapsed = time.time() - run_start_time
    new_files = total_files_after - total_files_before
    log(f"Done. Total elapsed: {elapsed:.0f}s. New raw files this run: {new_files}.")


if __name__ == "__main__":
    main()
