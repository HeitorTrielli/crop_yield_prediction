#!/usr/bin/env python3
"""
Daily soy-masked Sentinel-2 GeoTIFFs from Brazil Data Cube (no Earth Engine).

Same role as download_soy_gee_drive.py + merge_gee_tiles_to_tiff.py: one mosaic
per day over a polygon (state shapefile), MapBiomas soy, nodata -9999, then
clip with preprocessing/clip_tiffs_to_shapefiles.py.

Cloud filter is not Cloud Score+. Scenes with eo:cloud_cover >= --cloud-pct are
dropped; remaining pixels must be Sen2Cor SCL 4 or 5 (default).

Usage:
  python data_download/download_state_shapefile.py --state RS --output files/rs_state/rs.shp

  python data_download/download_soy_bdc_daily_tiff.py \\
      --shapefile files/rs_state/rs.shp --season-year 2024 \\
      --skip-existing --workers 2

  python data_download/download_soy_bdc_daily_tiff.py \\
      --shapefile files/rs_state/rs.shp --season-year 2022 \\
      --profile-full --profile-workers 1,2,4,8 --profile-block-size 2048

  python preprocessing/clip_tiffs_to_shapefiles.py \\
      --tiff-dir files/raw_tiff/2023-2024/rs --shapefile-dir files/shapefiles -j 20
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
from tqdm import tqdm

from bdc_zonal import (
    DEFAULT_SCL_CLEAR,
    MAPBIOMAS_COVERAGE_URL,
    S2_L2A_COLLECTION,
    group_items_by_date,
    mapbiomas_url,
    profile_daily_soy_windows,
    search_s2_items,
    silence_bdc_numpy_warnings,
    write_daily_soy_tiff,
    write_mapbiomas_soy_mask,
)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_season_years_spec(seasons_spec: str | None, single: int | None) -> list[int]:
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
    out: list[int] = []
    seen: set[int] = set()
    for y in years:
        if y not in seen:
            seen.add(y)
            out.append(y)
    return out


def season_dates(season_year: int) -> tuple[str, str, int, int]:
    start = datetime(season_year - 1, 10, 1)
    end = datetime(season_year, 3, 31)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), start.year, end.year


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--shapefile",
        type=Path,
        required=True,
        help="Polygon to mosaic (e.g. files/rs_state/rs.shp). Stem becomes the folder name.",
    )
    p.add_argument("--season-year", type=int, default=None)
    p.add_argument("--season-years", type=str, default=None)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("files/raw_tiff"),
        help="Base dir (default: files/raw_tiff/{YYYY-YYYY}/{stem}/)",
    )
    p.add_argument("--cloud-pct", type=float, default=20.0)
    p.add_argument(
        "--scl-clear",
        type=str,
        default="4,5",
        help="Sen2Cor SCL classes treated as clear (default: 4,5)",
    )
    p.add_argument("--collection", type=str, default=S2_L2A_COLLECTION)
    p.add_argument(
        "--mapbiomas-url-template",
        type=str,
        default=MAPBIOMAS_COVERAGE_URL,
    )
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument(
        "--adaptive-block",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Larger tiles on granule-heavy dates (8–14 → 1024, ≥15 → 2048). "
            "Light dates keep --block-size. Default: on."
        ),
    )
    p.add_argument(
        "--heavy-granules",
        type=int,
        default=15,
        help="Granule count treated as a heavy date (default: 15)",
    )
    p.add_argument(
        "--heavy-workers",
        type=int,
        default=2,
        help="Parallel workers for heavy dates (default: 2). Light dates use --workers.",
    )
    p.add_argument(
        "--heavy-block-size",
        type=int,
        default=2048,
        help="Tile size for heavy dates when --adaptive-block is on (default: 2048)",
    )
    p.add_argument(
        "--harmonize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Subtract 1000 DN on spectral bands when processing baseline >= 04.00 "
            "(GEE S2_SR_HARMONIZED). Default: on."
        ),
    )
    p.add_argument(
        "--limit-dates",
        type=int,
        default=None,
        help="Only the first N dates with STAC items (debug)",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="Benchmark --workers and --block-size on a few soy windows, then exit",
    )
    p.add_argument(
        "--profile-full",
        action="store_true",
        help=(
            "Time complete daily mosaics (real GeoTIFF writes, same as the downloader). "
            "Writes into the season folder; picks dates not already on disk."
        ),
    )
    p.add_argument(
        "--profile-windows",
        type=int,
        default=12,
        help="Soy tiles per timed trial when using --profile (default: 12)",
    )
    p.add_argument(
        "--profile-dates",
        type=int,
        default=None,
        help=(
            "Dates per worker trial for --profile-full (default: one wave = worker count). "
            "Must be >= workers to measure that width."
        ),
    )
    p.add_argument(
        "--profile-block-size",
        type=int,
        default=None,
        help="Skip the block-size sweep and lock workers to this tile size",
    )
    p.add_argument(
        "--profile-block-sizes",
        type=str,
        default=None,
        help=(
            "Comma-separated tile sizes for --profile-full (default: 512,1024,2048). "
            "Ignored if --profile-block-size is set."
        ),
    )
    p.add_argument(
        "--profile-workers",
        type=str,
        default=None,
        help="Comma-separated worker counts (default: 1,2,4; --profile-full: 2,4,8)",
    )
    p.add_argument(
        "--profile-pick",
        choices=("median", "heavy"),
        default="median",
        help="Which unfinished dates --profile-full uses (default: median granule count)",
    )
    args = p.parse_args()
    args.season_years_list = parse_season_years_spec(args.season_years, args.season_year)
    if not args.season_years_list:
        p.error("Set --season-year or --season-years")
    args.workers = max(1, int(args.workers))
    args.heavy_workers = max(1, int(args.heavy_workers))
    args.heavy_granules = max(1, int(args.heavy_granules))
    args.heavy_block_size = max(16, int(args.heavy_block_size))
    args.scl_clear = frozenset(int(x) for x in args.scl_clear.split(",") if x.strip())
    if not args.scl_clear:
        args.scl_clear = DEFAULT_SCL_CLEAR
    return args


def read_geometry(shapefile: Path):
    gdf = gpd.read_file(shapefile)
    if gdf.empty:
        return None
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf.geometry.union_all()


def tiff_name(prefix: str, date_str: str) -> str:
    y, m, d = date_str.split("-")
    return f"{prefix}_{y}_{m}_{d}.tiff"


def tiff_dates_on_disk(season_root: Path, prefix: str) -> set[str]:
    found: set[str] = set()
    for p in season_root.glob(f"{prefix}_*.tiff"):
        parts = p.stem.rsplit("_", 3)
        if len(parts) != 4:
            continue
        _, y, m, d = parts
        if len(y) == 4 and len(m) == 2 and len(d) == 2:
            found.add(f"{y}-{m}-{d}")
    return found


def _parse_profile_workers(spec: str | None, cpu: int, *, full: bool) -> list[int]:
    if spec:
        raw = [int(x) for x in spec.split(",") if x.strip()]
    else:
        raw = [2, 4, 8] if full else [1, 2, 4]
    out = [w for w in raw if 1 <= w <= max(1, cpu)]
    if not out:
        sys.exit("Profile: no valid --profile-workers for this CPU count")
    return out


def mosaic_block_size(n_granules: int, args: argparse.Namespace) -> int:
    """Light dates keep --block-size; busy dates use larger COG windows."""
    if not getattr(args, "adaptive_block", True):
        return max(16, int(args.block_size))
    n = int(n_granules)
    if n >= int(args.heavy_granules):
        return max(16, int(args.heavy_block_size))
    if n >= 8:
        return 1024
    return max(16, int(args.block_size))


def _pick_unfinished_dates(
    by_date: dict[str, list],
    existing: set[str],
    n: int,
    pick: str = "median",
) -> list[str]:
    pending = [d for d in by_date if d not in existing]
    if len(pending) < n:
        sys.exit(
            f"Profile: need {n} unfinished dates, have {len(pending)} "
            f"({len(existing)} already on disk)"
        )
    pending.sort(key=lambda d: (len(by_date[d]), d))
    if pick == "heavy":
        return pending[-n:]
    start = max(0, (len(pending) - n) // 2)
    return pending[start : start + n]


def _prepare_profile(
    args: argparse.Namespace, geometry
) -> tuple[int, Path, str, Path, dict[str, list]]:
    season_year = int(args.season_years_list[0])
    start_date, end_date, year_start, year_end = season_dates(season_year)
    prefix = Path(args.shapefile).stem
    season_root = (
        Path(args.output_dir).expanduser().resolve() / f"{year_start}-{year_end}" / prefix
    )
    season_root.mkdir(parents=True, exist_ok=True)
    mask_path = season_root / "_mapbiomas_soy_mask.tif"
    if mask_path.is_file():
        log(f"Profile: reusing soy mask {mask_path}")
    else:
        mb_path = mapbiomas_url(season_year, args.mapbiomas_url_template)
        log(f"Profile: writing soy mask from {mb_path}")
        write_mapbiomas_soy_mask(geometry, mb_path, mask_path)

    log(f"Profile: STAC search {args.collection} {start_date}..{end_date}")
    items = search_s2_items(
        geometry, start_date, end_date, args.cloud_pct, collection=args.collection
    )
    by_date = group_items_by_date(items)
    if not by_date:
        sys.exit("Profile: no STAC dates")
    return season_year, season_root, prefix, mask_path, by_date


def _profile_job(payload: dict[str, Any]) -> dict[str, Any]:
    return profile_daily_soy_windows(
        payload["date"],
        payload["records"],
        payload["soy_mask_path"],
        scl_clear=frozenset(payload["scl_clear"]),
        block_size=int(payload["block_size"]),
        max_soy_windows=int(payload["max_soy_windows"]),
        harmonize=bool(payload.get("harmonize", True)),
    )


def run_profile(args: argparse.Namespace, geometry) -> None:
    """Time block-size and worker grids on a small soy-window sample (no output TIFFs)."""
    season_year, season_root, prefix, mask_path, by_date = _prepare_profile(args, geometry)
    del season_root, prefix
    ranked = sorted(by_date, key=lambda d: len(by_date[d]), reverse=True)
    sample_date = ranked[0]
    n_windows = max(1, int(args.profile_windows))
    cpu = os.cpu_count() or 4
    log(
        f"Profile: busiest date {sample_date} ({len(by_date[sample_date])} granules), "
        f"{n_windows} soy windows/trial, cpu={cpu}"
    )

    block_rows: list[dict[str, Any]] = []
    if args.profile_block_size:
        best_bs = max(16, int(args.profile_block_size))
        log(f"Profile: skipping block sweep; locked block-size={best_bs}")
    else:
        block_sizes = [512, 1024]
        for bs in block_sizes:
            log(f"Profile block-size={bs} ...")
            row = profile_daily_soy_windows(
                sample_date,
                by_date[sample_date],
                str(mask_path),
                scl_clear=args.scl_clear,
                block_size=bs,
                max_soy_windows=n_windows,
                harmonize=bool(args.harmonize),
            )
            block_rows.append(row)
            log(
                f"  {bs}: {row['seconds']}s  {row['soy_windows']} windows  "
                f"{row['window_mpix']} Mpx  {row['sec_per_mpix']} s/Mpx  "
                f"{row['granule_reads']} granule reads"
            )
        scored = [r for r in block_rows if r.get("sec_per_mpix")]
        best_block = (
            min(scored, key=lambda r: r["sec_per_mpix"]) if scored else block_rows[0]
        )
        best_bs = int(best_block["block_size"])

    worker_choices = _parse_profile_workers(args.profile_workers, cpu, full=False)
    worker_dates = ranked[: max(worker_choices)]
    worker_rows: list[dict[str, Any]] = []
    for nw in worker_choices:
        dates = worker_dates[:nw]
        payloads = [
            {
                "date": d,
                "records": by_date[d],
                "soy_mask_path": str(mask_path),
                "scl_clear": list(args.scl_clear),
                "block_size": best_bs,
                "max_soy_windows": n_windows,
                "harmonize": bool(args.harmonize),
            }
            for d in dates
        ]
        log(f"Profile workers={nw} dates={dates} ...")
        t0 = time.perf_counter()
        if nw == 1:
            parts = [_profile_job(payloads[0])]
        else:
            with ProcessPoolExecutor(
                max_workers=nw, initializer=silence_bdc_numpy_warnings
            ) as pool:
                parts = list(pool.map(_profile_job, payloads))
        wall = time.perf_counter() - t0
        work = sum(float(p["seconds"]) for p in parts)
        worker_rows.append(
            {
                "workers": nw,
                "dates": len(dates),
                "wall_s": round(wall, 2),
                "cpu_s": round(work, 2),
                "speedup": round(work / wall, 2) if wall else None,
                "sec_per_date_wall": round(wall / len(dates), 2),
            }
        )
        log(
            f"  workers={nw}: wall {wall:.1f}s  cpu-sum {work:.1f}s  "
            f"speedup {work / wall:.2f}x  ({len(dates)} dates)"
        )

    if block_rows:
        print("\n=== block-size (higher throughput = lower s/Mpx) ===")
        print(f"{'block':>8} {'sec':>8} {'windows':>8} {'Mpx':>8} {'s/Mpx':>8}")
        for r in block_rows:
            print(
                f"{r['block_size']:>8} {r['seconds']:>8} {r['soy_windows']:>8} "
                f"{r['window_mpix']:>8} {r['sec_per_mpix']:>8}"
            )
    print(
        f"\n=== workers at block-size={best_bs} "
        "(higher speedup vs 1 process is better; watch INPE/disk) ==="
    )
    print(f"{'workers':>8} {'dates':>8} {'wall_s':>8} {'cpu_s':>8} {'speedup':>8}")
    for r in worker_rows:
        print(
            f"{r['workers']:>8} {r['dates']:>8} {r['wall_s']:>8} "
            f"{r['cpu_s']:>8} {r['speedup']:>8}"
        )

    # Prefer the largest worker count that keeps ≥70% of linear speedup.
    recommended_workers = worker_rows[0]["workers"]
    for r in worker_rows:
        expected = float(r["workers"])
        got = float(r["speedup"] or 0)
        if got >= 0.7 * expected or r["workers"] <= 2:
            recommended_workers = r["workers"]
        else:
            break

    print(
        f"\nRecommended: --workers {recommended_workers} --block-size {best_bs}\n"
        f"python data_download/download_soy_bdc_daily_tiff.py \\\n"
        f"  --shapefile {args.shapefile} --season-year {season_year} \\\n"
        f"  --skip-existing --workers {recommended_workers} --block-size {best_bs}"
    )


def _date_job(payload: dict[str, Any]) -> dict[str, str]:
    return write_daily_soy_tiff(
        payload["date"],
        payload["records"],
        payload["soy_mask_path"],
        payload["out_path"],
        scl_clear=frozenset(payload["scl_clear"]),
        block_size=int(payload["block_size"]),
        harmonize=bool(payload.get("harmonize", True)),
    )


def _date_job_timed(payload: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    row = dict(_date_job(payload))
    row["seconds"] = round(time.perf_counter() - t0, 2)
    return row


def _mosaic_wave(
    *,
    workers: int,
    dates: list[str],
    by_date: dict[str, list],
    mask_path: Path,
    season_root: Path,
    prefix: str,
    scl_clear: list[int],
    block_size: int,
    harmonize: bool,
) -> tuple[list[dict[str, Any]], float]:
    payloads = [
        {
            "date": d,
            "records": by_date[d],
            "soy_mask_path": str(mask_path),
            "out_path": str(season_root / tiff_name(prefix, d)),
            "scl_clear": list(scl_clear),
            "block_size": block_size,
            "harmonize": bool(harmonize),
        }
        for d in dates
    ]
    log(f"Profile-full workers={workers} block={block_size} dates={dates} ...")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers, initializer=silence_bdc_numpy_warnings
    ) as pool_ex:
        parts = list(pool_ex.map(_date_job_timed, payloads))
    wall = time.perf_counter() - t0
    for p in parts:
        log(
            f"  {p.get('date')}: {p.get('status')} {p.get('note', '')} "
            f"in {p.get('seconds')}s"
        )
    work = sum(float(p["seconds"]) for p in parts)
    ok = sum(1 for p in parts if p.get("status") == "success")
    sec_per_date = wall / len(dates) if dates else None
    log(
        f"  workers={workers} block={block_size}: wall {wall:.1f}s  "
        f"{ok}/{len(dates)} ok  {sec_per_date:.1f}s/date  "
        f"speedup {work / wall:.2f}x"
    )
    return parts, wall


def run_profile_full(args: argparse.Namespace, geometry) -> None:
    """Time complete daily mosaics (same write_daily_soy_tiff path as the downloader)."""
    season_year, season_root, prefix, mask_path, by_date = _prepare_profile(args, geometry)
    cpu = os.cpu_count() or 4
    worker_choices = _parse_profile_workers(args.profile_workers, cpu, full=True)
    if args.profile_block_size:
        block_sizes: list[int] = []
        best_bs = max(16, int(args.profile_block_size))
        log(f"Profile-full: skipping block sweep; locked block-size={best_bs}")
    else:
        if args.profile_block_sizes:
            block_sizes = [
                int(x) for x in args.profile_block_sizes.split(",") if x.strip()
            ]
        else:
            block_sizes = [512, 1024, 2048]
        block_sizes = [max(16, b) for b in block_sizes]
        if not block_sizes:
            sys.exit("Profile-full: no block sizes")
        best_bs = block_sizes[0]
    dates_per_trial = (
        max(1, int(args.profile_dates)) if args.profile_dates else None
    )
    n_needed = len(block_sizes) + sum(
        dates_per_trial if dates_per_trial is not None else nw for nw in worker_choices
    )
    existing = tiff_dates_on_disk(season_root, prefix)
    pool = _pick_unfinished_dates(
        by_date, existing, n_needed, pick=str(args.profile_pick)
    )
    log(
        f"Profile-full: {season_root}  cpu={cpu}  "
        f"{len(existing)} dates on disk, {len(by_date) - len(existing)} remaining"
    )
    log(
        f"  block sizes={block_sizes or [best_bs]}  workers={worker_choices}  "
        f"dates={pool}  granules={[len(by_date[d]) for d in pool]}"
    )

    cursor = 0
    block_rows: list[dict[str, Any]] = []
    for bs in block_sizes:
        date_str = pool[cursor]
        cursor += 1
        parts, wall = _mosaic_wave(
            workers=1,
            dates=[date_str],
            by_date=by_date,
            mask_path=mask_path,
            season_root=season_root,
            prefix=prefix,
            scl_clear=list(args.scl_clear),
            block_size=bs,
            harmonize=bool(args.harmonize),
        )
        part = parts[0]
        block_rows.append(
            {
                "block_size": bs,
                "date": date_str,
                "granules": len(by_date[date_str]),
                "status": part.get("status"),
                "wall_s": round(wall, 2),
                "ok": 1 if part.get("status") == "success" else 0,
            }
        )

    if block_rows:
        scored = [r for r in block_rows if r["ok"] and r["wall_s"] > 0]
        if scored:
            best_bs = int(min(scored, key=lambda r: r["wall_s"])["block_size"])
        print("\n=== full mosaic block-size (1 worker, 1 date; lower wall is better) ===")
        print(f"{'block':>8} {'date':>12} {'granules':>8} {'status':>12} {'wall_s':>10}")
        for r in block_rows:
            print(
                f"{r['block_size']:>8} {r['date']:>12} {r['granules']:>8} "
                f"{r['status']:>12} {r['wall_s']:>10}"
            )
        log(f"Profile-full: winning block-size={best_bs}")

    worker_rows: list[dict[str, Any]] = []
    for nw in worker_choices:
        n_dates = dates_per_trial if dates_per_trial is not None else nw
        dates = pool[cursor : cursor + n_dates]
        cursor += n_dates
        parts, wall = _mosaic_wave(
            workers=nw,
            dates=dates,
            by_date=by_date,
            mask_path=mask_path,
            season_root=season_root,
            prefix=prefix,
            scl_clear=list(args.scl_clear),
            block_size=best_bs,
            harmonize=bool(args.harmonize),
        )
        work = sum(float(p["seconds"]) for p in parts)
        ok = sum(1 for p in parts if p.get("status") == "success")
        sec_per_date = wall / len(dates) if dates else None
        worker_rows.append(
            {
                "workers": nw,
                "dates": len(dates),
                "ok": ok,
                "wall_s": round(wall, 2),
                "cpu_s": round(work, 2),
                "speedup": round(work / wall, 2) if wall else None,
                "sec_per_date_wall": round(sec_per_date, 2) if sec_per_date else None,
            }
        )

    print(
        f"\n=== full mosaic workers at block-size={best_bs} "
        "(lower s/date is better) ==="
    )
    print(
        f"{'workers':>8} {'dates':>8} {'ok':>8} {'wall_s':>10} {'cpu_s':>10} "
        f"{'s/date':>10} {'speedup':>8}"
    )
    for r in worker_rows:
        print(
            f"{r['workers']:>8} {r['dates']:>8} {r['ok']:>8} {r['wall_s']:>10} "
            f"{r['cpu_s']:>10} {r['sec_per_date_wall']:>10} {r['speedup']:>8}"
        )

    scored_w = [r for r in worker_rows if r.get("sec_per_date_wall") and r["ok"] > 0]
    best_w = (
        min(scored_w, key=lambda r: r["sec_per_date_wall"])
        if scored_w
        else worker_rows[0]
    )
    recommended_workers = best_w["workers"]
    print(
        f"\nRecommended: --workers {recommended_workers} --block-size {best_bs}\n"
        f"python data_download/download_soy_bdc_daily_tiff.py \\\n"
        f"  --shapefile {args.shapefile} --season-year {season_year} \\\n"
        f"  --skip-existing --workers {recommended_workers} --block-size {best_bs}"
    )
    print(
        "\nTIFFs from this profile were written into the season folder "
        "(same files the downloader would produce)."
    )


def _run_date_jobs(jobs: list[dict[str, Any]], workers: int, finish) -> None:
    if not jobs:
        return
    if workers <= 1 or len(jobs) <= 1:
        for payload in jobs:
            finish(_date_job(payload))
        return
    with ProcessPoolExecutor(
        max_workers=workers, initializer=silence_bdc_numpy_warnings
    ) as pool:
        futures = {pool.submit(_date_job, payload): payload["date"] for payload in jobs}
        for fut in as_completed(futures):
            date_str = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                row = {
                    "date": date_str,
                    "status": "error",
                    "note": str(exc),
                    "path": "",
                }
            finish(row)


def run_season(args: argparse.Namespace, season_year: int, geometry) -> None:
    start_date, end_date, year_start, year_end = season_dates(season_year)
    prefix = Path(args.shapefile).stem
    season_root = Path(args.output_dir).expanduser().resolve() / f"{year_start}-{year_end}" / prefix
    season_root.mkdir(parents=True, exist_ok=True)
    mb_path = mapbiomas_url(season_year, args.mapbiomas_url_template)
    mask_path = season_root / "_mapbiomas_soy_mask.tif"
    log(
        f"Season {year_start}-{year_end} (MapBiomas={season_year}): "
        f"writing soy mask from {mb_path}"
    )
    info = write_mapbiomas_soy_mask(geometry, mb_path, mask_path)
    log(
        f"Soy mask {info['width']}x{info['height']} "
        f"({info['soy_pixels']} soy pixels) -> {mask_path}"
    )
    if info["soy_pixels"] <= 0:
        log("No MapBiomas soy in polygon; skipping S2")
        return

    log(f"STAC search {args.collection} {start_date}..{end_date} cloud<{args.cloud_pct}")
    items = search_s2_items(
        geometry,
        start_date,
        end_date,
        args.cloud_pct,
        collection=args.collection,
    )
    by_date = group_items_by_date(items)
    dates = sorted(by_date)
    if args.limit_dates:
        dates = dates[: int(args.limit_dates)]
    log(f"{len(items)} STAC items -> {len(dates)} dates with usable granules")
    if not items:
        log(
            "BDC S2_L2A-1 has no daily COGs for this polygon/dates "
            "(RS coverage starts ~Oct 2021; earlier years exist in other regions). "
            "Use GEE for those seasons, or S2-16D-2 16-day composites."
        )

    jobs: list[dict[str, Any]] = []
    summary: list[dict[str, str]] = []
    for date_str in dates:
        dest = season_root / tiff_name(prefix, date_str)
        if args.skip_existing and dest.is_file():
            summary.append(
                {"date": date_str, "status": "success", "note": "already on disk"}
            )
            continue
        n_granules = len(by_date[date_str])
        jobs.append(
            {
                "date": date_str,
                "records": by_date[date_str],
                "soy_mask_path": str(mask_path),
                "out_path": str(dest),
                "scl_clear": list(args.scl_clear),
                "block_size": mosaic_block_size(n_granules, args),
                "harmonize": bool(args.harmonize),
            }
        )

    n_skip = len(dates) - len(jobs)
    heavy_cut = int(args.heavy_granules)
    if args.adaptive_block:
        light_jobs = [j for j in jobs if len(j["records"]) < heavy_cut]
        heavy_jobs = [j for j in jobs if len(j["records"]) >= heavy_cut]
    else:
        light_jobs = jobs
        heavy_jobs = []
    if args.adaptive_block and heavy_jobs:
        log(
            f"{n_skip} dates already on disk; mosaicking {len(jobs)} remaining "
            f"of {len(dates)}: {len(light_jobs)} light @ {args.workers}w×"
            f"{args.block_size}, {len(heavy_jobs)} heavy (≥{heavy_cut} granules) @ "
            f"{args.heavy_workers}w×{args.heavy_block_size}"
        )
    else:
        log(
            f"{n_skip} dates already on disk; mosaicking {len(jobs)} remaining "
            f"of {len(dates)} with {args.workers} worker(s)"
        )
    pbar = tqdm(
        total=len(dates),
        initial=n_skip,
        desc=f"{year_start}-{year_end}",
        unit="date",
        dynamic_ncols=True,
    )

    def _finish_date(row: dict[str, str]) -> None:
        summary.append(row)
        pbar.update(1)
        left = max(0, int(pbar.total) - int(pbar.n))
        pbar.set_postfix_str(
            f"{row['date']} {row['status']}  left={left}", refresh=True
        )
        pbar.write(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"  [{pbar.n}/{pbar.total}] {row['date']}: {row['status']} {row['note']}"
        )

    try:
        if light_jobs and heavy_jobs:
            log(
                f"Light wave first ({len(light_jobs)} dates), "
                f"then heavy ({len(heavy_jobs)} dates)"
            )
        _run_date_jobs(light_jobs, args.workers, _finish_date)
        _run_date_jobs(heavy_jobs, args.heavy_workers, _finish_date)
    finally:
        pbar.close()

    log_path = season_root / "download_log.csv"
    by_date_status = {r["date"]: r for r in summary}
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "status", "note"])
        w.writeheader()
        for date_str in dates:
            row = by_date_status.get(date_str) or {
                "date": date_str,
                "status": "no_imagery",
                "note": "",
            }
            w.writerow(
                {
                    "date": row["date"],
                    "status": row["status"],
                    "note": row.get("note", ""),
                }
            )
    n_ok = sum(1 for r in summary if r["status"] == "success")
    log(f"Season {year_start}-{year_end}: {n_ok}/{len(dates)} daily TIFFs -> {season_root}")


def main() -> None:
    args = parse_args()
    shp = Path(args.shapefile).expanduser().resolve()
    if not shp.is_file():
        sys.exit(f"Shapefile not found: {shp}")
    geom = read_geometry(shp)
    if geom is None or geom.is_empty:
        sys.exit(f"Empty geometry: {shp}")
    log(
        f"BDC daily TIFF: {shp.stem} seasons={args.season_years_list} "
        f"cloud<{args.cloud_pct} SCL={sorted(args.scl_clear)} "
        f"harmonize={args.harmonize} workers={args.workers}"
    )
    log("Note: SCL replaces GEE Cloud Score+; do not mix with GEE TIFFs without checking.")
    if args.profile_full:
        run_profile_full(args, geom)
        return
    if args.profile:
        run_profile(args, geom)
        return
    for sy in args.season_years_list:
        run_season(args, sy, geom)


if __name__ == "__main__":
    main()
