#!/usr/bin/env python3
"""
Build Paraná municipal daily TIFFs with live progress bars.

Pipeline per season:
  1) Merge GEE tiles in files/raw_tiff/{season}/parana/raw
     -> state mosaics in files/raw_tiff/{season}/parana/state_41_YYYY_MM_DD.tiff
  2) Clip state mosaics to PR municipalities
     -> files/daily_tiff/{season}/{muni}/{muni}_YYYY_MM_DD.tiff

Example (from repo root, in WSL)::

  # Kill any old RAM-heavy merge first, then:
  .venv/bin/python scripts/build_parana_daily_tiff.py --fast \\
    --mosaic-root ~/sits_moco_data/state_tiff \\
    --output-dir ~/sits_moco_data/daily_tiff \\
    --stage-raw

  # Streaming merge is low-RAM → --fast uses merge_workers=6 by default
  .venv/bin/python scripts/build_parana_daily_tiff.py --merge-workers 8 --clip-workers 20
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.clip_tiffs_to_shapefiles import (  # noqa: E402
    _process_one_tiff,
    collect_municipalities,
    parse_date_from_tiff_path,
)
from preprocessing.merge_gee_tiles_to_tiff import (  # noqa: E402
    _merge_one_date,
    collect_jobs,
)

# GDAL/rasterio + fork() deadlocks under Linux; always spawn worker processes.
_MP_CTX = mp.get_context("spawn")

DEFAULT_SEASONS = (
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge Paraná GEE tiles then clip to municipal daily TIFFs (with tqdm)."
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repo root (default: parent of scripts/)",
    )
    p.add_argument(
        "--seasons",
        nargs="+",
        default=list(DEFAULT_SEASONS),
        help="Season folders under files/raw_tiff (default: all 2018-2024)",
    )
    p.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Override files/raw_tiff root (tile inputs)",
    )
    p.add_argument(
        "--mosaic-root",
        type=Path,
        default=None,
        help=(
            "Where to write/read state mosaics ({mosaic_root}/{season}/parana/*.tiff). "
            "Default: same as raw ({raw_root}/{season}/parana). Prefer a Linux path "
            "under ~/… — writing mosaics to /mnt/c is very slow from WSL."
        ),
    )
    p.add_argument(
        "--shapefile-dir",
        type=Path,
        default=None,
        help="Municipal shapefiles (default: files/shapefiles_pr)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "daily_tiff output root (default: files/daily_tiff). "
            "Prefer ~/sits_moco_data/daily_tiff on WSL."
        ),
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Faster defaults: merge>=6, clip>=8, gdal>=4 (clip>8 often OOMs on 62GB WSL)",
    )
    p.add_argument(
        "--merge-workers",
        type=int,
        default=6,
        help="Parallel date merges (default: 6). Streaming copy ~1–2GB RAM/worker.",
    )
    p.add_argument(
        "--clip-workers",
        type=int,
        default=8,
        help="Parallel day clips (default: 8). Each opens a state mosaic; 20 OOMed WSL.",
    )
    p.add_argument(
        "--gdal-threads",
        type=int,
        default=4,
        help="GDAL_NUM_THREADS per merge worker (default: 4)",
    )
    p.add_argument(
        "--stage-raw",
        action="store_true",
        help=(
            "Copy each season's raw/ tiles to --stage-root on the Linux FS before merge "
            "(large one-time copy; much faster reads than /mnt/c)."
        ),
    )
    p.add_argument(
        "--stage-root",
        type=Path,
        default=None,
        help="Staging root for --stage-raw (default: ~/sits_moco_data/raw_tiff_stage)",
    )
    p.add_argument(
        "--skip-existing-merge",
        action="store_true",
        default=True,
        help="Skip dates with existing state mosaic (default: on)",
    )
    p.add_argument(
        "--no-skip-existing-merge",
        action="store_false",
        dest="skip_existing_merge",
    )
    p.add_argument(
        "--skip-existing-clip",
        action="store_true",
        default=True,
        help="Skip non-empty municipal day TIFFs already on disk (default: on)",
    )
    p.add_argument(
        "--no-skip-existing-clip",
        action="store_false",
        dest="skip_existing_clip",
    )
    p.add_argument(
        "--skip-clip",
        action="store_true",
        help="Only run merge (no municipal clip)",
    )
    p.add_argument(
        "--skip-merge",
        action="store_true",
        help="Only run clip (assume state mosaics already exist)",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Log failed days and keep going (default: on)",
    )
    p.add_argument(
        "--fail-fast",
        action="store_false",
        dest="continue_on_error",
        help="Abort on first merge/clip failure",
    )
    p.add_argument(
        "--progress-json",
        type=Path,
        default=None,
        help="Live status JSON (default: files/build_parana_daily_tiff_progress.json)",
    )
    p.add_argument(
        "--failures-log",
        type=Path,
        default=None,
        help="Append failed dates here (default: files/build_parana_daily_tiff_failures.log)",
    )
    return p.parse_args()


def _append_failure(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def run_merge_season(
    *,
    season: str,
    raw_dir: Path,
    state_dir: Path,
    merge_workers: int,
    gdal_threads: int,
    skip_existing: bool,
    continue_on_error: bool,
    failures_log: Path,
    progress_json: Path,
    progress: dict,
) -> tuple[int, int, int]:
    """Returns (ok, failed, skipped)."""
    import time

    state_dir.mkdir(parents=True, exist_ok=True)
    jobs, skipped = collect_jobs(raw_dir, state_dir, skip_existing=skip_existing)
    ok = 0
    failed = 0
    day_seconds: list[float] = []

    progress["phase"] = "merge"
    progress["season"] = season
    progress["merge"] = {
        "total": len(jobs),
        "skipped_existing": skipped,
        "done": 0,
        "failed": 0,
        "current": None,
        "current_started_at": None,
        "avg_sec_per_day": None,
        "eta_sec": None,
    }
    _write_progress(progress_json, progress)

    if not jobs:
        tqdm.write(f"[{season}] merge: nothing to do (skipped_existing={skipped})")
        return 0, 0, skipped

    workers = max(1, merge_workers)
    pbar = tqdm(
        total=len(jobs),
        desc=f"{season} merge",
        unit="day",
        dynamic_ncols=True,
        leave=True,
    )

    def _on_done(date_str: str, success: bool, detail: str, elapsed: float | None) -> None:
        nonlocal ok, failed
        if success:
            ok += 1
            if elapsed is not None and elapsed > 0:
                day_seconds.append(elapsed)
            avg = sum(day_seconds) / len(day_seconds) if day_seconds else None
            remaining = len(jobs) - (ok + failed)
            eta = (avg * remaining) if avg is not None else None
            eta_str = f" ETA~{eta/60:.0f}m" if eta is not None else ""
            avg_str = f" avg={avg:.0f}s/day" if avg is not None else ""
            pbar.set_postfix_str(f"{date_str}{avg_str}{eta_str}")
            tqdm.write(
                f"[{season}] MERGE OK {date_str}"
                + (f" ({elapsed:.0f}s)" if elapsed is not None else "")
            )
            if avg is not None:
                tqdm.write(
                    f"[{season}] pace: {avg:.0f}s/day → ~{eta/60:.0f} min left "
                    f"({remaining} days)"
                )
            progress["merge"]["avg_sec_per_day"] = avg
            progress["merge"]["eta_sec"] = eta
        else:
            failed += 1
            tqdm.write(f"[{season}] MERGE FAIL {date_str}: {detail}")
            _append_failure(
                failures_log,
                f"{_utc_now()}\t{season}\tmerge\t{date_str}\t{detail}",
            )
            if not continue_on_error:
                raise RuntimeError(detail)
        progress["merge"]["done"] = ok + failed
        progress["merge"]["failed"] = failed
        progress["merge"]["current"] = date_str
        progress["merge"]["current_started_at"] = None
        progress["updated_at"] = _utc_now()
        _write_progress(progress_json, progress)
        pbar.update(1)

    try:
        if workers == 1:
            for date_str, tile_paths, out_path in jobs:
                tqdm.write(
                    f"[{season}] MERGE start {date_str} "
                    f"({len(tile_paths)} tiles) → {Path(out_path).name}"
                )
                progress["merge"]["current"] = date_str
                progress["merge"]["current_started_at"] = _utc_now()
                progress["updated_at"] = _utc_now()
                _write_progress(progress_json, progress)
                t0 = time.perf_counter()
                try:
                    msg = _merge_one_date(
                        date_str, tile_paths, out_path, gdal_threads, verbose=True
                    )
                    _on_done(date_str, True, msg, time.perf_counter() - t0)
                except Exception as e:
                    _on_done(
                        date_str,
                        False,
                        f"{type(e).__name__}: {e}",
                        time.perf_counter() - t0,
                    )
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=_MP_CTX) as ex:
                futures = {
                    ex.submit(
                        _merge_one_date,
                        date_str,
                        tile_paths,
                        out_path,
                        gdal_threads,
                        True,
                    ): (date_str, time.perf_counter())
                    for date_str, tile_paths, out_path in jobs
                }
                for fut in as_completed(futures):
                    date_str, t0 = futures[fut]
                    try:
                        msg = fut.result()
                        _on_done(date_str, True, msg, time.perf_counter() - t0)
                    except Exception as e:
                        _on_done(
                            date_str,
                            False,
                            f"{type(e).__name__}: {e}",
                            time.perf_counter() - t0,
                        )
    finally:
        pbar.close()

    return ok, failed, skipped


def run_clip_season(
    *,
    season: str,
    state_dir: Path,
    shapefile_dir: Path,
    output_dir: Path,
    clip_workers: int,
    skip_existing: bool,
    continue_on_error: bool,
    failures_log: Path,
    progress_json: Path,
    progress: dict,
) -> tuple[int, int]:
    """Returns (ok, failed)."""
    tiffs = sorted(
        [
            p
            for p in state_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".tif", ".tiff"}
            and ".tmp." not in p.name
            and parse_date_from_tiff_path(p)
        ]
    )
    if not tiffs:
        tqdm.write(f"[{season}] clip: no state mosaics in {state_dir}")
        return 0, 0

    municipalities = collect_municipalities(shapefile_dir=shapefile_dir)
    if not municipalities:
        raise SystemExit(f"No shapefiles in {shapefile_dir}")

    ok = 0
    failed = 0
    workers = max(1, clip_workers)

    progress["phase"] = "clip"
    progress["season"] = season
    progress["clip"] = {
        "total": len(tiffs),
        "done": 0,
        "failed": 0,
        "municipalities": len(municipalities),
        "skip_existing": skip_existing,
        "current": None,
    }
    _write_progress(progress_json, progress)

    pbar = tqdm(
        total=len(tiffs),
        desc=f"{season} clip",
        unit="day",
        dynamic_ncols=True,
        leave=True,
    )

    def _on_done(name: str, success: bool, detail: str) -> None:
        nonlocal ok, failed
        if success:
            ok += 1
            pbar.set_postfix_str(name)
            tqdm.write(f"[{season}] CLIP OK {name} ({detail})")
        else:
            failed += 1
            tqdm.write(f"[{season}] CLIP FAIL {name}: {detail}")
            _append_failure(
                failures_log,
                f"{_utc_now()}\t{season}\tclip\t{name}\t{detail}",
            )
            if not continue_on_error:
                raise RuntimeError(detail)
        progress["clip"]["done"] = ok + failed
        progress["clip"]["failed"] = failed
        progress["clip"]["current"] = name
        progress["updated_at"] = _utc_now()
        _write_progress(progress_json, progress)
        pbar.update(1)

    try:
        if workers == 1:
            for tiff_path in tiffs:
                try:
                    name, written = _process_one_tiff(
                        tiff_path,
                        municipalities,
                        output_dir,
                        season,
                        True,
                        skip_existing,
                        50 if workers == 1 else 0,
                    )
                    _on_done(name, True, f"wrote {written}")
                except Exception as e:
                    _on_done(tiff_path.name, False, f"{type(e).__name__}: {e}")
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=_MP_CTX) as ex:
                futures = {
                    ex.submit(
                        _process_one_tiff,
                        tiff_path,
                        municipalities,
                        output_dir,
                        season,
                        True,
                        skip_existing,
                        0,
                    ): tiff_path.name
                    for tiff_path in tiffs
                }
                for fut in as_completed(futures):
                    name = futures[fut]
                    try:
                        out_name, written = fut.result()
                        _on_done(out_name, True, f"wrote {written}")
                    except Exception as e:
                        _on_done(name, False, f"{type(e).__name__}: {e}")
    finally:
        pbar.close()

    return ok, failed


def _stage_raw_season(
    src_raw: Path,
    dest_raw: Path,
) -> Path:
    """Copy raw tiles to Linux FS (rsync if available, else shutil). Returns dest_raw."""
    import shutil
    import subprocess

    dest_raw.parent.mkdir(parents=True, exist_ok=True)
    if dest_raw.is_dir() and any(dest_raw.iterdir()):
        tqdm.write(f"stage: using existing {dest_raw}")
        return dest_raw

    tqdm.write(f"stage: copying {src_raw} → {dest_raw} (one-time, may take a while)")
    dest_raw.mkdir(parents=True, exist_ok=True)
    rsync = shutil.which("rsync")
    if rsync:
        subprocess.run(
            [rsync, "-a", "--info=progress2", f"{src_raw}/", f"{dest_raw}/"],
            check=True,
        )
    else:
        for p in src_raw.iterdir():
            if p.is_file():
                shutil.copy2(p, dest_raw / p.name)
    return dest_raw


def main() -> None:
    args = parse_args()
    if args.fast:
        args.merge_workers = max(args.merge_workers, 6)
        args.clip_workers = max(args.clip_workers, 8)
        args.gdal_threads = max(args.gdal_threads, 4)
    if args.clip_workers > 12:
        print(
            f"WARNING: clip_workers={args.clip_workers} — each worker opens a full state "
            f"mosaic. On ~62GB WSL, 20 workers previously OOM-killed the VM. Prefer 6–8.",
            flush=True,
        )
    root = args.repo_root.resolve()
    raw_root = (args.raw_root or root / "files" / "raw_tiff").expanduser().resolve()
    mosaic_root = (
        args.mosaic_root.expanduser().resolve()
        if args.mosaic_root
        else raw_root
    )
    shapefile_dir = (args.shapefile_dir or root / "files" / "shapefiles_pr").expanduser().resolve()
    output_dir = (args.output_dir or root / "files" / "daily_tiff").expanduser().resolve()
    progress_json = (
        args.progress_json or root / "files" / "build_parana_daily_tiff_progress.json"
    ).resolve()
    failures_log = (
        args.failures_log or root / "files" / "build_parana_daily_tiff_failures.log"
    ).resolve()

    if not shapefile_dir.is_dir():
        raise SystemExit(f"Shapefile dir missing: {shapefile_dir}")

    def _is_mnt_c(p: Path) -> bool:
        s = str(p).replace("\\", "/").lower()
        return s.startswith("/mnt/c/") or s.startswith("c:/") or s.startswith("c:\\")

    if any(_is_mnt_c(p) for p in (mosaic_root, output_dir)):
        print(
            "WARNING: mosaic/output is under /mnt/c (Windows NTFS via WSL). "
            "This keeps CPU idle while waiting on 9p I/O. Prefer e.g.\n"
            "  --mosaic-root ~/sits_moco_data/state_tiff "
            "--output-dir ~/sits_moco_data/daily_tiff",
            flush=True,
        )

    seasons = list(args.seasons)
    progress: dict = {
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "repo_root": str(root),
        "seasons": seasons,
        "merge_workers": args.merge_workers,
        "clip_workers": args.clip_workers,
        "gdal_threads": args.gdal_threads,
        "mosaic_root": str(mosaic_root),
        "output_dir": str(output_dir),
        "phase": "starting",
        "season": None,
        "season_index": 0,
        "season_total": len(seasons),
        "summary": {},
    }
    _write_progress(progress_json, progress)

    print(f"repo_root     = {root}")
    print(f"raw_root      = {raw_root}")
    print(f"mosaic_root   = {mosaic_root}")
    print(f"shapefile_dir = {shapefile_dir}")
    print(f"output_dir    = {output_dir}")
    print(f"progress_json = {progress_json}")
    print(f"failures_log  = {failures_log}")
    print(
        f"merge_workers={args.merge_workers} clip_workers={args.clip_workers} "
        f"gdal_threads={args.gdal_threads} continue_on_error={args.continue_on_error} "
        f"stage_raw={args.stage_raw}"
    )
    print(f"seasons={seasons}")
    print()

    stage_root = None
    if args.stage_raw:
        stage_root = (
            args.stage_root
            or Path.home() / "sits_moco_data" / "raw_tiff_stage"
        ).expanduser().resolve()
        print(f"stage_root    = {stage_root}")

    season_bar = tqdm(
        seasons,
        desc="seasons",
        unit="season",
        dynamic_ncols=True,
        leave=True,
    )
    try:
        for idx, season in enumerate(season_bar, start=1):
            season_bar.set_postfix_str(season)
            raw_dir = raw_root / season / "parana" / "raw"
            state_dir = mosaic_root / season / "parana"
            progress["season_index"] = idx
            progress["season"] = season
            progress["updated_at"] = _utc_now()
            _write_progress(progress_json, progress)

            if not raw_dir.is_dir() and not args.skip_merge:
                tqdm.write(f"[{season}] SKIP missing {raw_dir}")
                progress["summary"][season] = {"status": "skipped_missing_raw"}
                continue

            season_summary: dict = {"status": "ok"}

            if not args.skip_merge:
                if not raw_dir.is_dir():
                    raise SystemExit(f"Missing raw dir: {raw_dir}")
                merge_raw = raw_dir
                if stage_root is not None:
                    merge_raw = _stage_raw_season(
                        raw_dir, stage_root / season / "parana" / "raw"
                    )
                ok, failed, skipped = run_merge_season(
                    season=season,
                    raw_dir=merge_raw,
                    state_dir=state_dir,
                    merge_workers=args.merge_workers,
                    gdal_threads=args.gdal_threads,
                    skip_existing=args.skip_existing_merge,
                    continue_on_error=args.continue_on_error,
                    failures_log=failures_log,
                    progress_json=progress_json,
                    progress=progress,
                )
                season_summary["merge"] = {
                    "ok": ok,
                    "failed": failed,
                    "skipped_existing": skipped,
                }

            if not args.skip_clip:
                if not state_dir.is_dir():
                    raise SystemExit(f"Missing state dir: {state_dir}")
                ok, failed = run_clip_season(
                    season=season,
                    state_dir=state_dir,
                    shapefile_dir=shapefile_dir,
                    output_dir=output_dir,
                    clip_workers=args.clip_workers,
                    skip_existing=args.skip_existing_clip,
                    continue_on_error=args.continue_on_error,
                    failures_log=failures_log,
                    progress_json=progress_json,
                    progress=progress,
                )
                season_summary["clip"] = {"ok": ok, "failed": failed}

            progress["summary"][season] = season_summary
            progress["updated_at"] = _utc_now()
            _write_progress(progress_json, progress)
    except Exception:
        progress["phase"] = "failed"
        progress["error"] = traceback.format_exc()
        progress["updated_at"] = _utc_now()
        _write_progress(progress_json, progress)
        raise
    finally:
        season_bar.close()

    progress["phase"] = "done"
    progress["finished_at"] = _utc_now()
    progress["updated_at"] = _utc_now()
    _write_progress(progress_json, progress)
    print("\nDone.")
    print(f"Progress: {progress_json}")
    print(f"Failures: {failures_log}")
    print(f"Output:   {output_dir}")


if __name__ == "__main__":
    main()
