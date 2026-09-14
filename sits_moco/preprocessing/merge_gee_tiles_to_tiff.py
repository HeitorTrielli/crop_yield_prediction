#!/usr/bin/env python3
"""
Merge GEE tile .tif files (e.g. from .../raw/) into a single .tiff mosaic per date.

Use when you have raw tiles from data_download/download_soy_gee_drive.py under .../raw/:
- .tif tiles: state_41_2022-10-02-0000000000-0000009984.tif, etc.

This script groups by date, merges with rasterio, and writes one .tiff per date.
Raw .tif files are never deleted (kept as archive).

Usage:
  python preprocessing/merge_gee_tiles_to_tiff.py --dir files/raw_tiff/2022-2023/parana/raw
  python preprocessing/merge_gee_tiles_to_tiff.py --dir .../raw -j 2 --skip-existing
"""

from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import multiprocessing as mp

_MP_CTX = mp.get_context("spawn")

# Must match GEE export and downstream (clip, training). Merged output always uses this.
NO_DATA_VALUE = -9999


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge .tif tiles and existing .tiff into one mosaic per date."
    )
    p.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Directory containing .tif tile files (e.g. .../2022-2023/raw)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for merged .tiff files (default: parent of --dir, so merged go next to raw/)",
    )
    p.add_argument(
        "-j",
        "--workers",
        type=int,
        default=1,
        help=(
            "Parallel date merges (default: 1). Streaming merge uses ~1–2GB/worker; "
            "4–8 is usually fine on 64GB WSL."
        ),
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip dates whose output .tiff already exists (default: on)",
    )
    p.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Re-merge even if output .tiff already exists",
    )
    p.add_argument(
        "--gdal-threads",
        type=int,
        default=2,
        help="GDAL_NUM_THREADS per merge worker for compression (default: 2)",
    )
    return p.parse_args()


def date_from_tiff_stem(stem: str) -> str | None:
    """From state_41_2022_10_02 return 2022-10-02."""
    parts = stem.split("_")
    if (
        len(parts) >= 4
        and parts[-3].isdigit()
        and parts[-2].isdigit()
        and parts[-1].isdigit()
    ):
        y, m, d = parts[-3], parts[-2], parts[-1]
        if len(y) == 4 and len(m) == 2 and len(d) == 2:
            return f"{y}-{m}-{d}"
    return None


def date_from_tif_stem(stem: str) -> str | None:
    """From state_41_2022-10-02-0000000000-0000009984 return 2022-10-02."""
    parts = stem.rsplit("-", 2)
    if (
        len(parts) == 3
        and len(parts[1]) == 10
        and parts[1].isdigit()
        and len(parts[2]) == 10
        and parts[2].isdigit()
    ):
        prefix = parts[0]  # state_41_2022-10-02
        if len(prefix) >= 11 and prefix[-11] == "_":
            return prefix[-10:]  # YYYY-MM-DD
        match = re.match(r".*_(\d{4}-\d{2}-\d{2})$", prefix)
        if match:
            return match.group(1)
    return None


def _merge_heartbeat(date_str: str, phase: str, detail: str = "") -> None:
    """Print a flush line so long merges don't look stuck (esp. reading /mnt/c)."""
    ts = datetime.now().strftime("%H:%M:%S")
    extra = f" | {detail}" if detail else ""
    print(f"  [{ts}] {date_str}  {phase}{extra}", flush=True)


def _mosaic_layout(
    tile_paths_p: list[Path],
) -> tuple[dict, list[tuple[Path, int, int, int, int]]]:
    """
    Compute mosaic profile + per-tile destination window offsets.

    GEE tiles share CRS/resolution and abut on a grid; we copy each tile into
    the mosaic window instead of rasterio.merge (which materializes ~12GB RAM).
    Returns (out_meta_base, [(path, col_off, row_off, width, height), ...]).
    """
    import rasterio
    from rasterio.transform import from_bounds

    infos: list[tuple[Path, object, object]] = []
    for p in sorted(tile_paths_p):
        with rasterio.open(p) as src:
            infos.append((p, src.profile.copy(), src.bounds))

    if not infos:
        raise ValueError("No tiles to merge")

    first_profile = infos[0][1]
    res_x = abs(first_profile["transform"].a)
    res_y = abs(first_profile["transform"].e)
    left = min(b.left for _, _, b in infos)
    bottom = min(b.bottom for _, _, b in infos)
    right = max(b.right for _, _, b in infos)
    top = max(b.top for _, _, b in infos)
    width = int(round((right - left) / res_x))
    height = int(round((top - bottom) / res_y))
    transform = from_bounds(left, bottom, right, top, width, height)

    placements: list[tuple[Path, int, int, int, int]] = []
    for p, profile, bounds in infos:
        col_off = int(round((bounds.left - left) / res_x))
        row_off = int(round((top - bounds.top) / res_y))
        tw = int(profile["width"])
        th = int(profile["height"])
        placements.append((p, col_off, row_off, tw, th))

    meta = {
        "driver": "GTiff",
        "dtype": first_profile["dtype"],
        "count": first_profile["count"],
        "crs": first_profile["crs"],
        "transform": transform,
        "width": width,
        "height": height,
        "nodata": first_profile.get("nodata", NO_DATA_VALUE),
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "YES",
        "NUM_THREADS": "ALL_CPUS",
    }
    return meta, placements


def _merge_one_date(
    date_str: str,
    tile_paths: list[str],
    out_path: str,
    gdal_threads: int,
    verbose: bool = True,
) -> str:
    """
    Merge one date's tiles into out_path with low RAM (copy tile→mosaic by band).

    Peak RAM is roughly one band of one tile (~0.4GB), not the full mosaic (~12GB),
    so several merge workers can run safely on a 64GB WSL VM.
    """
    import time

    import rasterio
    from rasterio.windows import Window

    if gdal_threads > 0:
        os.environ["GDAL_NUM_THREADS"] = str(gdal_threads)

    out_path_p = Path(out_path)
    tile_paths_p = [Path(p) for p in tile_paths]
    preferred_compress = (
        {"compress": "zstd", "zstd_level": 3},
        {"compress": "deflate", "predictor": 2, "zlevel": 3},
        {"compress": "lzw"},
    )
    t0 = time.perf_counter()

    def beat(phase: str, detail: str = "") -> None:
        if verbose:
            elapsed = time.perf_counter() - t0
            suffix = f"{detail} (+{elapsed:.0f}s)".strip() if detail else f"(+{elapsed:.0f}s)"
            _merge_heartbeat(date_str, phase, suffix)

    beat("layout", f"{len(tile_paths_p)} tiles")
    meta, placements = _mosaic_layout(tile_paths_p)
    beat(
        "layout_done",
        f"mosaic {meta['count']}x{meta['height']}x{meta['width']}",
    )

    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path_p.with_name(out_path_p.stem + f".{os.getpid()}.tmp.tiff")
    last_err: Exception | None = None

    try:
        for compress_opts in preferred_compress:
            write_meta = dict(meta)
            write_meta.update(compress_opts)
            codec = compress_opts.get("compress", "?")
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            beat("write_start", f"compress={codec}")
            try:
                with rasterio.open(tmp, "w", **write_meta) as dst:
                    n = len(placements)
                    for i, (p, col_off, row_off, tw, th) in enumerate(placements, start=1):
                        beat("copy_tile", f"{i}/{n} {p.name}")
                        with rasterio.open(p) as src:
                            dst_win = Window(col_off, row_off, tw, th)
                            # Band-by-band keeps peak RAM ~one float32 panel (~0.4GB).
                            for bidx in range(1, src.count + 1):
                                band = src.read(bidx)
                                dst.write(band, bidx, window=dst_win)
                last_err = None
                break
            except Exception as e:
                last_err = e
                beat("write_retry", f"{codec} failed: {type(e).__name__}: {e}")
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
        if last_err is not None:
            raise last_err
        size_mb = tmp.stat().st_size / (1024**2) if tmp.exists() else 0.0
        tmp.replace(out_path_p)
        elapsed = time.perf_counter() - t0
        beat("done", f"{out_path_p.name} ({size_mb:.0f} MiB) in {elapsed:.0f}s")
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

    return (
        f"{date_str}: merged {len(tile_paths_p)} .tif tile(s) -> {out_path_p.name} "
        f"({time.perf_counter() - t0:.0f}s)"
    )


def collect_jobs(
    folder: Path,
    out_dir: Path,
    *,
    skip_existing: bool,
) -> tuple[list[tuple[str, list[str], str]], int]:
    """Return (jobs, skipped_count). Each job is (date_str, tile_path_strs, out_path_str)."""
    date_to_paths: dict[str, list[Path]] = {}
    date_to_code: dict[str, str] = {}

    for path in folder.iterdir():
        if not path.is_file():
            continue
        stem = path.stem
        if path.suffix.lower() == ".tiff":
            d = date_from_tiff_stem(stem)
            if d:
                date_to_paths.setdefault(d, []).append(path)
                if d not in date_to_code:
                    date_to_code[d] = stem[:-11] if len(stem) > 11 else stem
        elif path.suffix.lower() == ".tif":
            d = date_from_tif_stem(stem)
            if d:
                date_to_paths.setdefault(d, []).append(path)
                if d not in date_to_code:
                    parts = stem.rsplit("-", 2)
                    prefix = parts[0] if len(parts) == 3 else stem
                    date_str = (
                        prefix[-10:] if len(prefix) >= 10 and prefix[-11] == "_" else ""
                    )
                    if re.match(r"\d{4}-\d{2}-\d{2}", date_str):
                        date_to_code[d] = prefix[: -len(date_str) - 1]

    jobs: list[tuple[str, list[str], str]] = []
    skipped = 0
    for date_str in sorted(date_to_paths.keys()):
        paths = date_to_paths[date_str]
        tile_paths = [p for p in paths if p.suffix.lower() == ".tif"]
        if not tile_paths:
            continue
        code = date_to_code.get(date_str, "state_41")
        y, m, d = date_str.split("-")
        out_name = f"{code}_{y}_{m}_{d}.tiff"
        out_path = out_dir / out_name
        if skip_existing and out_path.is_file() and out_path.stat().st_size > 0:
            skipped += 1
            continue
        jobs.append(
            (date_str, [str(p.resolve()) for p in tile_paths], str(out_path.resolve()))
        )
    return jobs, skipped


def main() -> None:
    args = parse_args()
    folder = Path(args.dir).resolve()
    if not folder.is_dir():
        raise SystemExit(f"Not a directory: {folder}")
    out_dir = Path(args.output_dir).resolve() if args.output_dir else folder.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    workers = max(1, int(args.workers))
    jobs, skipped = collect_jobs(folder, out_dir, skip_existing=args.skip_existing)
    if not jobs and skipped == 0:
        _log("No .tiff or .tif files with recognizable date pattern found.")
        return

    _log(
        f"dates_to_merge={len(jobs)} skipped_existing={skipped} "
        f"workers={workers} gdal_threads={args.gdal_threads} out={out_dir}"
    )
    if not jobs:
        _log("Nothing to do.")
        return

    if workers == 1:
        for date_str, tile_paths, out_path in jobs:
            msg = _merge_one_date(
                date_str, tile_paths, out_path, args.gdal_threads
            )
            _log(msg)
        return

    # Cap concurrent merges: each loads a full state mosaic into RAM.
    with ProcessPoolExecutor(max_workers=workers, mp_context=_MP_CTX) as ex:
        futures = {
            ex.submit(
                _merge_one_date,
                date_str,
                tile_paths,
                out_path,
                args.gdal_threads,
            ): date_str
            for date_str, tile_paths, out_path in jobs
        }
        for fut in as_completed(futures):
            date_str = futures[fut]
            try:
                _log(fut.result())
            except Exception as e:
                _log(f"{date_str}: FAILED {e!r}")
                raise


if __name__ == "__main__":
    main()
