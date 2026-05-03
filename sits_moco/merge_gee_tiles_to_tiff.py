#!/usr/bin/env python3
"""
Merge GEE tile .tif files (e.g. from .../raw/) into a single .tiff mosaic per date.

Use when you have raw tiles from download_soy_gee_drive.py under .../raw/:
- .tif tiles: state_41_2022-10-02-0000000000-0000009984.tif, etc.

This script groups by date, merges with rasterio, and writes one .tiff per date.
Raw .tif files are never deleted (kept as archive).

Usage:
  python merge_gee_tiles_to_tiff.py --dir files/raw_tiff/2022-2023/parana/raw
  (merged .tiff files go to parent dir by default, e.g. .../parana/state_41_YYYY_MM_DD.tiff)
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

try:
    import rasterio
    from rasterio.merge import merge as rasterio_merge
except ImportError:
    raise SystemExit("Install rasterio: pip install rasterio")

# Must match GEE export and downstream (clip, training). Merged output always uses this.
NO_DATA_VALUE = -9999


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
    # Strip GEE tile suffix -NNNNNNNNNN-NNNNNNNNNN
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


def main() -> None:
    args = parse_args()
    folder = Path(args.dir).resolve()
    if not folder.is_dir():
        raise SystemExit(f"Not a directory: {folder}")
    out_dir = Path(args.output_dir).resolve() if args.output_dir else folder.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect .tiff (day) and .tif (tile) files and group by date
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
                # code is stem minus _YYYY_MM_DD
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

    if not date_to_paths:
        _log("No .tiff or .tif files with recognizable date pattern found.")
        return

    for date_str in sorted(date_to_paths.keys()):
        paths = date_to_paths[date_str]
        tile_paths = [p for p in paths if p.suffix.lower() == ".tif"]
        code = date_to_code.get(date_str, "state_41")
        y, m, d = date_str.split("-")
        out_name = f"{code}_{y}_{m}_{d}.tiff"
        out_path = out_dir / out_name

        if not tile_paths:
            if len(paths) == 1 and paths[0].suffix.lower() == ".tiff":
                _log(f"{date_str}: single .tiff only, nothing to merge")
            continue

        datasets = []
        try:
            for p in sorted(tile_paths):
                datasets.append(rasterio.open(p))
            first_meta = datasets[0].meta
            merged_data, merged_transform = rasterio_merge(
                datasets, nodata=NO_DATA_VALUE
            )
            meta = first_meta.copy()
            meta.update(
                height=merged_data.shape[1],
                width=merged_data.shape[2],
                transform=merged_transform,
                compress="LZW",
                nodata=NO_DATA_VALUE,
            )
        finally:
            for d in datasets:
                d.close()

        # Write to temp then replace, so we don't overwrite a file we (or OneDrive/IDE) may still have open on Windows
        tmp = out_dir / (out_path.stem + "_tmp.tiff")
        try:
            with rasterio.open(tmp, "w", **meta) as dst:
                dst.write(merged_data)
            tmp.replace(out_path)
            _log(f"{date_str}: merged {len(tile_paths)} .tif tile(s) -> {out_name}")
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass


if __name__ == "__main__":
    main()
