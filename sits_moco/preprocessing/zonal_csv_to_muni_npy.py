#!/usr/bin/env python3
"""
Convert GEE zonal-mean CSVs to municipal mega-pixel .npy files [1, T, C].

Spectral channels stay in Sentinel-2 DN (same as daily TIFF .npy; training
applies *1e-4). Channel 10 is season DOY (Oct 1 = 1), not calendar DOY.

When a donor .npy exists for the same muni/season with extra channels, those
are copied by matching season DOY (rain/climate). New municipalities without a
donor are written as 11-channel (spectral + DOY) files.

Example:
  python preprocessing/zonal_csv_to_muni_npy.py \\
      --zonal-root files/zonal_mean \\
      --keep-list files/zonal_mean_keep_0.8_1.2.csv \\
      --output-dir ~/sits_moco_data/npy_muni_mean \\
      --skip-existing
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from datasets.pixel_transform import DOY_CHANNEL, NO_DATA_VALUE, NUM_SPECTRAL_CHANNELS
from preprocessing.preprocess_daily_to_npy import _save_npy_atomic
from preprocessing.preprocess_tiff_to_npy import (
    date_to_season_doy,
    season_start_from_year_range,
)

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--zonal-root",
        type=Path,
        default=_REPO / "files" / "zonal_mean",
        help="Root with {YYYY-YYYY}/{code}/{code}_zonal.csv",
    )
    p.add_argument(
        "--keep-list",
        type=Path,
        default=None,
        help="CSV with municipality_code,year (harvest year). If omitted, convert every zonal CSV.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/sits_moco_data/npy_muni_mean"),
        help="Write {output}/{YYYY-YYYY}/{code}/{code}.npy",
    )
    p.add_argument(
        "--extras-from",
        type=Path,
        default=None,
        help="Donor npy root for rain/climate extras (default: same as --output-dir)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not overwrite an existing destination .npy",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files (spectral from zonal; extras kept if donor exists)",
    )
    return p.parse_args()


def harvest_to_season(harvest_year: int) -> str:
    return f"{harvest_year - 1}-{harvest_year}"


def _code7(value: object) -> str:
    return f"{int(float(value)):07d}"


def load_keep_list(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            year_raw = r.get("year") or r.get("season_year")
            if year_raw is None or str(year_raw).strip() == "":
                continue
            rows.append((_code7(r["municipality_code"]), int(float(year_raw))))
    return rows


def discover_zonal_jobs(root: Path) -> list[tuple[str, int, Path]]:
    jobs: list[tuple[str, int, Path]] = []
    for csv_path in sorted(root.glob("*/*/*_zonal.csv")):
        season = csv_path.parent.parent.name
        if "-" not in season:
            continue
        y1, y2 = season.split("-", 1)
        if not (y1.isdigit() and y2.isdigit()):
            continue
        jobs.append((_code7(csv_path.parent.name), int(y2), csv_path))
    return jobs


def parse_date(value: object) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text[:10], fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "")).date()
    except ValueError:
        return None


def _is_bad_band(value: float) -> bool:
    return (not np.isfinite(value)) or value == NO_DATA_VALUE


def zonal_csv_to_array(path: Path, year_range: str) -> np.ndarray | None:
    """Return [1, T, 11] or None if no usable dates."""
    df = pd.read_csv(path)
    if df.empty:
        return None
    missing = [b for b in BANDS if b not in df.columns]
    if missing or "date" not in df.columns:
        return None

    season_start = season_start_from_year_range(year_range)
    rows: list[tuple[date, list[float], float]] = []
    for rec in df.to_dict("records"):
        d = parse_date(rec.get("date"))
        if d is None:
            continue
        try:
            vpc = rec.get("valid_pixel_count")
            if vpc is not None and str(vpc).strip() != "" and int(float(vpc)) <= 0:
                continue
        except (TypeError, ValueError):
            pass
        bands: list[float] = []
        bad = False
        for name in BANDS:
            try:
                v = float(rec[name])
            except (TypeError, ValueError, KeyError):
                bad = True
                break
            if _is_bad_band(v):
                bad = True
                break
            bands.append(v)
        if bad:
            continue
        rows.append((d, bands, float(date_to_season_doy(d, season_start))))

    if not rows:
        return None
    rows.sort(key=lambda x: x[0])
    t = len(rows)
    out = np.full((1, t, NUM_SPECTRAL_CHANNELS + 1), NO_DATA_VALUE, dtype=np.float32)
    for i, (_d, bands, doy) in enumerate(rows):
        out[0, i, :NUM_SPECTRAL_CHANNELS] = np.asarray(bands, dtype=np.float32)
        out[0, i, DOY_CHANNEL] = np.float32(doy)
    return out


def attach_extras_by_doy(base: np.ndarray, donor: np.ndarray) -> np.ndarray:
    """Copy donor channels 11+ onto ``base`` by matching season DOY."""
    if donor.ndim != 3 or donor.shape[2] <= NUM_SPECTRAL_CHANNELS + 1:
        return base
    extra_c = int(donor.shape[2] - (DOY_CHANNEL + 1))
    if extra_c <= 0:
        return base
    doy_to_i: dict[int, int] = {}
    for i, raw in enumerate(donor[0, :, DOY_CHANNEL]):
        v = float(raw)
        if np.isfinite(v) and v != NO_DATA_VALUE:
            doy_to_i.setdefault(int(round(v)), i)
    if not doy_to_i:
        return base
    out = np.full((1, base.shape[1], DOY_CHANNEL + 1 + extra_c), NO_DATA_VALUE, dtype=np.float32)
    out[:, :, : DOY_CHANNEL + 1] = base
    matched = 0
    for t in range(base.shape[1]):
        v = float(base[0, t, DOY_CHANNEL])
        if not np.isfinite(v):
            continue
        i = doy_to_i.get(int(round(v)))
        if i is None:
            continue
        out[0, t, DOY_CHANNEL + 1 :] = donor[0, i, DOY_CHANNEL + 1 :]
        matched += 1
    if matched == 0:
        return base
    return out


def npy_path(root: Path, season: str, code: str) -> Path:
    return root / season / code / f"{code}.npy"


def convert_one(
    zonal_csv: Path,
    dest: Path,
    extras_path: Path | None,
    skip_existing: bool,
) -> str:
    if skip_existing and dest.is_file():
        return "skipped"
    season = dest.parent.parent.name
    arr = zonal_csv_to_array(zonal_csv, season)
    if arr is None:
        return "empty"
    if extras_path is not None and extras_path.is_file():
        try:
            donor = np.load(extras_path)
            arr = attach_extras_by_doy(arr, donor)
        except OSError:
            pass
    _save_npy_atomic(dest, arr.astype(np.float32, copy=False))
    return f"C={arr.shape[2]},T={arr.shape[1]}"


def main() -> None:
    args = parse_args()
    zonal_root = args.zonal_root.expanduser().resolve()
    out_root = args.output_dir.expanduser().resolve()
    extras_root = (
        args.extras_from.expanduser().resolve()
        if args.extras_from is not None
        else out_root
    )
    skip_existing = bool(args.skip_existing) and not bool(args.overwrite)

    if args.keep_list is not None:
        wanted = load_keep_list(args.keep_list.resolve())
        jobs: list[tuple[str, str, Path]] = []
        for code, harvest in wanted:
            season = harvest_to_season(harvest)
            csv_path = zonal_root / season / code / f"{code}_zonal.csv"
            if csv_path.is_file():
                jobs.append((code, season, csv_path))
    else:
        jobs = [
            (code, harvest_to_season(harvest), path)
            for code, harvest, path in discover_zonal_jobs(zonal_root)
        ]

    if not jobs:
        raise SystemExit(f"No zonal CSVs to convert under {zonal_root}")

    counts: Counter[str] = Counter()
    for code, season, csv_path in tqdm(jobs, desc="zonal→npy", unit="muni"):
        dest = npy_path(out_root, season, code)
        extras = npy_path(extras_root, season, code)
        try:
            status = convert_one(csv_path, dest, extras, skip_existing)
        except Exception as exc:
            counts["error"] += 1
            print(f"[ERROR] {csv_path}: {exc}")
            continue
        counts[status.split(",")[0]] += 1

    print(f"Jobs {len(jobs)}: {dict(counts)}")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()
