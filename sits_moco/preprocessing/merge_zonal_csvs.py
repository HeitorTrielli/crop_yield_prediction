#!/usr/bin/env python3
"""
Merge municipal zonal-mean CSVs from two roots (e.g. BDC + GEE).

For each municipality/season present in either root:
  - union rows by ``date``
  - if both have the same date, keep the row with higher ``valid_pixel_count``
  - on a tie, prefer the GEE (``--gee-root``) row

When overwriting an existing BDC-built ``{code}_zonal.csv`` under ``--bdc-root``,
back it up to ``{code}_zonal.bdc.csv`` once (does not overwrite an existing backup).

Example:
  python preprocessing/merge_zonal_csvs.py \\
      --bdc-root files/zonal_mean \\
      --gee-root files/zonal_mean_gee \\
      --seasons 2019-2020,2020-2021,2023-2024 \\
      --muni-prefix 43
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from preprocessing.state_tiff_to_zonal_csv import ZONAL_CSV_FIELDS  # noqa: E402

PREFERRED_FIELDS = list(ZONAL_CSV_FIELDS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bdc-root",
        type=Path,
        required=True,
        help="Primary / destination zonal root (usually files/zonal_mean)",
    )
    p.add_argument(
        "--gee-root",
        type=Path,
        required=True,
        help="GEE zonal root to merge in (e.g. files/zonal_mean_gee)",
    )
    p.add_argument(
        "--seasons",
        type=str,
        default=None,
        help="Comma-separated YYYY-YYYY seasons (default: intersection of both roots)",
    )
    p.add_argument(
        "--muni-prefix",
        type=str,
        default="43",
        help="Only merge municipality folders starting with this prefix (default: 43)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report merges without writing",
    )
    return p.parse_args()


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _valid_count(row: dict[str, str]) -> float:
    raw = row.get("valid_pixel_count", "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float("-inf")


def _normalize_date(row: dict[str, str]) -> str | None:
    d = (row.get("date") or "").strip()
    return d[:10] if d else None


def merge_rows(
    bdc_rows: list[dict[str, str]], gee_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    by_date: dict[str, tuple[dict[str, str], str]] = {}
    for row in bdc_rows:
        d = _normalize_date(row)
        if d is None:
            continue
        by_date[d] = (row, "bdc")
    for row in gee_rows:
        d = _normalize_date(row)
        if d is None:
            continue
        if d not in by_date:
            by_date[d] = (row, "gee")
            continue
        existing, _src = by_date[d]
        bdc_n = _valid_count(existing)
        gee_n = _valid_count(row)
        # Prefer higher valid_pixel_count; tie → GEE
        if gee_n >= bdc_n:
            by_date[d] = (row, "gee")
    out: list[dict[str, str]] = []
    for d in sorted(by_date):
        row, _ = by_date[d]
        cleaned = {k: row.get(k, "") for k in PREFERRED_FIELDS}
        # Fill any extra columns from source if PREFERRED missing keys
        for k, v in row.items():
            if k not in cleaned:
                cleaned[k] = v
        out.append(cleaned)
    return out


def _season_dirs(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {
        p.name
        for p in root.iterdir()
        if p.is_dir() and "-" in p.name and p.name[0].isdigit()
    }


def _muni_codes(season_dir: Path, prefix: str) -> set[str]:
    if not season_dir.is_dir():
        return set()
    return {
        p.name
        for p in season_dir.iterdir()
        if p.is_dir() and p.name.startswith(prefix)
    }


def main() -> None:
    args = parse_args()
    bdc_root = args.bdc_root.expanduser().resolve()
    gee_root = args.gee_root.expanduser().resolve()
    prefix = args.muni_prefix.strip()

    if args.seasons:
        seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    else:
        seasons = sorted(_season_dirs(bdc_root) | _season_dirs(gee_root))

    n_written = 0
    n_backed = 0
    n_unchanged = 0
    for season in seasons:
        codes = _muni_codes(bdc_root / season, prefix) | _muni_codes(
            gee_root / season, prefix
        )
        for code in sorted(codes):
            bdc_path = bdc_root / season / code / f"{code}_zonal.csv"
            gee_path = gee_root / season / code / f"{code}_zonal.csv"
            bdc_rows = _read_rows(bdc_path)
            gee_rows = _read_rows(gee_path)
            if not gee_rows and not bdc_rows:
                continue
            if not gee_rows:
                n_unchanged += 1
                continue
            merged = merge_rows(bdc_rows, gee_rows)
            dest = bdc_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            backup = dest.with_name(f"{code}_zonal.bdc.csv")
            will_backup = (
                dest.is_file()
                and bdc_rows
                and not backup.is_file()
                and len(merged) != len(bdc_rows)
            ) or (
                dest.is_file()
                and bdc_rows
                and not backup.is_file()
                and merged != bdc_rows
            )
            # Always backup existing BDC file once before first overwrite from merge
            if dest.is_file() and bdc_rows and not backup.is_file():
                will_backup = True

            if args.dry_run:
                print(
                    f"{season}/{code}: bdc={len(bdc_rows)} gee={len(gee_rows)} "
                    f"→ merged={len(merged)}"
                    + (" [backup]" if will_backup else "")
                )
                n_written += 1
                continue

            if will_backup:
                shutil.copy2(dest, backup)
                n_backed += 1

            fieldnames = list(PREFERRED_FIELDS)
            extra = [
                k
                for k in merged[0].keys()
                if k not in fieldnames
            ] if merged else []
            fieldnames.extend(extra)
            tmp = dest.with_suffix(".tmp.csv")
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                w.writerows(merged)
            tmp.replace(dest)
            # Copy area sidecar from GEE if BDC lacks it
            gee_area = gee_path.parent / f"{code}_area.csv"
            bdc_area = dest.parent / f"{code}_area.csv"
            if gee_area.is_file() and not bdc_area.is_file():
                shutil.copy2(gee_area, bdc_area)
            n_written += 1
            print(
                f"{season}/{code}: bdc={len(bdc_rows)} gee={len(gee_rows)} "
                f"→ merged={len(merged)}"
            )

    print(
        f"Done: wrote={n_written} backed_up={n_backed} "
        f"gee_empty_skipped={n_unchanged} dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
