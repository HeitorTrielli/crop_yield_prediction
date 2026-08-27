#!/usr/bin/env python3
"""
Compare MapBiomas soybean area (from zonal download sidecars) vs IBGE PAM planted area.

Writes a coverage report and optional keep-list for municipalities whose
mapbiomas_soy_area_ha / ibge_area_planted_ha ratio is within bounds.

Usage:
  python data_download/filter_zonal_area_coverage.py \\
      --zonal-root files/zonal_mean \\
      --pam-csv files/pam_soy_br_2020_2024.csv \\
      --min-ratio 0.8 --max-ratio 1.2 \\
      --output files/zonal_mean_area_coverage.csv

  # Training yield CSV (also writes the keep-list used by zonal_csv_to_muni_npy.py):
  python data_download/build_mapbiomas_coverage_yield_csv.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--zonal-root",
        type=Path,
        required=True,
        help="Root with {year_range}/{muni}/{muni}_area.csv sidecars",
    )
    p.add_argument(
        "--pam-csv",
        type=Path,
        required=True,
        help="IBGE PAM CSV with municipality_code, year, area_planted_ha",
    )
    p.add_argument(
        "--min-ratio",
        type=float,
        default=0.8,
        help="Minimum mapbiomas/IBGE area ratio to keep (default: 0.8)",
    )
    p.add_argument(
        "--max-ratio",
        type=float,
        default=1.2,
        help="Maximum mapbiomas/IBGE area ratio to keep (default: 1.2)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write full report CSV (default: {zonal_root}/area_coverage_report.csv)",
    )
    p.add_argument(
        "--keep-list",
        type=Path,
        default=None,
        help="Optional: write municipality_code,year rows that pass filter",
    )
    return p.parse_args()


def load_area_sidecars(root: Path) -> list[dict]:
    rows: list[dict] = []
    for area_csv in sorted(root.glob("*/*/*_area.csv")):
        with open(area_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
    return rows


def load_pam(path: Path) -> dict[tuple[int, int], float]:
    """(municipality_code, year) -> area_planted_ha."""
    out: dict[tuple[int, int], float] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                code = int(r["municipality_code"])
                year = int(float(r["year"]))
                area = float(r["area_planted_ha"])
            except (KeyError, ValueError, TypeError):
                continue
            if area > 0:
                out[(code, year)] = area
    return out


def main() -> None:
    args = parse_args()
    sidecars = load_area_sidecars(args.zonal_root.resolve())
    pam = load_pam(args.pam_csv.resolve())

    report: list[dict] = []
    for s in sidecars:
        code = int(s["municipality_code"])
        year = int(s["season_year"])
        mb_ha = float(s["mapbiomas_soy_area_ha"])
        ibge_ha = pam.get((code, year))
        ratio = (mb_ha / ibge_ha) if ibge_ha and ibge_ha > 0 else None
        keep = (
            ratio is not None
            and args.min_ratio <= ratio <= args.max_ratio
        )
        report.append(
            {
                "municipality_code": code,
                "season_year": year,
                "year_range": s.get("year_range", ""),
                "mapbiomas_soy_area_ha": f"{mb_ha:.4f}",
                "mapbiomas_soy_pixel_count": s.get("mapbiomas_soy_pixel_count", ""),
                "ibge_area_planted_ha": f"{ibge_ha:.4f}" if ibge_ha else "",
                "area_ratio_mapbiomas_over_ibge": f"{ratio:.4f}" if ratio else "",
                "keep": keep,
            }
        )

    out_path = args.output or (args.zonal_root / "area_coverage_report.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(report[0].keys()) if report else [
        "municipality_code",
        "season_year",
        "mapbiomas_soy_area_ha",
        "ibge_area_planted_ha",
        "area_ratio_mapbiomas_over_ibge",
        "keep",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(report)

    n_keep = sum(1 for r in report if r.get("keep"))
    print(f"Wrote {len(report)} rows to {out_path}")
    print(
        f"Keep {n_keep}/{len(report)} with ratio in "
        f"[{args.min_ratio}, {args.max_ratio}]"
    )

    if args.keep_list:
        args.keep_list.parent.mkdir(parents=True, exist_ok=True)
        with open(args.keep_list, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["municipality_code", "year"])
            for r in report:
                if r.get("keep"):
                    w.writerow([r["municipality_code"], r["season_year"]])
        print(f"Keep list: {args.keep_list}")


if __name__ == "__main__":
    main()
