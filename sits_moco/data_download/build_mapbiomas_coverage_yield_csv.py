#!/usr/bin/env python3
"""
Build a training yield CSV from MapBiomas soy area vs IBGE PAM planted area.

Keeps municipality–years whose mapbiomas_soy_area_ha / area_planted_ha is in
[min_ratio, max_ratio] (default 0.8–1.2) and that have a municipal .npy
(existing mega-pixel cube and/or a zonal CSV that can be converted).

Splits are copied from --existing-yield-csv when the (muni, year) or municipality
already has a label. New non-holdout rows are train. New holdout-year municipalities
are split valid/test with the same seed as add_train_valid_test_split.py.

Example:
  python data_download/build_mapbiomas_coverage_yield_csv.py \\
      --zonal-root files/zonal_mean \\
      --pam-csv files/pam_soy_br_2020_2024.csv \\
      --existing-yield-csv files/pam_soy_pr_2019_2025_coverage_0.8_1.2.csv \\
      --npy-root ~/sits_moco_data/npy_muni_mean \\
      --output files/pam_soy_br_mapbiomas_coverage_0.8_1.2.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data_download.filter_zonal_area_coverage import load_area_sidecars

OUTPUT_COLUMNS = [
    "municipality_code",
    "municipality_name",
    "year",
    "production_t",
    "yield_t_ha",
    "area_planted_ha",
    "pixel_area_ha",
    "coverage_ratio",
    "production_t_s2_adj",
    "split",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--zonal-root",
        type=Path,
        default=_REPO / "files" / "zonal_mean",
        help="Root with {YYYY-YYYY}/{code}/{code}_area.csv sidecars",
    )
    p.add_argument(
        "--pam-csv",
        type=Path,
        default=_REPO / "files" / "pam_soy_br_2020_2024.csv",
        help="IBGE PAM with municipality_code, year, area_planted_ha, production, yield",
    )
    p.add_argument(
        "--existing-yield-csv",
        type=Path,
        default=_REPO / "files" / "pam_soy_pr_2019_2025_coverage_0.8_1.2.csv",
        help="Previous yield CSV; used to copy train/valid/test labels",
    )
    p.add_argument(
        "--npy-root",
        type=Path,
        default=Path("~/sits_moco_data/npy_muni_mean"),
        help="Municipal mega-pixel .npy root",
    )
    p.add_argument(
        "--min-ratio",
        type=float,
        default=0.8,
    )
    p.add_argument(
        "--max-ratio",
        type=float,
        default=1.2,
    )
    p.add_argument(
        "--holdout-years",
        type=str,
        default="2021",
        help="Comma-separated harvest years that are valid/test only",
    )
    p.add_argument(
        "--valid-ratio",
        type=float,
        default=0.5,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=111,
    )
    p.add_argument(
        "--harvest-years",
        type=str,
        default="2020,2021,2022,2023,2024",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_REPO / "files" / "pam_soy_br_mapbiomas_coverage_0.8_1.2.csv",
    )
    p.add_argument(
        "--keep-list",
        type=Path,
        default=_REPO / "files" / "zonal_mean_keep_0.8_1.2.csv",
        help="Write municipality_code,year rows that pass the filter and have imagery",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=_REPO / "files" / "zonal_mean_area_coverage.csv",
        help="Write the full MapBiomas/IBGE ratio report",
    )
    return p.parse_args()


def code7(value: object) -> str:
    return f"{int(float(value)):07d}"


def harvest_years_set(spec: str) -> set[int]:
    return {int(x.strip()) for x in spec.split(",") if x.strip()}


def load_pam_rows(path: Path) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                code = code7(r["municipality_code"])
                year = int(float(r["year"]))
                area = float(r["area_planted_ha"])
                production = float(r["production_t"])
                yld = float(r["yield_t_ha"])
            except (KeyError, TypeError, ValueError):
                continue
            if area <= 0 or production <= 0 or yld <= 0:
                continue
            out[(code, year)] = {
                "municipality_code": int(code),
                "municipality_name": str(r.get("municipality_name") or "").strip(),
                "year": year,
                "production_t": production,
                "yield_t_ha": yld,
                "area_planted_ha": area,
            }
    return out


def load_existing_splits(path: Path) -> tuple[dict[tuple[str, int], str], dict[str, str]]:
    """(code, year) -> split, and municipality -> split from holdout rows."""
    pair: dict[tuple[str, int], str] = {}
    by_muni: dict[str, str] = {}
    if not path.is_file():
        return pair, by_muni
    df = pd.read_csv(path)
    if "split" not in df.columns:
        return pair, by_muni
    for rec in df.to_dict("records"):
        try:
            code = code7(rec["municipality_code"])
            year = int(float(rec["year"]))
            split = str(rec["split"]).strip().lower()
        except (KeyError, TypeError, ValueError):
            continue
        if split not in {"train", "valid", "test"}:
            continue
        pair[(code, year)] = split
        if split in {"valid", "test"}:
            by_muni[code] = split
    return pair, by_muni


def npy_exists(root: Path, code: str, harvest_year: int) -> bool:
    season = f"{harvest_year - 1}-{harvest_year}"
    p = root / season / code / f"{code}.npy"
    if p.is_file():
        return True
    return (root / season / f"{code}.npy").is_file()


def zonal_csv_exists(root: Path, code: str, harvest_year: int) -> bool:
    season = f"{harvest_year - 1}-{harvest_year}"
    return (root / season / code / f"{code}_zonal.csv").is_file()


def assign_splits(
    rows: list[dict],
    pair_splits: dict[tuple[str, int], str],
    muni_holdout_splits: dict[str, str],
    holdout_years: set[int],
    valid_ratio: float,
    seed: int,
) -> None:
    pending: list[str] = []
    seen: set[str] = set()
    for row in rows:
        code = code7(row["municipality_code"])
        year = int(row["year"])
        if (code, year) in pair_splits:
            row["split"] = pair_splits[(code, year)]
            continue
        if year not in holdout_years:
            row["split"] = "train"
            continue
        if code in muni_holdout_splits:
            row["split"] = muni_holdout_splits[code]
            continue
        row["split"] = ""
        if code not in seen:
            pending.append(code)
            seen.add(code)

    if pending:
        rng = np.random.RandomState(seed)
        shuffled = np.array(pending, dtype=object)
        rng.shuffle(shuffled)
        n_valid = int(round(len(shuffled) * valid_ratio))
        if len(shuffled) == 1:
            n_valid = 1
        elif n_valid <= 0:
            n_valid = 1
        elif n_valid >= len(shuffled):
            n_valid = len(shuffled) - 1
        assigned = {
            str(code): ("valid" if i < n_valid else "test")
            for i, code in enumerate(shuffled)
        }
        for row in rows:
            if row["split"]:
                continue
            row["split"] = assigned[code7(row["municipality_code"])]


def main() -> None:
    args = parse_args()
    zonal_root = args.zonal_root.expanduser().resolve()
    npy_root = args.npy_root.expanduser().resolve()
    harvest_years = harvest_years_set(args.harvest_years)
    holdout_years = harvest_years_set(args.holdout_years)
    pam = load_pam_rows(args.pam_csv.resolve())
    pair_splits, muni_holdout = load_existing_splits(args.existing_yield_csv.resolve())

    sidecars = load_area_sidecars(zonal_root)
    report: list[dict] = []
    kept: list[dict] = []
    n_no_pam = 0
    n_no_imagery = 0
    n_ratio = 0
    for s in sidecars:
        try:
            code = code7(s["municipality_code"])
            year = int(s["season_year"])
            mb_ha = float(s["mapbiomas_soy_area_ha"])
        except (KeyError, TypeError, ValueError):
            continue
        if year not in harvest_years:
            continue
        pam_row = pam.get((code, year))
        ibge_ha = float(pam_row["area_planted_ha"]) if pam_row else None
        ratio = (mb_ha / ibge_ha) if ibge_ha and ibge_ha > 0 else None
        in_range = (
            ratio is not None and args.min_ratio <= ratio <= args.max_ratio
        )
        has_npy = npy_exists(npy_root, code, year)
        has_zonal = zonal_csv_exists(zonal_root, code, year)
        keep = bool(in_range and pam_row and (has_npy or has_zonal))
        report.append(
            {
                "municipality_code": int(code),
                "season_year": year,
                "year_range": s.get("year_range", ""),
                "mapbiomas_soy_area_ha": f"{mb_ha:.4f}",
                "ibge_area_planted_ha": f"{ibge_ha:.4f}" if ibge_ha else "",
                "area_ratio_mapbiomas_over_ibge": f"{ratio:.4f}" if ratio else "",
                "has_npy": has_npy,
                "has_zonal_csv": has_zonal,
                "keep": keep,
            }
        )
        if pam_row is None:
            n_no_pam += 1
            continue
        if not in_range:
            n_ratio += 1
            continue
        if not (has_npy or has_zonal):
            n_no_imagery += 1
            continue
        adj = float(pam_row["production_t"]) * min(float(ratio), 1.0)
        kept.append(
            {
                **pam_row,
                "pixel_area_ha": mb_ha,
                "coverage_ratio": float(ratio),
                "production_t_s2_adj": adj,
                "split": "",
            }
        )

    # One row per (muni, year); last sidecar wins if duplicates.
    uniq: dict[tuple[str, int], dict] = {}
    for row in kept:
        uniq[(code7(row["municipality_code"]), int(row["year"]))] = row
    kept = list(uniq.values())
    assign_splits(
        kept,
        pair_splits,
        muni_holdout,
        holdout_years,
        args.valid_ratio,
        args.seed,
    )

    out_df = pd.DataFrame(kept, columns=OUTPUT_COLUMNS)
    out_df = out_df.sort_values(["municipality_code", "year"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", newline="", encoding="utf-8") as f:
        fields = list(report[0].keys()) if report else ["municipality_code"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(report)

    args.keep_list.parent.mkdir(parents=True, exist_ok=True)
    with open(args.keep_list, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["municipality_code", "year"])
        for row in out_df.itertuples(index=False):
            w.writerow([int(row.municipality_code), int(row.year)])

    n_new = sum(
        1
        for row in kept
        if (code7(row["municipality_code"]), int(row["year"])) not in pair_splits
    )
    print(f"Sidecars in harvest years: {len(report)}")
    print(f"Dropped no PAM: {n_no_pam}; ratio outside [{args.min_ratio}, {args.max_ratio}]: {n_ratio}; no npy/zonal: {n_no_imagery}")
    print(f"Kept {len(out_df)} municipality–years ({out_df['municipality_code'].nunique()} munis)")
    print(f"  of which {n_new} are new vs {args.existing_yield_csv.name}")
    print("  splits:", out_df["split"].value_counts().to_dict())
    print("  years:", out_df.groupby("year").size().to_dict())
    print(f"Wrote {args.output}")
    print(f"Keep list: {args.keep_list}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
