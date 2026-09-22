"""Compare the yield CSV a run trained on against the unfiltered PAM source.

Answers: is the high end of the yield distribution present in training, or is it
removed upstream by the coverage filter / split design?

    python debug_target_csv.py --filtered files/pam_soy_pr_mapbiomas_coverage_0.8_1.2.csv \
                               --raw files/pam_soy_pr_2019_2025.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

NULLS = ["-", "", "nan", "NaN", "null", "NULL"]


def load(p: Path, col: str) -> pl.DataFrame:
    df = pl.read_csv(p, null_values=NULLS, infer_schema_length=10000)
    return df.with_columns(pl.col("municipality_code").cast(pl.Utf8)).filter(
        pl.col(col).is_not_null()
    )


def describe(df: pl.DataFrame, col: str, label: str, years: list[int]) -> None:
    print(f"\n--- {label} ---")
    print(
        df.filter(pl.col("year").is_in(years))
        .group_by("year")
        .agg(
            pl.len().alias("n"),
            pl.col(col).mean().round(3).alias("mean"),
            pl.col(col).std().round(3).alias("std"),
            pl.col(col).min().round(3).alias("min"),
            pl.col(col).quantile(0.90).round(3).alias("p90"),
            pl.col(col).quantile(0.99).round(3).alias("p99"),
            pl.col(col).max().round(3).alias("max"),
        )
        .sort("year")
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--filtered", required=True, type=Path)
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--col", default="yield_t_ha")
    ap.add_argument("--years", default="2020,2021,2022,2023,2024")
    args = ap.parse_args()

    years = [int(y) for y in args.years.split(",")]
    f, r = load(args.filtered, args.col), load(args.raw, args.col)

    describe(r, args.col, f"RAW  {args.raw.name}", years)
    describe(f, args.col, f"USED {args.filtered.name}", years)

    print("\n=== WHAT THE FILTER REMOVED (per year) ===")
    fk = set(zip(f["municipality_code"], f["year"]))
    dropped = r.filter(
        ~pl.struct(["municipality_code", "year"]).map_elements(
            lambda s: (s["municipality_code"], s["year"]) in fk, return_dtype=pl.Boolean
        )
    )
    for y in years:
        ry = r.filter(pl.col("year") == y)[args.col].to_numpy()
        fy = f.filter(pl.col("year") == y)[args.col].to_numpy()
        dy = dropped.filter(pl.col("year") == y)[args.col].to_numpy()
        if ry.size == 0:
            continue
        print(
            f"  {y}: raw n={ry.size:4d} max={ry.max():.3f} | kept n={fy.size:4d} "
            f"max={fy.max() if fy.size else float('nan'):.3f} | "
            f"dropped n={dy.size:4d} "
            f"mean={dy.mean() if dy.size else float('nan'):.3f} "
            f"max={dy.max() if dy.size else float('nan'):.3f}"
        )

    print("\n=== TOP-END CHECK: is the raw source itself capped? ===")
    for y in years:
        v = np.sort(r.filter(pl.col("year") == y)[args.col].to_numpy())
        if v.size < 12:
            continue
        print(f"  {y}: 10 largest raw yields = {np.round(v[-10:], 3).tolist()}")

    print("\n=== SPLIT DESIGN ===")
    if "split" in f.columns:
        print(f.group_by(["split", "year"]).agg(pl.len().alias("n")).sort(["split", "year"]))
        print(
            "\n  If every eval row is a single year, the model is scored on ONE year's\n"
            "  within-year spread while it was trained on pooled across-year spread."
        )


if __name__ == "__main__":
    main()
