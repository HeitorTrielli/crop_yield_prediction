"""Summarise a leave-one-year-out tuning study by held-out year.

Answers whether the model has any year-transferable skill, or whether it just
predicts the pooled training mean (in which case R2 tracks how close the
held-out year's mean is to the training mean).

    python debug_loo_summary.py results/tuning/productivity_muni_mean_feature_sweep_year_loo
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl


def main() -> None:
    study = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "results/tuning/productivity_muni_mean_feature_sweep_year_loo"
    )
    rows = []
    for trial in sorted(study.glob("trial_*")):
        params = {}
        pj = trial / "params.json"
        if pj.is_file():
            try:
                params = json.loads(pj.read_text())
            except (OSError, ValueError):
                pass
        for fold in sorted(trial.glob("loo*")):
            tl = fold / "training" / "testlog.csv"
            if not tl.is_file():
                continue
            try:
                df = pl.read_csv(tl)
            except (OSError, ValueError):
                continue
            if df.height == 0:
                continue
            r = df.row(-1, named=True)
            rows.append(
                {
                    "trial": trial.name,
                    "holdout": fold.name.replace("loo", ""),
                    "layout": str(params.get("feature_layout", "?")),
                    "rmse": r.get("rmse"),
                    "r2": r.get("r2"),
                    "mae": r.get("mae"),
                }
            )

    df = pl.DataFrame(rows)
    if df.height == 0:
        raise SystemExit(f"no LOO fold testlogs under {study}")
    print(f"folds: {df.height} across {df['trial'].n_unique()} trials\n")

    print("=== TEST R2 BY HELD-OUT YEAR (across all configurations) ===")
    print(
        df.group_by("holdout")
        .agg(
            pl.len().alias("n_trials"),
            pl.col("r2").median().round(3).alias("r2_med"),
            pl.col("r2").max().round(3).alias("r2_best"),
            pl.col("rmse").median().round(3).alias("rmse_med"),
            pl.col("rmse").min().round(3).alias("rmse_best"),
        )
        .sort("holdout")
    )

    print("\n=== BEST CONFIGURATION PER HELD-OUT YEAR ===")
    for y in sorted(df["holdout"].unique().to_list()):
        b = df.filter(pl.col("holdout") == y).sort("r2", descending=True).row(0, named=True)
        print(
            f"  {y}: r2={b['r2']:+.3f}  rmse={b['rmse']:.3f}  "
            f"{b['trial']}  layout={b['layout'][:60]}"
        )


if __name__ == "__main__":
    main()
