"""
Add train/valid/test split to the yield CSV.

Strategy (year holdout + municipality split):
- All rows in years NOT listed in --holdout-years → train
- Rows in holdout year(s) → valid or test, by municipality:
  - Holdout municipalities are shuffled and split by --valid-ratio
  - Each municipality keeps the same split across all holdout years

Training never sees holdout years. Validation and test only use holdout years,
on disjoint sets of municipalities.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

YIELD_CSV = Path("files/pam_soy_pr_2019_2025.csv")


def add_split_to_csv(
    yield_csv,
    holdout_years,
    valid_ratio=0.5,
    seed=111,
    output_csv=None,
):
    if not (0.0 < valid_ratio < 1.0):
        raise ValueError(f"valid_ratio must be between 0 and 1, got {valid_ratio}")

    if isinstance(holdout_years, (int, str)):
        holdout_years = [int(holdout_years)]
    else:
        holdout_years = [int(y) for y in holdout_years]
    if not holdout_years:
        raise ValueError("holdout_years must contain at least one year")

    holdout_years_set = set(holdout_years)
    test_ratio = 1.0 - valid_ratio

    yield_csv = Path(yield_csv)
    output_csv = Path(output_csv) if output_csv else yield_csv

    print(f"Loading yield CSV from {yield_csv}...")
    yield_df = pd.read_csv(yield_csv)

    municipality_code_col = "municipality_code"
    year_col = "year"
    for col in (municipality_code_col, year_col):
        if col not in yield_df.columns:
            raise ValueError(
                f"Yield CSV missing {col!r}. Available: {list(yield_df.columns)}"
            )

    yield_df[municipality_code_col] = yield_df[municipality_code_col].astype(str)
    yield_df[year_col] = pd.to_numeric(yield_df[year_col], errors="coerce")

    available_years = sorted(yield_df[year_col].dropna().astype(int).unique())
    missing_holdout = holdout_years_set - set(available_years)
    if missing_holdout:
        raise ValueError(
            f"Holdout year(s) not in CSV: {sorted(missing_holdout)}. "
            f"Available years: {available_years}"
        )

    train_years = sorted(set(available_years) - holdout_years_set)
    if not train_years:
        raise ValueError(
            f"All CSV years are holdout ({holdout_years}); nothing left for training."
        )

    print(f"\nTrain years:   {train_years}")
    print(f"Holdout years: {sorted(holdout_years_set)} (valid + test only)")

    holdout_mask = yield_df[year_col].isin(holdout_years_set)
    holdout_df = yield_df.loc[holdout_mask]
    holdout_municipalities = sorted(
        holdout_df[municipality_code_col].dropna().unique().tolist()
    )
    n_holdout_munis = len(holdout_municipalities)
    if n_holdout_munis == 0:
        raise ValueError("No municipalities found in holdout years")

    np.random.seed(seed)
    shuffled = np.array(holdout_municipalities, dtype=object)
    np.random.shuffle(shuffled)

    n_valid_munis = int(round(n_holdout_munis * valid_ratio))
    n_valid_munis = min(max(n_valid_munis, 0), n_holdout_munis)
    n_test_munis = n_holdout_munis - n_valid_munis
    if n_valid_munis == 0 or n_test_munis == 0:
        raise ValueError(
            f"valid_ratio={valid_ratio} yields {n_valid_munis} valid and "
            f"{n_test_munis} test municipalities; need at least 1 in each. "
            f"Adjust --valid-ratio or add more holdout municipalities."
        )

    valid_municipalities = set(shuffled[:n_valid_munis].tolist())
    test_municipalities = set(shuffled[n_valid_munis:].tolist())

    print(f"\nHoldout municipality split (seed={seed}):")
    print(
        f"  Valid: {n_valid_munis} municipalities ({100.0 * n_valid_munis / n_holdout_munis:.1f}%)"
    )
    print(
        f"  Test:  {n_test_munis} municipalities ({100.0 * n_test_munis / n_holdout_munis:.1f}%)"
    )

    if "split" in yield_df.columns:
        print("Overwriting existing 'split' column")
        yield_df = yield_df.drop(columns=["split"])

    def assign_split(row):
        year = int(row[year_col])
        muni = str(row[municipality_code_col])
        if year not in holdout_years_set:
            return "train"
        if muni in valid_municipalities:
            return "valid"
        if muni in test_municipalities:
            return "test"
        raise RuntimeError(f"Unassigned holdout row: {muni}, {year}")

    yield_df["split"] = yield_df.apply(assign_split, axis=1)

    print(f"\nRow counts:")
    for split in ["train", "valid", "test"]:
        split_df = yield_df[yield_df["split"] == split]
        count = len(split_df)
        pct = 100.0 * count / len(yield_df) if len(yield_df) else 0.0
        years = sorted(split_df[year_col].dropna().astype(int).unique().tolist())
        print(f"  {split}: {count} rows ({pct:.1f}%), years {years}")

    print(f"\nUnique municipalities per split:")
    for split in ["train", "valid", "test"]:
        split_df = yield_df[yield_df["split"] == split]
        print(f"  {split}: {split_df[municipality_code_col].nunique()} municipalities")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    yield_df.to_csv(output_csv, index=False)
    print(f"\nSaved yield CSV with 'split' column to: {output_csv}")

    return {
        "valid_municipalities": valid_municipalities,
        "test_municipalities": test_municipalities,
        "train_years": train_years,
        "holdout_years": sorted(holdout_years_set),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Add train/valid/test split: train on non-holdout years; "
            "split holdout-year municipalities into valid vs test."
        )
    )
    parser.add_argument(
        "--yield-csv",
        type=str,
        default=None,
        help=f"Yield CSV path (default: {YIELD_CSV})",
    )
    parser.add_argument(
        "--holdout-years",
        type=int,
        nargs="+",
        required=True,
        help=(
            "Harvest year(s) excluded from training. All rows in these years are "
            "valid or test only (municipalities split by --valid-ratio)."
        ),
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.5,
        help=(
            "Fraction of holdout municipalities assigned to validation; "
            "remainder go to test (default: 0.5)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=111,
        help="Random seed for municipality shuffle (default: 111)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: overwrites --yield-csv)",
    )
    args = parser.parse_args()

    yield_csv = Path(args.yield_csv) if args.yield_csv else YIELD_CSV
    if not yield_csv.exists():
        raise FileNotFoundError(f"Yield CSV not found: {yield_csv}")

    add_split_to_csv(
        yield_csv,
        args.holdout_years,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
