"""
Add train/valid/eval split to yield CSV file based on municipalities.
Splits municipalities (not pixels) to ensure no data leakage.

The split is stored in the yield CSV file as a 'split' column (train/valid/eval),
not in the Parquet index file, since the split is at municipality level.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def add_split_to_csv(
    yield_csv,
    train_ratio=0.6,
    valid_ratio=0.2,
    eval_ratio=0.2,
    seed=111,
    output_csv=None,
):
    """
    Add 'split' column to yield CSV file by splitting municipalities.

    Args:
        yield_csv: Path to yield CSV file (will add 'split' column)
        train_ratio: Proportion of municipalities for training (default 0.6)
        valid_ratio: Proportion of municipalities for validation (default 0.2)
        eval_ratio: Proportion of municipalities for evaluation (default 0.2)
        seed: Random seed for reproducibility
        output_csv: Optional output path (default: overwrites yield_csv)
    """
    # Validate ratios
    assert (
        abs(train_ratio + valid_ratio + eval_ratio - 1.0) < 1e-6
    ), "Ratios must sum to 1.0"

    yield_csv = Path(yield_csv)
    output_csv = Path(output_csv) if output_csv else yield_csv

    # Step 1: Load yield CSV and get unique municipalities
    print(f"Loading yield CSV from {yield_csv}...")
    yield_df = pd.read_csv(yield_csv)

    # Find municipality code column
    municipality_code_col = None
    for col in ["municipality_code", "code", "municipality", "muni_code"]:
        if col in yield_df.columns:
            municipality_code_col = col
            break

    if municipality_code_col is None:
        raise ValueError(
            f"Could not find municipality code column in {yield_csv}. "
            f"Available columns: {list(yield_df.columns)}"
        )

    print(f"Using '{municipality_code_col}' as municipality code column")

    # Convert to string for matching
    yield_df[municipality_code_col] = yield_df[municipality_code_col].astype(str)

    # Get unique municipalities from CSV
    municipalities = sorted(yield_df[municipality_code_col].unique())
    num_municipalities = len(municipalities)
    print(f"Found {num_municipalities} unique municipalities in CSV")

    # Step 2: Split municipalities
    np.random.seed(seed)
    indices = np.arange(num_municipalities)
    np.random.shuffle(indices)

    num_train = int(num_municipalities * train_ratio)
    num_valid = int(num_municipalities * valid_ratio)
    num_eval = num_municipalities - num_train - num_valid

    train_indices = indices[:num_train]
    valid_indices = indices[num_train : num_train + num_valid]
    eval_indices = indices[num_train + num_valid :]

    # Create mapping: municipality_code -> split
    municipality_to_split = {}
    for idx in train_indices:
        municipality_to_split[municipalities[idx]] = "train"
    for idx in valid_indices:
        municipality_to_split[municipalities[idx]] = "valid"
    for idx in eval_indices:
        municipality_to_split[municipalities[idx]] = "eval"

    print(f"\nSplit statistics:")
    print(f"  Train: {num_train} municipalities ({train_ratio*100:.1f}%)")
    print(f"  Valid: {num_valid} municipalities ({valid_ratio*100:.1f}%)")
    print(f"  Eval:  {num_eval} municipalities ({eval_ratio*100:.1f}%)")

    # Step 2: Add split column to CSV
    # Check if split column already exists
    if "split" in yield_df.columns:
        print("⚠️  Warning: 'split' column already exists. Overwriting...")
        yield_df = yield_df.drop(columns=["split"])

    # Add split column
    yield_df["split"] = yield_df[municipality_code_col].map(municipality_to_split)

    # Verify all municipalities got a split
    missing_splits = yield_df["split"].isna().sum()
    if missing_splits > 0:
        print(
            f"⚠️  Warning: {missing_splits} municipalities did not get a split assignment."
        )

    # Count municipalities per split
    print(f"\nMunicipality counts in CSV:")
    for split in ["train", "valid", "eval"]:
        count = (yield_df["split"] == split).sum()
        pct = count / len(yield_df) * 100 if len(yield_df) > 0 else 0
        print(f"  {split}: {count} municipalities ({pct:.1f}%)")

    # Save updated CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    yield_df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved yield CSV with 'split' column to: {output_csv}")

    return municipality_to_split


def main():
    parser = argparse.ArgumentParser(
        description="Add train/valid/eval split to yield CSV file based on municipalities"
    )
    parser.add_argument(
        "--yield-csv",
        type=str,
        default=None,
        help="Path to yield CSV file. Default: files/municipality_production_with_codes.csv",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Proportion for training (default: 0.6)",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.2,
        help="Proportion for validation (default: 0.2)",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.2,
        help="Proportion for evaluation (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=111,
        help="Random seed (default: 111)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: overwrites yield-csv)",
    )

    args = parser.parse_args()

    # Auto-detect yield CSV if not specified
    if args.yield_csv:
        yield_csv = Path(args.yield_csv)
    else:
        yield_csv = Path("files/municipality_production_with_codes.csv")
        if not yield_csv.exists():
            raise FileNotFoundError(
                f"Could not find yield CSV. Expected {yield_csv} or specify --yield-csv"
            )

    add_split_to_csv(
        yield_csv,
        args.train_ratio,
        args.valid_ratio,
        args.eval_ratio,
        args.seed,
        args.output,
    )


if __name__ == "__main__":
    main()
