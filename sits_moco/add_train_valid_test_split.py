"""
Add train/valid/test split to yield CSV file based on (municipality, year) pairs.
- Test years: All observations from specified years are marked as 'test'
- Train/Valid split: Remaining observations are split by (municipality, year) pairs.
  The same municipality can be in train for one year and valid for another year.

The split is stored in the yield CSV file as a 'split' column (train/valid/test).
Note: "valid" is used for validation during training, "test" is for final evaluation.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def add_split_to_csv(
    yield_csv,
    test_years=None,
    train_ratio=0.8,
    seed=111,
    output_csv=None,
):
    """
    Add 'split' column to yield CSV file by splitting (municipality, year) pairs.

    - All observations from test_years are marked as 'test'
    - Remaining observations are split by (municipality, year) pairs into 'train' and 'valid'
    - The same municipality can be train in one year and valid in another year

    Args:
        yield_csv: Path to yield CSV file (will add 'split' column)
        test_years: List of years to use for testing (e.g., [2023, 2024])
        train_ratio: Proportion of (municipality, year) pairs for training (default 0.8)
                     valid_ratio is automatically 1 - train_ratio
        seed: Random seed for reproducibility
        output_csv: Optional output path (default: overwrites yield_csv)
    """
    # Validate ratio
    assert 0 < train_ratio < 1, "train_ratio must be between 0 and 1"
    valid_ratio = 1.0 - train_ratio

    # Convert test_years to list if single value provided
    if test_years is None:
        test_years = []
    elif isinstance(test_years, (int, str)):
        test_years = [int(test_years)]
    else:
        test_years = [int(y) for y in test_years]

    yield_csv = Path(yield_csv)
    output_csv = Path(output_csv) if output_csv else yield_csv

    # Step 1: Load yield CSV
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

    # Year column is always 'year'
    year_col = "year"
    if year_col not in yield_df.columns:
        raise ValueError(
            f"Could not find 'year' column in {yield_csv}. "
            f"Available columns: {list(yield_df.columns)}"
        )

    print(f"Using '{municipality_code_col}' as municipality code column")
    print(f"Using '{year_col}' as year column")

    # Convert to appropriate types
    yield_df[municipality_code_col] = yield_df[municipality_code_col].astype(str)
    yield_df[year_col] = pd.to_numeric(yield_df[year_col], errors="coerce")

    # Step 2: Mark test years
    if test_years:
        print(f"\nTest years: {test_years}")
        test_mask = yield_df[year_col].isin(test_years)
        num_test_rows = test_mask.sum()
        print(f"  Found {num_test_rows} observations in test years")
    else:
        test_mask = pd.Series([False] * len(yield_df), index=yield_df.index)
        print("\nNo test years specified - all data will be split into train/eval")

    # Step 3: Get (municipality, year) pairs from non-test years for train/valid split
    non_test_df = yield_df[~test_mask].copy()

    # Get unique (municipality, year) pairs
    pairs = non_test_df[[municipality_code_col, year_col]].drop_duplicates()
    # Convert to list of tuples, ensuring municipality_code is string and year is int
    pairs_list = [
        (
            str(row[municipality_code_col]),
            int(pd.to_numeric(row[year_col], errors="coerce")),
        )
        for _, row in pairs.iterrows()
        if pd.notna(row[year_col])  # Skip rows with NaN years
    ]
    num_pairs = len(pairs_list)
    print(f"\nFound {num_pairs} unique (municipality, year) pairs in non-test years")

    # Step 4: Split (municipality, year) pairs for train/valid
    np.random.seed(seed)
    indices = np.arange(num_pairs)
    np.random.shuffle(indices)

    num_train = int(num_pairs * train_ratio)
    num_valid = num_pairs - num_train

    train_indices = indices[:num_train]
    valid_indices = indices[num_train:]

    # Create mapping: (municipality_code, year) -> split
    pair_to_split = {}
    for idx in train_indices:
        pair_to_split[pairs_list[idx]] = "train"
    for idx in valid_indices:
        pair_to_split[pairs_list[idx]] = "valid"

    print(f"\nSplit statistics (non-test (municipality, year) pairs):")
    print(f"  Train: {num_train} pairs ({train_ratio*100:.1f}%)")
    print(f"  Valid: {num_valid} pairs ({valid_ratio*100:.1f}%)")

    # Step 5: Add split column to CSV
    # Check if split column already exists
    if "split" in yield_df.columns:
        print("⚠️  Warning: 'split' column already exists. Overwriting...")
        yield_df = yield_df.drop(columns=["split"])

    # Initialize split column
    yield_df["split"] = None

    # Mark test years
    if test_years:
        yield_df.loc[test_mask, "split"] = "test"

    # Mark train/valid for non-test years using (municipality, year) pairs
    non_test_mask = ~test_mask

    # Create mapping dict with string keys for faster lookup
    # Ensure consistent formatting: municipality_code (str) + "_" + year (int/float as str)
    pair_to_split_str = {
        f"{str(m)}_{int(y)}": split for (m, y), split in pair_to_split.items()
    }

    # Create a Series with (municipality, year) tuples as string keys for efficient mapping
    # Convert year to int first to handle float years consistently
    pair_series = pd.Series(
        yield_df.loc[non_test_mask, municipality_code_col].astype(str)
        + "_"
        + yield_df.loc[non_test_mask, year_col].fillna(0).astype(int).astype(str),
        index=yield_df.loc[non_test_mask].index,
    )

    # Map pairs to splits
    yield_df.loc[non_test_mask, "split"] = pair_series.map(pair_to_split_str)

    # Verify all rows got a split
    missing_splits = yield_df["split"].isna().sum()
    if missing_splits > 0:
        print(f"⚠️  Warning: {missing_splits} rows did not get a split assignment.")

    # Count rows per split
    print(f"\nRow counts in CSV:")
    for split in ["train", "valid", "test"]:
        count = (yield_df["split"] == split).sum()
        pct = count / len(yield_df) * 100 if len(yield_df) > 0 else 0
        print(f"  {split}: {count} rows ({pct:.1f}%)")

    # Count unique (municipality, year) pairs per split
    print(f"\nUnique (municipality, year) pair counts per split:")
    for split in ["train", "valid", "test"]:
        split_df = yield_df[yield_df["split"] == split]
        if len(split_df) > 0:
            unique_pairs = (
                split_df[[municipality_code_col, year_col]].drop_duplicates().shape[0]
            )
            print(f"  {split}: {unique_pairs} (municipality, year) pairs")

    # Also show unique municipalities per split for reference
    print(f"\nUnique municipality counts per split (for reference):")
    for split in ["train", "valid", "test"]:
        split_df = yield_df[yield_df["split"] == split]
        if len(split_df) > 0:
            unique_munis = split_df[municipality_code_col].nunique()
            print(f"  {split}: {unique_munis} municipalities")

    # Save updated CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    yield_df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved yield CSV with 'split' column to: {output_csv}")

    return pair_to_split


def main():
    parser = argparse.ArgumentParser(
        description="Add train/valid/test split to yield CSV file based on municipalities and years. "
        "Test years are excluded from train/valid split."
    )
    parser.add_argument(
        "--yield-csv",
        type=str,
        default=None,
        help="Path to yield CSV file. Default: files/municipality_production_with_codes.csv",
    )
    parser.add_argument(
        "--test-years",
        type=int,
        nargs="+",
        default=None,
        help="Years to use for testing (e.g., --test-years 2023 2024). All observations from these years will be marked as 'test'.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of (municipality, year) pairs for training (default: 0.8). Valid ratio is automatically 1 - train_ratio. Only applies to non-test years.",
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

    # Validate ratio
    if not (0 < args.train_ratio < 1):
        parser.error(f"train_ratio must be between 0 and 1, got {args.train_ratio}")

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
        args.test_years,
        args.train_ratio,
        args.seed,
        args.output,
    )


if __name__ == "__main__":
    main()
