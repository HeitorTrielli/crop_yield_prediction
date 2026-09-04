#!/usr/bin/env python3
"""
Fit train-split StandardScaler for every npy input (S2 reflectance + Xavier extras).

Training also does this automatically on the first run (or with --refit-extra-scaler).
Use this script to write files/train_input_scaler.json without starting a full train.

  python preprocessing/fit_xavier_extra_scaler.py \\
      --datapath ~/sits_moco_data/npy \\
      --yield-csv files/pam_soy_pr_2019_2025_coverage_0.8_1.2.csv \\
      --harvest-years 2020,2021,2022,2023,2024 \\
      --no-coverage-filter
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.extra_scaler import (
    DEFAULT_INPUT_SCALER_PATH,
    fit_extra_scaler_from_dataset,
)
from datasets.uscrops_aggregated_npy_polars import (
    DEFAULT_MAX_COVERAGE_RATIO,
    DEFAULT_MIN_COVERAGE_RATIO,
    USCropsAggregatedNPY,
)
from env_config import resolve_datapath


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit train-split StandardScaler for S2 bands + Xavier rain/climate extras."
        )
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default=None,
        help="NPY root (default: SITS_MOCO_DATAPATH)",
    )
    parser.add_argument(
        "--yield-csv",
        type=str,
        default=str(REPO_ROOT / "files" / "pam_soy_pr_2019_2025.csv"),
    )
    parser.add_argument(
        "--harvest-years",
        type=str,
        required=True,
        help="Comma-separated harvest years, e.g. 2020,2021,2022,2023,2024",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_INPUT_SCALER_PATH),
        help=f"JSON destination (default: {DEFAULT_INPUT_SCALER_PATH})",
    )
    parser.add_argument(
        "--min-coverage-ratio",
        type=float,
        default=DEFAULT_MIN_COVERAGE_RATIO,
    )
    parser.add_argument(
        "--max-coverage-ratio",
        type=float,
        default=DEFAULT_MAX_COVERAGE_RATIO,
    )
    parser.add_argument(
        "--no-coverage-filter",
        action="store_true",
        help="Ignore coverage_ratio filters (same as training --no-coverage-filter).",
    )
    args = parser.parse_args()

    harvest_years = {int(x.strip()) for x in args.harvest_years.split(",") if x.strip()}
    min_cov = None if args.no_coverage_filter else args.min_coverage_ratio
    max_cov = None if args.no_coverage_filter else args.max_coverage_ratio

    dataset = USCropsAggregatedNPY(
        mode="train",
        root=resolve_datapath(args.datapath),
        yield_csv=args.yield_csv,
        year=None,
        sequencelength=45,
        harvest_years=harvest_years,
        feature_layout="spectral_xavier_climate",
        min_coverage_ratio=min_cov,
        max_coverage_ratio=max_cov,
    )
    scaler = fit_extra_scaler_from_dataset(dataset)
    dest = scaler.save(args.output)
    print(f"Saved {dest}")
    for line in scaler.summary_lines()[1:]:
        print(line)


if __name__ == "__main__":
    main()
