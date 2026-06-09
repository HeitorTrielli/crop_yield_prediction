"""
Script to compute evaluation metrics (RMSE, MAE, R²) directly from a CSV file with forecasts.
Can filter by split (train, valid, test) or compute for all splits.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_paths import predictions_dir, run_dir_from_forecasts_csv
from utils_aggregated import regression_metrics

YIELD_CSV = Path("files/pam_soy_pr_2019_2025.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compcomute metrics from forecasts CSV file."
    )
    parser.add_argument(
        "--forecasts-csv",
        type=str,
        required=True,
        help="CSV file with forecasts (must have 'forecast' column and optionally 'split' column)",
    )
    parser.add_argument(
        "--yield-csv",
        type=str,
        default=None,
        help=f"Path to yield CSV with ground truth (default: {YIELD_CSV})",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "valid", "test"],
        help="Filter by split (train/valid/test). If None, computes metrics for all splits separately",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Output CSV file path for metrics summary (optional)",
    )
    args = parser.parse_args()
    args.forecasts_csv = Path(args.forecasts_csv)
    args.yield_csv = Path(args.yield_csv) if args.yield_csv else YIELD_CSV
    if args.output_csv is not None:
        args.output_csv = Path(args.output_csv)

    return args


def compute_metrics_for_split(
    forecasts_df,
    yield_df,
    forecasts_muni_col,
    yield_muni_col,
    yield_col,
    split_name=None,
):
    """Compute metrics for a specific split or all data."""
    if split_name:
        if "split" not in forecasts_df.columns:
            raise ValueError(
                f"Forecasts CSV has no 'split' column; cannot filter by {split_name!r}"
            )
        filtered_df = forecasts_df[forecasts_df["split"] == split_name].copy()
        if len(filtered_df) == 0:
            raise ValueError(f"No municipalities found in forecasts '{split_name}' split")
    else:
        filtered_df = forecasts_df.copy()

    if forecasts_muni_col not in filtered_df.columns:
        raise ValueError(
            f"Forecasts CSV missing {forecasts_muni_col!r}. "
            f"Available: {list(filtered_df.columns)}"
        )
    if yield_muni_col not in yield_df.columns:
        raise ValueError(
            f"Yield CSV missing {yield_muni_col!r}. Available: {list(yield_df.columns)}"
        )
    if yield_col not in yield_df.columns:
        raise ValueError(
            f"Yield CSV missing {yield_col!r}. Available: {list(yield_df.columns)}"
        )

    filtered_df[forecasts_muni_col] = filtered_df[forecasts_muni_col].astype(str)
    yield_df[yield_muni_col] = yield_df[yield_muni_col].astype(str)

    merged_df = filtered_df.merge(
        yield_df[[yield_muni_col, yield_col]],
        left_on=forecasts_muni_col,
        right_on=yield_muni_col,
        how="inner",
    )
    if len(merged_df) == 0:
        raise ValueError(
            "No matching municipalities between forecasts and yield CSV"
        )

    y_pred = pd.to_numeric(merged_df["forecast"], errors="coerce").values
    y_true = pd.to_numeric(merged_df[yield_col], errors="coerce").values
    valid_mask = np.isfinite(y_pred) & np.isfinite(y_true)
    y_pred = y_pred[valid_mask]
    y_true = y_true[valid_mask]
    if len(y_pred) == 0:
        raise ValueError("No valid predictions with ground truth for metric computation")

    metrics = regression_metrics(y_pred, y_true)
    return {
        "split": split_name if split_name else "all",
        "num_municipalities": len(y_pred),
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "mape": metrics["mape"] if not np.isnan(metrics["mape"]) else None,
    }


def main():
    args = parse_args()

    if args.output_csv is None:
        run_dir = run_dir_from_forecasts_csv(args.forecasts_csv)
        if run_dir is not None:
            args.output_csv = (
                predictions_dir(run_dir, create=True) / "metrics_summary.csv"
            )
        else:
            args.output_csv = args.forecasts_csv.parent / "metrics_summary.csv"

    if not args.forecasts_csv.exists():
        raise FileNotFoundError(f"Forecasts CSV not found: {args.forecasts_csv}")
    if not args.yield_csv.exists():
        raise FileNotFoundError(f"Yield CSV not found: {args.yield_csv}")

    print(f"Loading forecasts from {args.forecasts_csv}...")
    forecasts_df = pd.read_csv(args.forecasts_csv)
    if "forecast" not in forecasts_df.columns:
        raise ValueError("Forecasts CSV must have 'forecast' column")
    print(f"Found {len(forecasts_df)} forecasts")

    print(f"Loading ground truth from {args.yield_csv}...")
    yield_df = pd.read_csv(args.yield_csv)

    all_metrics = []
    forecasts_muni_col = "municipality_code"
    yield_muni_col = "municipality_code"
    yield_col = "production_t"

    if args.split:
        print(f"\nComputing metrics for '{args.split}' split...")
        all_metrics.append(
            compute_metrics_for_split(
                forecasts_df,
                yield_df,
                forecasts_muni_col,
                yield_muni_col,
                yield_col,
                args.split,
            )
        )
    elif "split" in forecasts_df.columns:
        for split_name in ["train", "valid", "test"]:
            print(f"\nComputing metrics for '{split_name}' split...")
            all_metrics.append(
                compute_metrics_for_split(
                    forecasts_df,
                    yield_df,
                    forecasts_muni_col,
                    yield_muni_col,
                    yield_col,
                    split_name,
                )
            )
    else:
        print("\nComputing metrics for all forecasts...")
        all_metrics.append(
            compute_metrics_for_split(
                forecasts_df,
                yield_df,
                forecasts_muni_col,
                yield_muni_col,
                yield_col,
                None,
            )
        )

    print(f"\n{'='*60}")
    print("EVALUATION METRICS")
    print(f"{'='*60}")
    for metrics in all_metrics:
        split_name = metrics["split"]
        print(
            f"\n{split_name.upper()} Split ({metrics['num_municipalities']} municipalities):"
        )
        print(f"  RMSE: {metrics['rmse']:.2f} tons")
        print(f"  MAE:  {metrics['mae']:.2f} tons")
        print(f"  R²:   {metrics['r2']:.4f}")
        if metrics["mape"] is not None:
            print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"\n{'='*60}")

    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv(args.output_csv, index=False)
    print(f"\n✓ Saved metrics to {args.output_csv}")


if __name__ == "__main__":
    main()
