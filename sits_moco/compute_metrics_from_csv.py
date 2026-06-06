"""
Script to compute evaluation metrics (RMSE, MAE, R²) directly from a CSV file with forecasts.
Can filter by split (eval, valid, train) or compute for all splits.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_paths import predictions_dir, run_dir_from_forecasts_csv
from utils_aggregated import regression_metrics


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
        help="Path to yield CSV file with ground truth (default: files/municipality_production_with_codes.csv)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "valid", "eval"],
        help="Filter by split (train/valid/eval). If None, computes metrics for all splits separately",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Output CSV file path for metrics summary (optional)",
    )
    args = parser.parse_args()
    args.forecasts_csv = Path(args.forecasts_csv)
    if args.output_csv is not None:
        args.output_csv = Path(args.output_csv)

    return args


def find_columns(df):
    """Find municipality code and yield columns in dataframe."""
    muni_code_col = None
    for col in ["municipality_code", "code", "municipality", "muni_code"]:
        if col in df.columns:
            muni_code_col = col
            break

    yield_col = None
    for col in ["production", "yield", "yield_tons", "tons", "actual_yield"]:
        if col in df.columns:
            yield_col = col
            break

    return muni_code_col, yield_col


def compute_metrics_for_split(forecasts_df, yield_df, split_name=None):
    """Compute metrics for a specific split or all data."""
    # Filter by split if specified
    if split_name:
        if "split" not in forecasts_df.columns:
            print(
                f"  ⚠️  Warning: No 'split' column found, cannot filter by '{split_name}'"
            )
            return None

        filtered_df = forecasts_df[forecasts_df["split"] == split_name].copy()
        if len(filtered_df) == 0:
            print(f"  ⚠️  Warning: No municipalities found in '{split_name}' split")
            return None
    else:
        filtered_df = forecasts_df.copy()

    # Find columns
    muni_code_col, _ = find_columns(filtered_df)
    if muni_code_col is None:
        print("  ⚠️  Warning: Could not find municipality code column")
        return None

    # Merge with yield CSV to get ground truth
    yield_muni_col, yield_col = find_columns(yield_df)
    if yield_muni_col is None or yield_col is None:
        print("  ⚠️  Warning: Could not find required columns in yield CSV")
        return None

    # Convert to string for matching
    filtered_df[muni_code_col] = filtered_df[muni_code_col].astype(str)
    yield_df[yield_muni_col] = yield_df[yield_muni_col].astype(str)

    # Merge predictions with ground truth
    merged_df = filtered_df.merge(
        yield_df[[yield_muni_col, yield_col]],
        left_on=muni_code_col,
        right_on=yield_muni_col,
        how="inner",
    )

    if len(merged_df) == 0:
        print(
            "  ⚠️  Warning: No matching municipalities found between forecasts and yield CSV"
        )
        return None

    # Extract predictions and ground truth
    # Convert to numeric, handling any non-numeric values
    y_pred = pd.to_numeric(merged_df["forecast"], errors="coerce").values
    y_true = pd.to_numeric(merged_df[yield_col], errors="coerce").values

    # Filter out NaN/inf values
    valid_mask = np.isfinite(y_pred) & np.isfinite(y_true)
    y_pred = y_pred[valid_mask]
    y_true = y_true[valid_mask]

    if len(y_pred) == 0:
        print(
            "  ⚠️  Warning: No valid predictions with ground truth for metric computation"
        )
        return None

    # Compute metrics
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

    # Load forecasts CSV
    forecasts_csv = args.forecasts_csv
    if not forecasts_csv.exists():
        raise FileNotFoundError(f"Forecasts CSV not found: {forecasts_csv}")

    print(f"Loading forecasts from {forecasts_csv}...")
    forecasts_df = pd.read_csv(forecasts_csv)

    # Check required columns
    if "forecast" not in forecasts_df.columns:
        raise ValueError("Forecasts CSV must have 'forecast' column")

    print(f"Found {len(forecasts_df)} forecasts")

    # Load yield CSV for ground truth
    yield_csv_path = Path(
        args.yield_csv
        if args.yield_csv
        else "files/municipality_production_with_codes.csv"
    )

    if not yield_csv_path.exists():
        raise FileNotFoundError(
            f"Yield CSV not found at {yield_csv_path}. Please specify --yield-csv"
        )

    print(f"Loading ground truth from {yield_csv_path}...")
    yield_df = pd.read_csv(yield_csv_path)

    # Compute metrics
    all_metrics = []

    if args.split:
        # Compute for specific split
        print(f"\nComputing metrics for '{args.split}' split...")
        metrics = compute_metrics_for_split(forecasts_df, yield_df, args.split)
        if metrics:
            all_metrics.append(metrics)
    else:
        # Compute for all splits separately
        if "split" in forecasts_df.columns:
            splits = ["train", "valid", "eval"]
            for split_name in splits:
                print(f"\nComputing metrics for '{split_name}' split...")
                metrics = compute_metrics_for_split(forecasts_df, yield_df, split_name)
                if metrics:
                    all_metrics.append(metrics)
        else:
            # No split column, compute for all data
            print("\nComputing metrics for all forecasts...")
            metrics = compute_metrics_for_split(forecasts_df, yield_df, None)
            if metrics:
                all_metrics.append(metrics)

    # Display results
    if all_metrics:
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

        # Save to CSV if requested
        if args.output_csv:
            results_df = pd.DataFrame(all_metrics)
            results_df.to_csv(args.output_csv, index=False)
            print(f"\n✓ Saved metrics to {args.output_csv}")
    else:
        print("\n⚠️  No metrics computed!")


if __name__ == "__main__":
    main()
