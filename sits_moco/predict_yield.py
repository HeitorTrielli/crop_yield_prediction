"""
Script to load the best trained model and generate yield forecasts for all municipalities.
Processes .npy files, sums pixel-level predictions per municipality, and saves to CSV.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from datasets import USCropsAggregatedNPY
from datasets.feature_layout import (
    feature_layout_choices,
    feature_layout_input_dim,
    normalize_feature_layout,
)
from models import STNetRegression
from run_paths import predictions_dir, run_dir_from_path
from utils_aggregated import (
    regression_metrics,
    stnet_regression_input_dim_from_state_dict,
    sum_municipality_from_pixel_chunks,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate yield forecasts for municipalities using trained model."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model_best.pth checkpoint file",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="files/npy",
        help="Path to dataset root directory containing municipality .npy files",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Output CSV file path for forecasts (default: auto-generated with split suffix if using --eval-only, etc.)",
    )
    parser.add_argument(
        "--municipalities-csv",
        type=str,
        default=None,
        help="Optional CSV file with municipality codes to predict (if None, uses all in datapath)",
    )
    parser.add_argument(
        "--yield-csv",
        type=str,
        default=None,
        help="Path to yield CSV file with 'split' column (used with --eval-only, --valid-only, --train-only)",
    )
    parser.add_argument(
        "--yield-year",
        type=int,
        default=None,
        help=(
            "If the yield CSV has a year/harvest column, restrict merges (actual_yield, metrics) "
            "to this year only so labels match that season."
        ),
    )
    parser.add_argument(
        "--restrict-to-yield-year",
        action="store_true",
        help=(
            "Requires --yield-year and --yield-csv (or default yield path). "
            "Only predict municipalities that appear in the yield CSV for that year and have a "
            ".npy under --datapath (intersection)."
        ),
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only predict municipalities in the 'eval' split (requires --yield-csv)",
    )
    parser.add_argument(
        "--valid-only",
        action="store_true",
        help="Only predict municipalities in the 'valid' split (requires --yield-csv)",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only predict municipalities in the 'train' split (requires --yield-csv)",
    )
    parser.add_argument(
        "--sequencelength",
        type=int,
        default=6,
        help="Maximum length of time series data (default: 6)",
    )
    parser.add_argument(
        "--rc",
        action="store_true",
        help="Whether to use random choice for time series data",
    )
    parser.add_argument(
        "--interp",
        action="store_true",
        help="Whether to interpolate the time series data",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Number of pixels to process at once (default: 2000)",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        help='torch.Device. either "cpu" or "cuda". default will check by torch.cuda.is_available()',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=27,
        help="Random seed for reproducibility (default: 27)",
    )
    parser.add_argument(
        "--feature-layout",
        type=str,
        default="auto",
        metavar="NAME",
        help=(
            "Optional check against training: "
            + ", ".join(feature_layout_choices())
            + ', or "auto" (default) to infer input width only from the checkpoint.'
        ),
    )
    args = parser.parse_args()

    args.datapath = Path(args.datapath)
    args.checkpoint = Path(args.checkpoint)

    if args.feature_layout != "auto":
        normalize_feature_layout(args.feature_layout)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


def predict_municipality(
    model,
    municipality_code,
    dataset,
    chunk_size,
    device,
    year=None,
):
    """Predict yield via USCropsAggregatedNPY (same transform as training)."""
    chunks = dataset.load_pixels_from_municipality(
        municipality_code, year=year, chunk_size=chunk_size
    )
    return sum_municipality_from_pixel_chunks(model, chunks, device)


def main():
    args = parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=False
    )

    state_dict = checkpoint["model_state"]
    input_dim = stnet_regression_input_dim_from_state_dict(state_dict)
    print(f"STNet input_dim from checkpoint: {input_dim}")
    ck_fl = checkpoint.get("feature_layout")
    if ck_fl is not None:
        print(f"Checkpoint feature_layout={ck_fl!r} (saved at training)")
    if args.feature_layout != "auto":
        expected_dim = feature_layout_input_dim(args.feature_layout)
        if input_dim != expected_dim:
            raise ValueError(
                f"Checkpoint has input_dim={input_dim} but --feature-layout {args.feature_layout!r} "
                f"implies {expected_dim}. Use --feature-layout auto or match training."
            )
        if ck_fl is not None and normalize_feature_layout(
            ck_fl
        ) != normalize_feature_layout(args.feature_layout):
            raise ValueError(
                f"Checkpoint feature_layout={ck_fl!r} does not match --feature-layout {args.feature_layout!r}."
            )

    # Extract normalization parameters (stored for reference, but model outputs are in original scale)
    target_mean = checkpoint.get("target_mean", 0.0)
    target_std = checkpoint.get("target_std", 1.0)
    print(
        f"Training normalization stats (for reference): mean={target_mean:.2f}, std={target_std:.2f}"
    )
    print("Note: Model outputs are in original scale (tons), no denormalization needed")

    # Create model
    print("Creating model...")
    device = torch.device(args.device)
    model = STNetRegression(
        input_dim=input_dim,
        num_outputs=1,
        max_seq_len=args.sequencelength,
    ).to(device)

    # Load model weights
    # Handle compiled models
    if hasattr(model, "_orig_mod"):
        model._orig_mod.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    print("Model loaded successfully")

    if args.feature_layout == "auto":
        if ck_fl is not None:
            feature_layout = normalize_feature_layout(ck_fl)
        elif input_dim == 12:
            feature_layout = "spectral_xavier"
        else:
            feature_layout = "spectral"
    else:
        feature_layout = args.feature_layout

    yield_csv_for_dataset = Path(
        args.yield_csv
        if args.yield_csv
        else "files/municipality_production_with_codes.csv"
    )
    dataset = USCropsAggregatedNPY(
        mode="all",
        root=args.datapath.resolve(),
        yield_csv=yield_csv_for_dataset,
        year=None,
        sequencelength=args.sequencelength,
        randomchoice=args.rc,
        interp=args.interp,
        seed=args.seed,
        feature_layout=feature_layout,
    )

    # Get list of municipalities to predict
    municipality_list = None

    # Check if user wants to filter by split
    split_filter = None
    if args.eval_only:
        split_filter = "eval"
    elif args.valid_only:
        split_filter = "valid"
    elif args.train_only:
        split_filter = "train"

    if args.restrict_to_yield_year:
        if split_filter:
            raise ValueError(
                "--restrict-to-yield-year cannot be used with --eval-only / --valid-only / --train-only"
            )
        if args.yield_year is None:
            raise ValueError("--restrict-to-yield-year requires --yield-year")
        yield_csv_path = Path(
            args.yield_csv
            if args.yield_csv
            else "files/municipality_production_with_codes.csv"
        )
        if not yield_csv_path.exists():
            raise FileNotFoundError(f"Yield CSV not found: {yield_csv_path}")
        print(
            f"Restricting list to municipalities with yield year {args.yield_year} "
            f"(from {yield_csv_path.name}) ∩ .npy under {args.datapath}"
        )
        ydf = pd.read_csv(yield_csv_path)
        muni_code_col = None
        for col in ["municipality_code", "code", "municipality", "muni_code"]:
            if col in ydf.columns:
                muni_code_col = col
                break
        if muni_code_col is None:
            raise ValueError(
                f"Could not find municipality code column. Available: {list(ydf.columns)}"
            )
        year_col = None
        for c in ("year", "Year", "YEAR"):
            if c in ydf.columns:
                year_col = c
                break
        if year_col is None:
            raise ValueError(
                "Yield CSV has no year/Year/YEAR column; cannot use --restrict-to-yield-year"
            )
        ydf = ydf[ydf[year_col].astype(int) == int(args.yield_year)]
        codes_csv = set(ydf[muni_code_col].astype(str).str.strip())
        scanned = set()
        for muni_dir in args.datapath.iterdir():
            if muni_dir.is_dir():
                muni_code = muni_dir.name
                if (muni_dir / f"{muni_code}.npy").is_file():
                    scanned.add(muni_code)
        municipality_list = sorted(codes_csv & scanned)
        print(
            f"Found {len(municipality_list)} municipalities (CSV year {args.yield_year} ∩ datapath)"
        )
        if len(municipality_list) == 0:
            print(
                "⚠️  No municipalities matched — check --datapath season folder and yield CSV."
            )
            return

    elif split_filter:
        # Filter by split from yield CSV
        if not args.yield_csv:
            # Try to find default yield CSV
            default_yield_csv = Path("files/municipality_production_with_codes.csv")
            if default_yield_csv.exists():
                args.yield_csv = str(default_yield_csv)
                print(f"Using default yield CSV: {args.yield_csv}")
            else:
                raise ValueError(
                    f"--{split_filter}-only requires --yield-csv. "
                    f"Could not find default at {default_yield_csv}"
                )

        print(f"Loading yield CSV from {args.yield_csv}...")
        yield_df = pd.read_csv(args.yield_csv)

        # Find municipality code column
        muni_code_col = None
        for col in ["municipality_code", "code", "municipality", "muni_code"]:
            if col in yield_df.columns:
                muni_code_col = col
                break
        if muni_code_col is None:
            raise ValueError(
                f"Could not find municipality code column. Available: {list(yield_df.columns)}"
            )

        # Check if split column exists
        if "split" not in yield_df.columns:
            raise ValueError(
                f"CSV file {args.yield_csv} does not have 'split' column. "
                f"Run add_train_valid_eval_split.py first."
            )

        # Filter by split
        filtered_df = yield_df[yield_df["split"] == split_filter]
        municipality_list = filtered_df[muni_code_col].astype(str).tolist()
        print(
            f"Found {len(municipality_list)} municipalities in '{split_filter}' split"
        )

        if len(municipality_list) == 0:
            print(f"⚠️  Warning: No municipalities found in '{split_filter}' split!")
            return

    elif args.municipalities_csv:
        print(f"Loading municipality list from {args.municipalities_csv}...")
        muni_df = pd.read_csv(args.municipalities_csv)
        # Try to find municipality code column
        muni_code_col = None
        for col in ["municipality_code", "code", "municipality", "muni_code"]:
            if col in muni_df.columns:
                muni_code_col = col
                break
        if muni_code_col is None:
            raise ValueError(
                f"Could not find municipality code column. Available: {list(muni_df.columns)}"
            )
        municipality_list = muni_df[muni_code_col].astype(str).tolist()
        print(f"Found {len(municipality_list)} municipalities in CSV")
    else:
        # Get all municipalities from datapath
        print(f"Scanning {args.datapath} for municipality directories...")
        municipality_list = []
        for muni_dir in args.datapath.iterdir():
            if muni_dir.is_dir():
                muni_code = muni_dir.name
                muni_npy_file = muni_dir / f"{muni_code}.npy"
                if muni_npy_file.exists():
                    municipality_list.append(muni_code)
        municipality_list = sorted(municipality_list)
        print(f"Found {len(municipality_list)} municipalities with .npy files")

    # Determine output CSV filename (default: {run_dir}/predictions/)
    if args.output_csv is None:
        run_dir = run_dir_from_path(args.checkpoint)
        pred_dir = predictions_dir(run_dir, create=True)
        if split_filter:
            args.output_csv = pred_dir / f"yield_forecasts_{split_filter}.csv"
        else:
            args.output_csv = pred_dir / "yield_forecasts.csv"
    else:
        args.output_csv = Path(args.output_csv)

    # Generate predictions
    print(f"\nGenerating predictions for {len(municipality_list)} municipalities...")
    results = []
    failed_municipalities = []

    for municipality_code in tqdm(municipality_list, desc="Predicting"):
        prediction = predict_municipality(
            model,
            municipality_code,
            dataset,
            args.chunk_size,
            device,
        )

        if prediction is not None:
            # Model outputs are already in original scale (tons)
            # The normalization (target_mean, target_std) was only used during training
            # to normalize targets for loss computation, but model outputs raw values
            results.append(
                {
                    "municipality_code": municipality_code,
                    "forecast": prediction,
                }
            )
        else:
            failed_municipalities.append(municipality_code)

    # Save results to CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("municipality_code")
        results_df.to_csv(args.output_csv, index=False)
        print(f"\n✓ Saved {len(results)} forecasts to {args.output_csv}")
        print(
            f"  Forecast range: {results_df['forecast'].min():.2f} - {results_df['forecast'].max():.2f}"
        )
        print(f"  Forecast mean: {results_df['forecast'].mean():.2f}")

        # Compute metrics if yield CSV is available
        if (
            args.yield_csv
            or split_filter
            or args.yield_year is not None
            or args.restrict_to_yield_year
        ):
            yield_csv_path = Path(
                args.yield_csv
                if args.yield_csv
                else "files/municipality_production_with_codes.csv"
            )

            if yield_csv_path.exists():
                print(
                    f"\nComputing metrics using ground truth from {yield_csv_path}..."
                )
                yield_df = pd.read_csv(yield_csv_path)

                if args.yield_year is not None:
                    year_col_f = None
                    for c in ("year", "Year", "YEAR"):
                        if c in yield_df.columns:
                            year_col_f = c
                            break
                    if year_col_f is None:
                        print(
                            "  ⚠️  --yield-year ignored: no year/Year/YEAR column in yield CSV"
                        )
                    else:
                        yield_df = yield_df[
                            yield_df[year_col_f].astype(int) == int(args.yield_year)
                        ].copy()
                        print(
                            f"  Using yield rows for harvest year {args.yield_year} ({len(yield_df)} rows)"
                        )

                # Find columns
                muni_code_col = None
                for col in ["municipality_code", "code", "municipality", "muni_code"]:
                    if col in yield_df.columns:
                        muni_code_col = col
                        break

                yield_col = None
                for col in ["production", "yield", "yield_tons", "tons"]:
                    if col in yield_df.columns:
                        yield_col = col
                        break

                if muni_code_col and yield_col:
                    # Convert to string for matching
                    yield_df[muni_code_col] = yield_df[muni_code_col].astype(str)
                    results_df["municipality_code"] = results_df[
                        "municipality_code"
                    ].astype(str)

                    # Merge predictions with ground truth
                    merged_df = results_df.merge(
                        yield_df[[muni_code_col, yield_col]],
                        left_on="municipality_code",
                        right_on=muni_code_col,
                        how="inner",
                    )

                    if len(merged_df) > 0:
                        y_pred = merged_df["forecast"].values
                        y_true = merged_df[yield_col].values

                        # Filter out NaN/inf values
                        valid_mask = np.isfinite(y_pred) & np.isfinite(y_true)
                        y_pred = y_pred[valid_mask]
                        y_true = y_true[valid_mask]

                        if len(y_pred) > 0:
                            metrics = regression_metrics(y_pred, y_true)

                            print(f"\n{'='*60}")
                            print(f"EVALUATION METRICS ({len(y_pred)} municipalities)")
                            print(f"{'='*60}")
                            print(f"  RMSE: {metrics['rmse']:.2f} tons")
                            print(f"  MAE:  {metrics['mae']:.2f} tons")
                            print(f"  R²:   {metrics['r2']:.4f}")
                            if not np.isnan(metrics["mape"]):
                                print(f"  MAPE: {metrics['mape']:.2f}%")
                            print(f"{'='*60}")

                            # Add metrics to results CSV if requested
                            if len(merged_df) == len(results_df):
                                # All predictions have ground truth
                                results_df = results_df.merge(
                                    yield_df[[muni_code_col, yield_col]],
                                    left_on="municipality_code",
                                    right_on=muni_code_col,
                                    how="left",
                                )
                                results_df = results_df.rename(
                                    columns={yield_col: "actual_yield"}
                                )
                                results_df["error"] = (
                                    results_df["forecast"] - results_df["actual_yield"]
                                )
                                results_df["abs_error"] = np.abs(results_df["error"])
                                results_df.to_csv(args.output_csv, index=False)
                                print(
                                    f"\n✓ Updated CSV with actual_yield, error, and abs_error columns"
                                )
                        else:
                            print(
                                "  ⚠️  No valid predictions with ground truth for metric computation"
                            )
                    else:
                        print(
                            "  ⚠️  No matching municipalities found between predictions and yield CSV"
                        )
                else:
                    print(
                        f"  ⚠️  Could not find required columns in yield CSV for metric computation"
                    )
            else:
                print(
                    f"  ⚠️  Yield CSV not found at {yield_csv_path}, skipping metrics computation"
                )
    else:
        print("\n⚠️  No predictions generated!")

    if failed_municipalities:
        print(f"\n⚠️  Failed to predict {len(failed_municipalities)} municipalities:")
        for muni in failed_municipalities[:10]:  # Show first 10
            print(f"  - {muni}")
        if len(failed_municipalities) > 10:
            print(f"  ... and {len(failed_municipalities) - 10} more")


if __name__ == "__main__":
    main()
