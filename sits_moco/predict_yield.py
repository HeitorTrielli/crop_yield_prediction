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
from run_paths import (
    apply_run_config_to_args,
    load_latest_run_config,
    predictions_dir,
    resolve_checkpoint_path,
    run_dir_from_path,
)
from utils_aggregated import (
    aggregate_municipality_from_pixel_chunks,
    regression_metrics,
    resolve_inference_target,
    stnet_regression_input_dim_from_state_dict,
)

YIELD_CSV = Path("files/pam_soy_pr_2019_2025.csv")


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
        default=None,
        help="Dataset root (default: training/config.json from --checkpoint)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Output CSV path (default: auto-generated; split suffix if using --test-only, etc.)",
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
        help=f"Yield CSV (default: training/config.json; fallback {YIELD_CSV} if missing)",
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
        "--test-only",
        action="store_true",
        help="Only predict municipalities in the 'test' split",
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
        "-seq",
        "--sequencelength",
        type=int,
        default=None,
        help="Max time steps (default: training/config.json)",
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
        default=None,
        help="Random seed (default: training/config.json)",
    )
    parser.add_argument(
        "--feature-layout",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Override feature layout from training/config.json: "
            + ", ".join(feature_layout_choices())
        ),
    )
    args = parser.parse_args()

    args.checkpoint = resolve_checkpoint_path(args.checkpoint)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    run_config = load_latest_run_config(args.checkpoint)
    apply_run_config_to_args(args, run_config)
    if args.yield_csv is None:
        args.yield_csv = YIELD_CSV
    args.datapath = Path(args.datapath).expanduser()
    args.yield_csv = Path(args.yield_csv).expanduser()
    if args.feature_layout is not None:
        normalize_feature_layout(args.feature_layout)

    return args, run_config


def predict_municipality(
    model,
    municipality_code,
    dataset,
    chunk_size,
    device,
    year=None,
    aggregation: str = "sum",
):
    """Predict yield via USCropsAggregatedNPY (same transform as training)."""
    chunks = dataset.load_pixels_from_municipality(
        municipality_code, year=year, chunk_size=chunk_size
    )
    return aggregate_municipality_from_pixel_chunks(
        model, chunks, device, aggregation=aggregation
    )


def main():
    args, run_config = parse_args()

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
    if args.feature_layout is not None:
        expected_dim = feature_layout_input_dim(args.feature_layout)
        if input_dim != expected_dim:
            raise ValueError(
                f"Checkpoint has input_dim={input_dim} but --feature-layout {args.feature_layout!r} "
                f"implies {expected_dim}. Omit --feature-layout to use training/config.json."
            )
        if ck_fl is not None and normalize_feature_layout(
            ck_fl
        ) != normalize_feature_layout(args.feature_layout):
            raise ValueError(
                f"Checkpoint feature_layout={ck_fl!r} does not match --feature-layout {args.feature_layout!r}."
            )

    target, target_column, aggregation, target_unit = resolve_inference_target(
        run_config, checkpoint
    )
    target_mean = checkpoint.get("target_mean", 0.0)
    target_std = checkpoint.get("target_std", 1.0)
    print(
        f"Run config: {run_config['config_path']} (session {run_config['session_index']})"
    )
    print(
        f"Target: {target} ({target_column}, {aggregation} over pixels, {target_unit})"
    )
    print(
        f"Training normalization stats (for reference): mean={target_mean:.2f}, std={target_std:.2f}"
    )

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

    feature_layout = normalize_feature_layout(args.feature_layout)
    if ck_fl is not None and normalize_feature_layout(ck_fl) != feature_layout:
        raise ValueError(
            f"Checkpoint feature_layout={ck_fl!r} does not match "
            f"training config feature_layout={feature_layout!r}."
        )

    dataset = USCropsAggregatedNPY(
        mode="all",
        root=args.datapath.resolve(),
        yield_csv=args.yield_csv,
        year=None,
        sequencelength=args.sequencelength,
        randomchoice=args.rc,
        interp=args.interp,
        seed=args.seed,
        feature_layout=feature_layout,
        target_column=target_column,
    )

    # Get list of municipalities to predict
    municipality_list = None

    # Check if user wants to filter by split
    split_filter = None
    if args.test_only:
        split_filter = "test"
    elif args.valid_only:
        split_filter = "valid"
    elif args.train_only:
        split_filter = "train"

    if args.restrict_to_yield_year:
        if split_filter:
            raise ValueError(
                "--restrict-to-yield-year cannot be used with --test-only / --valid-only / --train-only"
            )
        if args.yield_year is None:
            raise ValueError("--restrict-to-yield-year requires --yield-year")
        if not args.yield_csv.exists():
            raise FileNotFoundError(f"Yield CSV not found: {args.yield_csv}")
        print(
            f"Restricting list to municipalities with yield year {args.yield_year} "
            f"(from {args.yield_csv.name}) ∩ .npy under {args.datapath}"
        )
        ydf = pd.read_csv(args.yield_csv)
        muni_code_col = "municipality_code"
        year_col = "year"
        if muni_code_col not in ydf.columns:
            raise ValueError(
                f"Yield CSV missing {muni_code_col!r}. Available: {list(ydf.columns)}"
            )
        if year_col not in ydf.columns:
            raise ValueError(
                f"Yield CSV missing {year_col!r}; required for --restrict-to-yield-year"
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
        if not args.yield_csv.exists():
            raise FileNotFoundError(f"Yield CSV not found: {args.yield_csv}")
        print(f"Loading yield CSV from {args.yield_csv}...")
        yield_df = pd.read_csv(args.yield_csv)
        muni_code_col = "municipality_code"
        if muni_code_col not in yield_df.columns:
            raise ValueError(
                f"Yield CSV missing {muni_code_col!r}. Available: {list(yield_df.columns)}"
            )
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
        muni_code_col = "municipality_code"
        if muni_code_col not in muni_df.columns:
            raise ValueError(
                f"Municipalities CSV missing {muni_code_col!r}. "
                f"Available: {list(muni_df.columns)}"
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
            aggregation=aggregation,
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
        if split_filter or args.yield_year is not None or args.restrict_to_yield_year:
            if not args.yield_csv.exists():
                raise FileNotFoundError(f"Yield CSV not found: {args.yield_csv}")
            print(f"\nComputing metrics using ground truth from {args.yield_csv}...")
            yield_df = pd.read_csv(args.yield_csv)
            muni_code_col = "municipality_code"
            yield_col = target_column
            year_col = "year"
            if muni_code_col not in yield_df.columns:
                raise ValueError(
                    f"Yield CSV missing {muni_code_col!r}. Available: {list(yield_df.columns)}"
                )
            if yield_col not in yield_df.columns:
                raise ValueError(
                    f"Yield CSV missing {yield_col!r}. Available: {list(yield_df.columns)}"
                )
            if args.yield_year is not None:
                if year_col not in yield_df.columns:
                    raise ValueError(
                        f"Yield CSV missing {year_col!r}; required for --yield-year"
                    )
                yield_df = yield_df[
                    yield_df[year_col].astype(int) == int(args.yield_year)
                ].copy()
                print(
                    f"  Using yield rows for harvest year {args.yield_year} ({len(yield_df)} rows)"
                )

            yield_df[muni_code_col] = yield_df[muni_code_col].astype(str)
            results_df["municipality_code"] = results_df["municipality_code"].astype(
                str
            )
            merged_df = results_df.merge(
                yield_df[[muni_code_col, yield_col]],
                left_on="municipality_code",
                right_on=muni_code_col,
                how="inner",
            )
            if len(merged_df) == 0:
                raise ValueError(
                    "No matching municipalities between predictions and yield CSV"
                )

            y_pred = merged_df["forecast"].values
            y_true = merged_df[yield_col].values
            valid_mask = np.isfinite(y_pred) & np.isfinite(y_true)
            y_pred = y_pred[valid_mask]
            y_true = y_true[valid_mask]
            if len(y_pred) == 0:
                raise ValueError(
                    "No valid predictions with ground truth for metric computation"
                )

            metrics = regression_metrics(y_pred, y_true, target_column=target_column)
            unit = target_unit
            print(f"\n{'='*60}")
            print(f"EVALUATION METRICS ({len(y_pred)} municipalities)")
            print(f"{'='*60}")
            print(f"  RMSE: {metrics['rmse']:.2f} {unit}")
            print(f"  MAE:  {metrics['mae']:.2f} {unit}")
            print(f"  R²:   {metrics['r2']:.4f}")
            if not np.isnan(metrics["mape"]):
                print(f"  MAPE: {metrics['mape']:.2f}%")
            print(f"{'='*60}")

            if len(merged_df) == len(results_df):
                results_df = results_df.merge(
                    yield_df[[muni_code_col, yield_col]],
                    left_on="municipality_code",
                    right_on=muni_code_col,
                    how="left",
                )
                results_df = results_df.rename(columns={yield_col: "actual_yield"})
                results_df["error"] = (
                    results_df["forecast"] - results_df["actual_yield"]
                )
                results_df["abs_error"] = np.abs(results_df["error"])
                results_df.to_csv(args.output_csv, index=False)
                print("\n✓ Updated CSV with actual_yield, error, and abs_error columns")
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
