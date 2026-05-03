"""
Script to load the best trained model and generate yield forecasts for all municipalities.
Processes .npy files, sums pixel-level predictions per municipality, and saves to CSV.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from tqdm import tqdm

from datasets.datautils import getWeight
from models import STNetRegression
from utils import recursive_todevice
from utils_aggregated import regression_metrics


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
    args = parser.parse_args()

    args.datapath = Path(args.datapath)
    args.checkpoint = Path(args.checkpoint)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


def transform_pixel(x, sequencelength, rc, interp, seed=None):
    """Transform pixel data: normalize, pad/sample to sequencelength, extract DOY."""
    if seed is not None:
        np.random.seed(seed)

    # Extract DOY (stored as float32, convert to int for indexing)
    doy = x[:, -1].astype(np.int32)
    x = x[:, :10] * 1e-4

    # Normalization parameters (same as dataset)
    mean = np.array(
        [[0.147, 0.169, 0.186, 0.221, 0.273, 0.297, 0.308, 0.316, 0.256, 0.188]]
    )
    std = np.array([0.227, 0.219, 0.222, 0.22, 0.2, 0.193, 0.192, 0.182, 0.123, 0.106])

    weight = getWeight(x)
    x = (x - mean) / std

    if interp:
        doy_pad = np.linspace(0, 366, sequencelength).astype("int")
        x_pad = np.array([np.interp(doy_pad, doy, x[:, i]) for i in range(10)]).T
        weight_pad = getWeight(x_pad * std + mean)
        mask = np.ones((sequencelength,), dtype=int)
    elif rc:
        replace = False if x.shape[0] >= sequencelength else True
        idxs = np.random.choice(x.shape[0], sequencelength, replace=replace)
        idxs.sort()
        x_pad = x[idxs]
        mask = np.ones((sequencelength,), dtype=int)
        doy_pad = doy[idxs]
        weight_pad = weight[idxs]
        weight_pad /= weight_pad.sum()
    else:
        x_length, c_length = x.shape
        if x_length == sequencelength:
            mask = np.ones((sequencelength,), dtype=int)
            x_pad = x
            doy_pad = doy
            weight_pad = weight
            weight_pad /= weight_pad.sum()
        elif x_length < sequencelength:
            mask = np.zeros((sequencelength,), dtype=int)
            mask[:x_length] = 1
            x_pad = np.zeros((sequencelength, c_length))
            x_pad[:x_length, :] = x[:x_length, :]
            doy_pad = np.zeros((sequencelength,), dtype=int)
            doy_pad[:x_length] = doy[:x_length]
            weight_pad = np.zeros((sequencelength,), dtype=float)
            weight_pad[:x_length] = weight[:x_length]
            weight_pad /= weight_pad.sum() if weight_pad.sum() > 0 else 1
        else:
            idxs = np.random.choice(x.shape[0], sequencelength, replace=False)
            idxs.sort()
            x_pad = x[idxs]
            mask = np.ones((sequencelength,), dtype=int)
            doy_pad = doy[idxs]
            weight_pad = weight[idxs]
            weight_pad /= weight_pad.sum()

    return (
        torch.from_numpy(x_pad).type(torch.FloatTensor),
        torch.from_numpy(mask == 0),
        torch.from_numpy(doy_pad).type(torch.LongTensor),
        torch.from_numpy(weight_pad).type(torch.FloatTensor),
    )


def predict_municipality(
    model,
    municipality_code,
    datapath,
    sequencelength,
    rc,
    interp,
    chunk_size,
    device,
    seed=None,
):
    """Predict yield for a single municipality by summing pixel-level predictions."""
    muni_npy_file = datapath / municipality_code / f"{municipality_code}.npy"

    if not muni_npy_file.exists():
        print(f"  ⚠️  Warning: {muni_npy_file} does not exist, skipping")
        return None

    try:
        municipality_data = np.load(muni_npy_file, mmap_mode="r")
    except Exception as e:
        print(f"  ⚠️  Warning: Could not load {muni_npy_file}: {e}")
        return None

    if len(municipality_data) == 0:
        print(f"  ⚠️  Warning: {municipality_code} has 0 pixels, skipping")
        return None

    model.eval()
    municipality_sum = None

    with torch.no_grad():
        # Process pixels in chunks
        current_chunk = []
        for pixel_index in range(len(municipality_data)):
            X = municipality_data[pixel_index].copy()
            X_tuple = transform_pixel(X, sequencelength, rc, interp, seed=seed)
            current_chunk.append(X_tuple)

            if len(current_chunk) >= chunk_size:
                # Process chunk
                chunk_x = torch.stack([p[0] for p in current_chunk])
                chunk_mask = torch.stack([p[1] for p in current_chunk])
                chunk_doy = torch.stack([p[2] for p in current_chunk])
                chunk_weight = torch.stack([p[3] for p in current_chunk])

                municipality_X_chunk = (chunk_x, chunk_mask, chunk_doy, chunk_weight)
                municipality_X_chunk = recursive_todevice(municipality_X_chunk, device)

                # Use autocast for faster computation
                with (
                    autocast("cuda", dtype=torch.bfloat16)
                    if device == "cuda"
                    else torch.no_grad()
                ):
                    chunk_predictions = model(municipality_X_chunk)

                chunk_sum = chunk_predictions.sum(dim=0).to(torch.float32)

                if municipality_sum is None:
                    municipality_sum = chunk_sum
                else:
                    municipality_sum = municipality_sum.to(
                        torch.float32
                    ) + chunk_sum.to(torch.float32)

                current_chunk = []

        # Process remaining pixels
        if current_chunk:
            chunk_x = torch.stack([p[0] for p in current_chunk])
            chunk_mask = torch.stack([p[1] for p in current_chunk])
            chunk_doy = torch.stack([p[2] for p in current_chunk])
            chunk_weight = torch.stack([p[3] for p in current_chunk])

            municipality_X_chunk = (chunk_x, chunk_mask, chunk_doy, chunk_weight)
            municipality_X_chunk = recursive_todevice(municipality_X_chunk, device)

            with (
                autocast("cuda", dtype=torch.bfloat16)
                if device == "cuda"
                else torch.no_grad()
            ):
                chunk_predictions = model(municipality_X_chunk)

            chunk_sum = chunk_predictions.sum(dim=0).to(torch.float32)

            if municipality_sum is None:
                municipality_sum = chunk_sum
            else:
                municipality_sum = municipality_sum.to(torch.float32) + chunk_sum.to(
                    torch.float32
                )

    if municipality_sum is None:
        return None

    # Return as scalar
    if municipality_sum.dim() > 0:
        municipality_sum = municipality_sum.squeeze()
        if municipality_sum.dim() > 0:
            municipality_sum = (
                municipality_sum[0] if len(municipality_sum) > 0 else torch.tensor(0.0)
            )

    return municipality_sum.item()


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
        input_dim=10,
        num_outputs=1,
        max_seq_len=args.sequencelength,
    ).to(device)

    # Load model weights
    state_dict = checkpoint["model_state"]
    # Handle compiled models
    if hasattr(model, "_orig_mod"):
        model._orig_mod.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    print("Model loaded successfully")

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

    if split_filter:
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

    # Determine output CSV filename
    if args.output_csv is None:
        if split_filter:
            args.output_csv = f"yield_forecasts_{split_filter}.csv"
        else:
            args.output_csv = "yield_forecasts.csv"

    # Generate predictions
    print(f"\nGenerating predictions for {len(municipality_list)} municipalities...")
    results = []
    failed_municipalities = []

    for municipality_code in tqdm(municipality_list, desc="Predicting"):
        prediction = predict_municipality(
            model,
            municipality_code,
            args.datapath,
            args.sequencelength,
            args.rc,
            args.interp,
            args.chunk_size,
            device,
            seed=args.seed,
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
        if args.yield_csv or split_filter:
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
