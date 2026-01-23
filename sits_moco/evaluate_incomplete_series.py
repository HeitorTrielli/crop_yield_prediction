"""
Script to evaluate model performance with incomplete time series.
Tests predictions using only the first N time periods (2-6) and computes metrics.
Only evaluates on the test split.
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
        description="Evaluate model with incomplete time series (2-6 periods) on test set."
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
        default="files/yield_dataset",
        help="Path to dataset root directory containing municipality .npy files",
    )
    parser.add_argument(
        "--yield-csv",
        type=str,
        default=None,
        help="Path to yield CSV file with 'split' column (default: files/municipality_production_with_codes.csv)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="incomplete_series_evaluation.csv",
        help="Output CSV file path for results",
    )
    parser.add_argument(
        "--sequencelength",
        type=int,
        default=6,
        help="Maximum sequence length for model (default: 6)",
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


def doy_to_month(doy):
    """Convert DOY (day of year) to month number (1-6).
    Based on preprocessing: month_to_doy = {1: 1, 2: 32, 3: 60, 4: 91, 5: 121, 6: 152}
    """
    if doy <= 1:
        return 1
    elif doy <= 32:
        return 2
    elif doy <= 60:
        return 3
    elif doy <= 91:
        return 4
    elif doy <= 121:
        return 5
    elif doy <= 152:
        return 6
    else:
        return 6  # Default to 6 for any value beyond expected range


def predict_municipality_with_periods(
    model,
    municipality_code,
    year,
    datapath,
    num_periods,
    sequencelength,
    rc,
    interp,
    chunk_size,
    device,
    seed=None,
):
    """
    Predict yield for a single municipality/year using only the first num_periods time periods.

    Args:
        municipality_code: Municipality code
        year: Year for the prediction (None for old structure without years)
        num_periods: Number of time periods to use (2-6). Only includes months 1, 2, ..., num_periods.
                    If a month is missing, subsequent months are not included.
    """
    # Try year-based structure first (matching dataset behavior)
    if year is not None:
        # Try direct path first: datapath/year/municipality_code.npy
        muni_npy_file = datapath / str(year) / f"{municipality_code}.npy"
        if not muni_npy_file.exists():
            # Try subdirectory path: datapath/year/municipality_code/municipality_code.npy
            muni_npy_file = (
                datapath / str(year) / municipality_code / f"{municipality_code}.npy"
            )
        # Fallback to old structure: datapath/municipality_code/municipality_code.npy
        if not muni_npy_file.exists():
            muni_npy_file = datapath / municipality_code / f"{municipality_code}.npy"
    else:
        # Old structure: datapath/municipality_code/municipality_code.npy
        muni_npy_file = datapath / municipality_code / f"{municipality_code}.npy"

    if not muni_npy_file.exists():
        return None

    try:
        municipality_data = np.load(muni_npy_file, mmap_mode="r")
    except Exception as e:
        return None

    if len(municipality_data) == 0:
        return None

    model.eval()
    municipality_sum = None

    with torch.no_grad():
        # Process pixels in chunks
        current_chunk = []
        for pixel_index in range(len(municipality_data)):
            X = municipality_data[pixel_index].copy()

            # Filter to only include the first num_periods consecutive months
            # X shape: [num_months, 11] where 11 = [10 bands + DOY]
            # Column 10 contains DOY values
            doys = X[:, 10].astype(int)  # Extract DOY values
            month_numbers = np.array([doy_to_month(doy) for doy in doys])

            # Find the first num_periods consecutive months starting from month 1
            # We need months 1, 2, ..., num_periods in order
            # If a month is missing (e.g., month 3), we stop at month 2
            valid_indices = []
            expected_month = 1
            for idx, month_num in enumerate(month_numbers):
                if month_num == expected_month:
                    valid_indices.append(idx)
                    expected_month += 1
                    if expected_month > num_periods:
                        break
                elif month_num > expected_month:
                    # We've skipped a month, stop here
                    break

            # Only keep entries that are part of the first num_periods consecutive months
            if len(valid_indices) == 0:
                continue

            X = X[valid_indices, :]

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
    if hasattr(model, "_orig_mod"):
        model._orig_mod.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    print("Model loaded successfully")

    # Load test municipalities from yield CSV
    yield_csv_path = Path(
        args.yield_csv
        if args.yield_csv
        else "files/municipality_production_with_codes.csv"
    )

    if not yield_csv_path.exists():
        raise FileNotFoundError(
            f"Yield CSV not found at {yield_csv_path}. Please specify --yield-csv"
        )

    print(f"Loading yield CSV from {yield_csv_path}...")
    yield_df = pd.read_csv(yield_csv_path)

    # Find columns
    muni_code_col = None
    for col in ["municipality_code", "code", "municipality", "muni_code"]:
        if col in yield_df.columns:
            muni_code_col = col
            break

    if muni_code_col is None:
        raise ValueError(
            f"Could not find municipality code column. Available: {list(yield_df.columns)}"
        )

    yield_col = None
    for col in ["production", "yield", "yield_tons", "tons"]:
        if col in yield_df.columns:
            yield_col = col
            break

    if yield_col is None:
        raise ValueError(
            f"Could not find yield column. Available: {list(yield_df.columns)}"
        )

    # Check if split column exists
    if "split" not in yield_df.columns:
        raise ValueError(
            f"CSV file {yield_csv_path} does not have 'split' column. "
            f"Run add_train_valid_eval_split.py first."
        )

    # Filter test municipalities (test years are marked as "test" in split column)
    yield_df[muni_code_col] = yield_df[muni_code_col].astype(str)
    test_df = yield_df[yield_df["split"] == "test"].copy()

    if len(test_df) == 0:
        raise ValueError(
            f"CSV file {yield_csv_path} does not have 'test' split. "
            f"Available splits: {yield_df['split'].unique()}"
        )

    print("Using 'test' split (test years)")

    # Check if year column exists
    year_col = None
    if "year" in test_df.columns:
        year_col = "year"
    elif "Year" in test_df.columns:
        year_col = "Year"
    elif "YEAR" in test_df.columns:
        year_col = "YEAR"

    if year_col is None:
        print(
            "⚠️  Warning: No 'year' column found in CSV. Using municipality_code only."
        )
        print("   This may cause issues if there are multiple years per municipality.")
        municipality_list = test_df[muni_code_col].astype(str).tolist()
        # Create mapping from municipality code to actual yield (old behavior)
        muni_to_yield = dict(zip(test_df[muni_code_col], test_df[yield_col]))
    else:
        # Use (municipality_code, year) pairs
        test_df[year_col] = test_df[year_col].astype(int)
        municipality_list = [
            (str(muni_code), int(year))
            for muni_code, year in zip(test_df[muni_code_col], test_df[year_col])
        ]
        # Create mapping from (municipality_code, year) to actual yield
        muni_to_yield = {
            (str(muni_code), int(year)): yield_val
            for muni_code, year, yield_val in zip(
                test_df[muni_code_col], test_df[year_col], test_df[yield_col]
            )
        }

    print(
        f"Found {len(municipality_list)} municipality{'/year' if year_col else ''} entries in test split"
    )

    if len(municipality_list) == 0:
        print("⚠️  Warning: No municipalities found in test split!")
        return

    # Test with different numbers of time periods
    num_periods_list = [2, 3, 4, 5, 6]
    all_results = []

    for num_periods in num_periods_list:
        print(f"\n{'='*60}")
        print(f"Evaluating with {num_periods} time periods")
        print(f"{'='*60}")

        predictions = {}
        failed_entries = []

        for entry in tqdm(
            municipality_list, desc=f"Predicting ({num_periods} periods)"
        ):
            if year_col is not None:
                # Entry is (municipality_code, year) tuple
                municipality_code, year = entry
                prediction = predict_municipality_with_periods(
                    model,
                    municipality_code,
                    year,
                    args.datapath,
                    num_periods,
                    args.sequencelength,
                    args.rc,
                    args.interp,
                    args.chunk_size,
                    device,
                    seed=args.seed,
                )
            else:
                # Entry is just municipality_code (old behavior)
                municipality_code = entry
                prediction = predict_municipality_with_periods(
                    model,
                    municipality_code,
                    None,  # No year available
                    args.datapath,
                    num_periods,
                    args.sequencelength,
                    args.rc,
                    args.interp,
                    args.chunk_size,
                    device,
                    seed=args.seed,
                )

            if prediction is not None:
                predictions[entry] = prediction
            else:
                failed_entries.append(entry)

        # Compute metrics
        y_pred = []
        y_true = []
        valid_entries = []

        for entry in municipality_list:
            if entry in predictions and entry in muni_to_yield:
                pred = predictions[entry]
                true_val = muni_to_yield[entry]

                # Convert to numeric types
                pred = pd.to_numeric(pred, errors="coerce")
                true_val = pd.to_numeric(true_val, errors="coerce")

                # Filter out NaN/inf
                if np.isfinite(pred) and np.isfinite(true_val):
                    y_pred.append(pred)
                    y_true.append(true_val)
                    valid_entries.append(entry)

        if len(y_pred) > 0:
            y_pred = np.array(y_pred)
            y_true = np.array(y_true)

            metrics = regression_metrics(y_pred, y_true)

            print(
                f"\nResults for {num_periods} time periods ({len(y_pred)} municipalities):"
            )
            print(f"  RMSE: {metrics['rmse']:.2f} tons")
            print(f"  MAE:  {metrics['mae']:.2f} tons")
            print(f"  R²:   {metrics['r2']:.4f}")
            if not np.isnan(metrics["mape"]):
                print(f"  MAPE: {metrics['mape']:.2f}%")

            # Store results
            all_results.append(
                {
                    "num_periods": num_periods,
                    "num_municipalities": len(y_pred),
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                    "mape": metrics["mape"] if not np.isnan(metrics["mape"]) else None,
                }
            )
        else:
            print(f"  ⚠️  No valid predictions for {num_periods} time periods")
            all_results.append(
                {
                    "num_periods": num_periods,
                    "num_municipalities": 0,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "r2": np.nan,
                    "mape": np.nan,
                }
            )

        if failed_entries:
            print(f"  ⚠️  Failed to predict {len(failed_entries)} entries")

    # Save results to CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values("num_periods")
        results_df.to_csv(args.output_csv, index=False)

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(results_df.to_string(index=False))
        print(f"\n✓ Saved results to {args.output_csv}")


if __name__ == "__main__":
    main()
