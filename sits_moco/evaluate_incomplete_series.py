"""
Script to evaluate model performance with incomplete time series.
Uses daily rows per season: for k=1..6, all observations in season months 1..k are
concatenated in time (sorted by DOY), up to --sequencelength (default 45) earliest days.
Municipality prediction = sum of per-pixel model outputs, compared to CSV label.
"""

import argparse
from datetime import datetime
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
        description="Evaluate with all daily timesteps in season months 1, then 1–2, …, 1–6 (6 runs)."
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
        default="files/npy/2022-2023",
        help="Season root: npy/2022-2023/<municipio>/<municipio>.npy (one folder per municipality, .npy inside)",
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
        default=None,
        help="Output CSV path (default: {run_dir}/predictions/incomplete_series_evaluation.csv)",
    )
    parser.add_argument(
        "--sequencelength",
        type=int,
        default=45,
        help="Model max time steps; match training -seq. Extra days in a window are dropped (earliest 45 by DOY).",
    )
    parser.add_argument(
        "--rc",
        action="store_true",
        help="Ignored for incomplete-period eval (kept for compatibility with dataset init).",
    )
    parser.add_argument(
        "--interp",
        action="store_true",
        help="Ignored for incomplete-period eval (kept for compatibility with dataset init).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=400,
        help="Number of pixels to process at once (default: 400, match training)",
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
        default="spectral",
        choices=feature_layout_choices(),
        metavar="NAME",
        help="Must match training: spectral (10 inputs) or spectral_xavier (12).",
    )
    parser.add_argument(
        "--reference-date",
        type=str,
        default="2022-10-01",
        help="Date where DOY 1 falls, YYYY-MM-DD (default: 2022-10-01)",
    )
    args = parser.parse_args()
    args.reference_date = datetime.strptime(args.reference_date, "%Y-%m-%d").date()
    args.datapath = Path(args.datapath).resolve()
    args.checkpoint = Path(args.checkpoint)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


def predict_municipality_with_periods(
    model,
    municipality_code,
    year,
    dataset,
    num_periods,
    chunk_size,
    device,
    reference_date,
):
    """Predict yield using first num_periods season months (same PixelTransform as training)."""
    chunks = dataset.load_pixels_from_municipality_with_periods(
        municipality_code, year, num_periods, chunk_size, reference_date
    )
    return sum_municipality_from_pixel_chunks(model, chunks, device)


def main():
    args = parse_args()

    if args.output_csv is None:
        pred_dir = predictions_dir(run_dir_from_path(args.checkpoint), create=True)
        args.output_csv = pred_dir / "incomplete_series_evaluation.csv"
    else:
        args.output_csv = Path(args.output_csv)

    print(f"Data root (year/.npy): {args.datapath}")
    if not args.datapath.exists():
        print(
            f"  ⚠️  Path does not exist. Pass --datapath with the same path you use for training."
        )
    else:
        subdirs = [d.name for d in args.datapath.iterdir() if d.is_dir()][:5]
        print(f"  Subdirs (e.g.): {subdirs}")

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
    expected_dim = feature_layout_input_dim(args.feature_layout)
    if input_dim != expected_dim:
        raise ValueError(
            f"Checkpoint has input_dim={input_dim} but --feature-layout {args.feature_layout!r} "
            f"implies {expected_dim}. Use the same --feature-layout as training."
        )
    ck_fl = checkpoint.get("feature_layout")
    if ck_fl is not None:
        print(f"Checkpoint feature_layout={ck_fl!r} (saved at training)")
        if normalize_feature_layout(ck_fl) != normalize_feature_layout(
            args.feature_layout
        ):
            raise ValueError(
                f"Checkpoint feature_layout={ck_fl!r} does not match --feature-layout {args.feature_layout!r}."
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

    # Use full CSV (train + valid + test) so we have yield targets for all years (e.g. 2023 in train/valid)
    yield_df[muni_code_col] = yield_df[muni_code_col].astype(str)
    print("Using all splits (train/valid/test) for yield targets")

    # Check if year column exists
    year_col = None
    if "year" in yield_df.columns:
        year_col = "year"
    elif "Year" in yield_df.columns:
        year_col = "Year"
    elif "YEAR" in yield_df.columns:
        year_col = "YEAR"

    if year_col is None:
        print(
            "⚠️  Warning: No 'year' column found in CSV. Using municipality_code only."
        )
        muni_to_yield = dict(zip(yield_df[muni_code_col], yield_df[yield_col]))
    else:
        yield_df[year_col] = yield_df[year_col].astype(int)
        muni_to_yield = {
            (str(muni_code), int(year)): yield_val
            for muni_code, year, yield_val in zip(
                yield_df[muni_code_col], yield_df[year_col], yield_df[yield_col]
            )
        }

    # Dataset for vectorized loading and transform (mode="all" = use full CSV so we get all munis with .npy)
    print("Building dataset for pixel loading...")
    dataset = USCropsAggregatedNPY(
        mode="all",
        root=args.datapath,
        yield_csv=yield_csv_path,
        year=None,
        sequencelength=args.sequencelength,
        randomchoice=args.rc,
        interp=args.interp,
        seed=args.seed,
        feature_layout=args.feature_layout,
    )

    single_season = getattr(dataset, "is_single_season", False) and year_col is not None
    if single_season:
        # One folder = one safra. Target = yield of second year (e.g. 2022-2023 → 2023).
        folder_name = args.datapath.name
        if "-" in folder_name and folder_name.split("-")[-1].isdigit():
            target_year = int(folder_name.split("-")[-1])
        else:
            target_year = None
        if target_year is None:
            target_year = 2023  # fallback
        munis_with_npy = set(m for m, y in dataset.municipality_list)
        # Munis that have .npy and appear in CSV (any year)
        munis_in_csv = set(m for m, y in muni_to_yield.keys())
        municipality_list = sorted(m for m in munis_with_npy if m in munis_in_csv)

        # Yield: only target_year (e.g. 2023); if not present, use 0 (do not use another year)
        muni_to_yield_single = {
            m: muni_to_yield.get((m, target_year), 0.0) for m in municipality_list
        }
        print(
            f"Single season (target year {target_year}): {len(municipality_list)} municipalities with .npy and yield (missing {target_year} → 0)"
        )
    else:
        muni_to_yield_single = None
        municipality_list = dataset.municipality_list
        print(
            f"Found {len(municipality_list)} municipality{'/year' if year_col else ''} entries (all splits)"
        )

    if len(municipality_list) == 0:
        print("⚠️  Warning: No municipalities found!")
        return

    # Test with different numbers of time periods
    num_periods_list = [1, 2, 3, 4, 5, 6]
    all_results = []

    for num_periods in num_periods_list:
        print(f"\n{'='*60}")
        print(f"Evaluating with {num_periods} time periods")
        print(f"{'='*60}")

        predictions = {}
        failed_entries = []

        if single_season:
            for municipality_code in tqdm(
                municipality_list, desc=f"Predicting ({num_periods} periods)"
            ):
                prediction = predict_municipality_with_periods(
                    model,
                    municipality_code,
                    None,
                    dataset,
                    num_periods,
                    args.chunk_size,
                    device,
                    reference_date=args.reference_date,
                )
                if prediction is not None:
                    predictions[municipality_code] = prediction
                else:
                    failed_entries.append(municipality_code)
        else:
            for entry in tqdm(
                municipality_list, desc=f"Predicting ({num_periods} periods)"
            ):
                if year_col is not None:
                    municipality_code, year = entry
                else:
                    municipality_code = entry
                    year = None
                prediction = predict_municipality_with_periods(
                    model,
                    municipality_code,
                    year,
                    dataset,
                    num_periods,
                    args.chunk_size,
                    device,
                    reference_date=args.reference_date,
                )
                if prediction is not None:
                    predictions[entry] = prediction
                else:
                    failed_entries.append(entry)

        # Compute metrics
        y_pred = []
        y_true = []
        valid_entries = []
        yield_lookup = muni_to_yield_single if single_season else muni_to_yield

        for entry in municipality_list:
            if entry in predictions and entry in yield_lookup:
                pred = predictions[entry]
                true_val = yield_lookup[entry]

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
