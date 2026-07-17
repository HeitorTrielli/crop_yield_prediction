"""
Script to evaluate model performance with incomplete time series.
Uses daily rows per season: for k=1..6, all observations in season months 1..k are
concatenated in time (sorted by DOY), up to --sequencelength (default 45) earliest days.
Municipality prediction = sum of per-pixel model outputs, compared to CSV label.
"""

import argparse
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from datasets import (
    DEFAULT_MAX_COVERAGE_RATIO,
    DEFAULT_MIN_COVERAGE_RATIO,
    USCropsAggregatedNPY,
    filter_yield_pandas_by_coverage,
)
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
    aggregation_for_target_column,
    regression_metrics,
    resolve_inference_chunk_size,
    resolve_inference_target,
    resolve_model_kwargs,
    resolve_pixel_chunk_size,
    stnet_regression_input_dim_from_state_dict,
)

YIELD_CSV = Path("files/pam_soy_pr_2019_2025.csv")


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
        default=None,
        help="Season root (default: training/config.json; override for a specific harvest folder)",
    )
    parser.add_argument(
        "--yield-csv",
        type=str,
        default=None,
        help=f"Yield CSV (default: training/config.json; fallback {YIELD_CSV})",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Output CSV path (default: {run_dir}/predictions/incomplete_series_evaluation.csv)",
    )
    parser.add_argument(
        "-seq",
        "--sequencelength",
        type=int,
        default=None,
        help="Model max time steps (default: training/config.json)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Pixels per GPU forward pass (default: pixel_chunk_size * chunks_per_grad from training config)",
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
        choices=feature_layout_choices(),
        metavar="NAME",
        help="Override feature layout from training/config.json",
    )
    parser.add_argument(
        "--reference-date",
        type=str,
        default="2022-10-01",
        help="Date where DOY 1 falls, YYYY-MM-DD (default: 2022-10-01)",
    )
    parser.add_argument(
        "--min-coverage-ratio",
        type=float,
        default=DEFAULT_MIN_COVERAGE_RATIO,
        help=(
            "Keep rows with coverage_ratio >= this (default: no filter). "
            "Example: --min-coverage-ratio 0.8 --max-coverage-ratio 1.2"
        ),
    )
    parser.add_argument(
        "--max-coverage-ratio",
        type=float,
        default=DEFAULT_MAX_COVERAGE_RATIO,
        help="Keep rows with coverage_ratio <= this (default: no filter)",
    )
    parser.add_argument(
        "--no-coverage-filter",
        action="store_true",
        help="Do not filter yield CSV rows by coverage_ratio (default behavior)",
    )
    args = parser.parse_args()
    args.reference_date = datetime.strptime(args.reference_date, "%Y-%m-%d").date()
    args.checkpoint = resolve_checkpoint_path(args.checkpoint)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    run_config = load_latest_run_config(args.checkpoint)
    apply_run_config_to_args(args, run_config)
    if args.yield_csv is None:
        args.yield_csv = YIELD_CSV
    args.datapath = Path(args.datapath).resolve()
    if args.no_coverage_filter:
        args.min_coverage_ratio = None
        args.max_coverage_ratio = None
    args.yield_csv = Path(args.yield_csv)
    if args.feature_layout is not None:
        normalize_feature_layout(args.feature_layout)

    args.chunk_size = resolve_inference_chunk_size(run_config, args.chunk_size)

    return args, run_config


def inference_args_from_run_config(
    run_config: dict, *, quiet: bool = True
) -> SimpleNamespace:
    """Minimal args namespace for training's chunk prefetch / H2D pipeline."""
    cli = run_config.get("cli") or {}
    return SimpleNamespace(
        prefetch_chunks=int(cli.get("prefetch_chunks", 2)),
        disable_pipeline_h2d=bool(cli.get("disable_pipeline_h2d", False)),
        quiet_training=quiet,
        h2d_pin_host=bool(cli.get("h2d_pin_host", False)),
    )


def inference_batch_size_from_run_config(run_config: dict, default: int = 10) -> int:
    cli = run_config.get("cli") or {}
    return int(cli.get("batchsize", default))


def predict_municipalities_with_periods(
    model,
    entries: list[tuple[str, int]],
    dataset,
    num_periods: int,
    chunk_size: int,
    device,
    reference_date,
    args,
    aggregation: str | None = None,
) -> dict[tuple[str, int], float]:
    """Batched incomplete-series inference (training chunk/H2D pipeline)."""
    from training.inference_batch import run_period_inference_batch

    if aggregation is None:
        aggregation = aggregation_for_target_column(dataset.target_column)
    return run_period_inference_batch(
        model=model,
        dataset=dataset,
        entries=entries,
        device=device,
        args=args,
        chunk_size=chunk_size,
        num_periods=num_periods,
        reference_date=reference_date,
        aggregation=aggregation,
    )


def predict_municipality_with_periods(
    model,
    municipality_code,
    year,
    dataset,
    num_periods,
    chunk_size,
    device,
    reference_date,
    args=None,
    run_config: dict | None = None,
):
    """Predict one municipality-year (wrapper around batched inference)."""
    if args is None:
        if run_config is None:
            raise ValueError(
                "predict_municipality_with_periods needs args or run_config"
            )
        args = inference_args_from_run_config(run_config)
    if year is None:
        if getattr(dataset, "use_multi_year", False):
            raise ValueError("year required for multi-year dataset")
        year = dataset.municipality_years[str(municipality_code)]
    results = predict_municipalities_with_periods(
        model,
        [(str(municipality_code), int(year))],
        dataset,
        num_periods,
        chunk_size,
        device,
        reference_date,
        args,
    )
    return results.get((str(municipality_code), int(year)))


def batched_predict_with_periods(
    model,
    municipality_list,
    dataset,
    num_periods,
    chunk_size,
    device,
    reference_date,
    args,
    *,
    batch_size: int = 10,
    single_season: bool = False,
    target_year: int | None = None,
    desc: str | None = None,
) -> tuple[dict, list]:
    """Predict incomplete-series yields in municipality batches (training pipeline)."""
    from itertools import islice

    if batch_size < 1:
        batch_size = 1

    predictions: dict = {}
    failed_entries: list = []
    iterator = iter(municipality_list)
    total = len(municipality_list)
    progress = tqdm(total=total, desc=desc, leave=False) if desc else None

    try:
        while True:
            batch = list(islice(iterator, batch_size))
            if not batch:
                break

            if single_season:
                if target_year is None:
                    raise ValueError("target_year required for single_season batching")
                entries = [(str(m), int(target_year)) for m in batch]
                result_keys = list(batch)
            else:
                entries = [(str(m), int(y)) for m, y in batch]
                result_keys = list(batch)

            batch_results = predict_municipalities_with_periods(
                model,
                entries,
                dataset,
                num_periods,
                chunk_size,
                device,
                reference_date,
                args,
            )
            for entry, result_key in zip(entries, result_keys):
                if entry in batch_results:
                    predictions[result_key] = batch_results[entry]
                else:
                    failed_entries.append(result_key)

            if progress is not None:
                progress.update(len(batch))
    finally:
        if progress is not None:
            progress.close()

    return predictions, failed_entries


def main():
    args, run_config = parse_args()

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
    model_kw = resolve_model_kwargs(run_config, checkpoint)
    model = STNetRegression(
        input_dim=input_dim,
        num_outputs=1,
        max_seq_len=args.sequencelength,
        **model_kw,
    ).to(device)

    # Load model weights
    if hasattr(model, "_orig_mod"):
        model._orig_mod.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    print("Model loaded successfully")

    target, target_column, aggregation, target_unit = resolve_inference_target(
        run_config, checkpoint
    )
    print(
        f"Run config: {run_config['config_path']} (session {run_config['session_index']})"
    )
    print(
        f"Target: {target} ({target_column}, {aggregation} over pixels, {target_unit})"
    )

    if not args.yield_csv.exists():
        raise FileNotFoundError(f"Yield CSV not found: {args.yield_csv}")

    print(f"Loading yield CSV from {args.yield_csv}...")
    yield_df = pd.read_csv(args.yield_csv)
    yield_df = filter_yield_pandas_by_coverage(
        yield_df,
        min_ratio=args.min_coverage_ratio,
        max_ratio=args.max_coverage_ratio,
    )

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
    if year_col not in yield_df.columns:
        raise ValueError(
            f"Yield CSV missing {year_col!r}. Available: {list(yield_df.columns)}"
        )

    yield_df[muni_code_col] = yield_df[muni_code_col].astype(str)
    yield_df[year_col] = yield_df[year_col].astype(int)
    print("Using all splits (train/valid/test) for yield targets")
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
        yield_csv=args.yield_csv,
        year=None,
        sequencelength=args.sequencelength,
        randomchoice=args.rc,
        interp=args.interp,
        seed=args.seed,
        feature_layout=normalize_feature_layout(args.feature_layout),
        target_column=target_column,
        min_coverage_ratio=args.min_coverage_ratio,
        max_coverage_ratio=args.max_coverage_ratio,
    )

    single_season = getattr(dataset, "is_single_season", False) and year_col is not None
    target_year: int | None = None
    if single_season:
        # One folder = one safra. Target = yield of second year (e.g. 2022-2023 → 2023).
        folder_name = args.datapath.name
        if "-" in folder_name and folder_name.split("-")[-1].isdigit():
            target_year = int(folder_name.split("-")[-1])
        else:
            target_year = None
        if target_year is None:
            raise ValueError(
                f"Cannot infer harvest year from datapath folder name {folder_name!r}. "
                "Use a season folder like 2022-2023."
            )
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
    inference_args = inference_args_from_run_config(run_config)
    inference_batch_size = inference_batch_size_from_run_config(run_config)

    for num_periods in num_periods_list:
        print(f"\n{'='*60}")
        print(f"Evaluating with {num_periods} time periods")
        print(f"{'='*60}")

        predictions, failed_entries = batched_predict_with_periods(
            model,
            municipality_list,
            dataset,
            num_periods,
            args.chunk_size,
            device,
            reference_date=args.reference_date,
            args=inference_args,
            batch_size=inference_batch_size,
            single_season=single_season,
            target_year=target_year,
            desc=f"Predicting ({num_periods} periods)",
        )

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

            metrics = regression_metrics(y_pred, y_true, target_column=target_column)

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
