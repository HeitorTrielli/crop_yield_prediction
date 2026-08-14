"""
End-to-end post-training results pipeline for a single STNet checkpoint.

Loads the model once and produces:
  1. Incomplete-series municipal evaluation (k=1..6) with per-year forecast CSVs.
     k=6 CSVs are written to predictions/yield_forecasts_{year}.csv (same as predict_yield).
     Also writes a pooled summary plus one summary per leave-out/holdout year.
  2. Guarapuava intramunicipal heatmaps (full series + k=1..6) for the holdout year.
  3. Talhão-level evaluation CSVs for each harvest year.
  4. Paraná municipal error choropleth for the holdout year.
  5. Predicted vs IBGE productivity scatter plots (per harvest year).

Example::

    python generate_results.py \\
        --checkpoint results/MyRun/spectral_xavier/training/model_best.pth \\
        --holdout-year 2021
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from create_pixel_heatmap import (
    reconstruct_spatial_predictions,
    save_individual_municipality_figures,
    save_yield_heatmap_only,
    season_start_from_year_range,
)
from datasets import (
    DEFAULT_MAX_COVERAGE_RATIO,
    DEFAULT_MIN_COVERAGE_RATIO,
    USCropsAggregatedNPY,
    filter_yield_pandas_by_coverage,
)
from datasets.feature_layout import feature_layout_input_dim, normalize_feature_layout
from evaluate_incomplete_series import (
    batched_predict_with_periods,
    inference_args_from_run_config,
    inference_batch_size_from_run_config,
)
from models import STNetRegression
from plot_municipal_yield_map import plot_choropleth_column
from plot_productivity_scatter import run_productivity_scatter_for_forecasts
from predict_yield_talhoes import run_talhao_predictions
from run_paths import (
    apply_run_config_to_args,
    intramunicipal_heatmap_dir,
    load_latest_run_config,
    municipal_heatmaps_dir,
    predictions_dir,
    resolve_checkpoint_path,
    run_dir_from_path,
)
from utils_aggregated import (
    regression_metrics,
    resolve_inference_chunk_size,
    resolve_inference_target,
    resolve_model_kwargs,
    resolve_pixel_chunk_size,
    stnet_regression_input_dim_from_state_dict,
)

REPO_ROOT = Path(__file__).resolve().parent
YIELD_CSV = REPO_ROOT / "files" / "pam_soy_pr_2019_2025.csv"
DEFAULT_TIFFPATH = REPO_ROOT / "files" / "daily_tiff"
DEFAULT_TALHOES = REPO_ROOT / "benchmark" / "talhoes_baseline.geojson"
DEFAULT_BASELINE_CSV = REPO_ROOT / "benchmark" / "produtividade_AgroIA_baseline.csv"
DEFAULT_GUARAPUAVA_CODE = "4109401"
NUM_PERIODS_LIST = list(range(1, 7))


def harvest_to_year_range(harvest_year: int) -> str:
    return f"{harvest_year - 1}-{harvest_year}"


@dataclass
class ModelContext:
    checkpoint: Path
    run_dir: Path
    run_config: dict
    checkpoint_data: dict
    model: STNetRegression
    device: torch.device
    input_dim: int
    feature_layout: str
    sequencelength: int
    seed: int
    rc: bool
    interp: bool
    target: str
    target_column: str
    aggregation: str
    target_unit: str
    chunk_size: int
    reference_date: date


@dataclass
class GenerateResultsOutput:
    incomplete_series_summary: Path | None = None
    incomplete_series_by_leave_out_year: dict[int, Path] = field(default_factory=dict)
    yield_forecasts_by_year: dict[int, Path] = field(default_factory=dict)
    incomplete_forecasts: dict[int, dict[int, Path]] = field(default_factory=dict)
    guarapuava_heatmaps: dict[str, Path] = field(default_factory=dict)
    talhao_forecasts: dict[int, Path] = field(default_factory=dict)
    municipal_error_map: Path | None = None
    productivity_scatter_plots: dict[int, dict[str, Path]] = field(default_factory=dict)


def load_model_context(
    checkpoint: Path | str,
    *,
    datapath: Path | None = None,
    yield_csv: Path | None = None,
    sequencelength: int | None = None,
    seed: int | None = None,
    feature_layout: str | None = None,
    device: str | None = None,
    chunk_size: int | None = None,
    reference_date: date | str | None = None,
    rc: bool | None = None,
    interp: bool | None = None,
) -> ModelContext:
    """Load checkpoint, run config, and STNet model once for reuse across pipeline steps."""
    checkpoint = resolve_checkpoint_path(checkpoint)
    run_dir = run_dir_from_path(checkpoint)
    run_config = load_latest_run_config(checkpoint)

    args = SimpleNamespace(
        datapath=datapath,
        yield_csv=yield_csv or YIELD_CSV,
        sequencelength=sequencelength,
        seed=seed,
        feature_layout=feature_layout,
        rc=rc,
        interp=interp,
    )
    apply_run_config_to_args(args, run_config)
    if args.feature_layout is not None:
        normalize_feature_layout(args.feature_layout)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    if reference_date is None:
        reference_date = date(2022, 10, 1)
    elif isinstance(reference_date, str):
        reference_date = datetime.strptime(reference_date, "%Y-%m-%d").date()

    if rc is None:
        rc = bool(args.rc)
    if interp is None:
        interp = bool(args.interp)

    chunk_size = resolve_inference_chunk_size(run_config, chunk_size)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    checkpoint_data = torch.load(
        checkpoint, map_location=torch_device, weights_only=False
    )
    state_dict = checkpoint_data["model_state"]
    input_dim = stnet_regression_input_dim_from_state_dict(state_dict)
    layout = normalize_feature_layout(args.feature_layout)
    expected_dim = feature_layout_input_dim(layout)
    if input_dim != expected_dim:
        raise ValueError(
            f"Checkpoint has input_dim={input_dim} but feature_layout {layout!r} "
            f"implies {expected_dim}."
        )
    ck_fl = checkpoint_data.get("feature_layout")
    if ck_fl is not None and normalize_feature_layout(ck_fl) != layout:
        raise ValueError(
            f"Checkpoint feature_layout={ck_fl!r} does not match config {layout!r}."
        )

    model_kw = resolve_model_kwargs(run_config, checkpoint_data)
    model = STNetRegression(
        input_dim=input_dim,
        num_outputs=1,
        max_seq_len=args.sequencelength,
        **model_kw,
    ).to(torch_device)
    if hasattr(model, "_orig_mod"):
        model._orig_mod.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    model.eval()

    target, target_column, aggregation, target_unit = resolve_inference_target(
        run_config, checkpoint_data
    )

    return ModelContext(
        checkpoint=checkpoint,
        run_dir=run_dir,
        run_config=run_config,
        checkpoint_data=checkpoint_data,
        model=model,
        device=torch_device,
        input_dim=input_dim,
        feature_layout=layout,
        sequencelength=args.sequencelength,
        seed=args.seed,
        rc=rc,
        interp=interp,
        target=target,
        target_column=target_column,
        aggregation=aggregation,
        target_unit=target_unit,
        chunk_size=chunk_size,
        reference_date=reference_date,
    )


def resolve_harvest_years(
    ctx: ModelContext,
    harvest_years: list[int] | None,
    yield_csv: Path,
) -> list[int]:
    if harvest_years:
        return sorted(int(y) for y in harvest_years)

    computed = ctx.run_config.get("computed") or {}
    from_config = computed.get("harvest_years_set") or computed.get(
        "harvest_years_filter"
    )
    if from_config:
        return sorted(int(y) for y in from_config)

    if not yield_csv.is_file():
        raise FileNotFoundError(f"Yield CSV not found: {yield_csv}")
    ydf = pd.read_csv(yield_csv)
    if "year" not in ydf.columns:
        raise ValueError("Cannot infer harvest years: yield CSV has no 'year' column")
    return sorted(ydf["year"].dropna().astype(int).unique().tolist())


def resolve_leave_out_years(
    ctx: ModelContext,
    holdout_year: int | None = None,
) -> list[int]:
    """Leave-out / holdout years from training split_design, else ``holdout_year``."""
    computed = ctx.run_config.get("computed") or {}
    split = computed.get("split_design") or {}
    holdouts = split.get("holdout_years") or computed.get("holdout_years")
    if holdouts:
        return sorted(int(y) for y in holdouts)
    if holdout_year is not None:
        return [int(holdout_year)]
    return []


def _metrics_row_from_forecasts(
    year_df: pd.DataFrame,
    *,
    num_periods: int,
    target_column: str,
) -> dict:
    """Build one incomplete-series summary row from a year forecasts dataframe."""
    if year_df.empty or "actual_yield" not in year_df.columns:
        return {
            "num_periods": num_periods,
            "num_municipalities": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "mape": np.nan,
        }

    pred = pd.to_numeric(year_df["forecast"], errors="coerce")
    true = pd.to_numeric(year_df["actual_yield"], errors="coerce")
    mask = np.isfinite(pred) & np.isfinite(true)
    y_pred = pred[mask].to_numpy(dtype=float)
    y_true = true[mask].to_numpy(dtype=float)
    if len(y_pred) == 0:
        return {
            "num_periods": num_periods,
            "num_municipalities": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "mape": np.nan,
        }

    metrics = regression_metrics(y_pred, y_true, target_column=target_column)
    return {
        "num_periods": num_periods,
        "num_municipalities": int(len(y_pred)),
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "mape": metrics["mape"] if not np.isnan(metrics["mape"]) else None,
    }


def _load_yield_lookup(
    yield_csv: Path,
    target_column: str,
    *,
    min_coverage_ratio: float | None = DEFAULT_MIN_COVERAGE_RATIO,
    max_coverage_ratio: float | None = DEFAULT_MAX_COVERAGE_RATIO,
) -> dict[tuple[str, int], float]:
    yield_df = pd.read_csv(yield_csv)
    yield_df = filter_yield_pandas_by_coverage(
        yield_df,
        min_ratio=min_coverage_ratio,
        max_ratio=max_coverage_ratio,
    )
    muni_code_col = "municipality_code"
    year_col = "year"
    for col in (muni_code_col, year_col, target_column):
        if col not in yield_df.columns:
            raise ValueError(
                f"Yield CSV missing {col!r}. Available: {list(yield_df.columns)}"
            )
    yield_df[muni_code_col] = yield_df[muni_code_col].astype(str)
    yield_df[year_col] = yield_df[year_col].astype(int)
    return {
        (str(muni_code), int(year)): yield_val
        for muni_code, year, yield_val in zip(
            yield_df[muni_code_col], yield_df[year_col], yield_df[target_column]
        )
    }


def _build_dataset(
    ctx: ModelContext,
    datapath: Path,
    yield_csv: Path,
    harvest_years: list[int] | None = None,
    *,
    min_coverage_ratio: float | None = DEFAULT_MIN_COVERAGE_RATIO,
    max_coverage_ratio: float | None = DEFAULT_MAX_COVERAGE_RATIO,
) -> USCropsAggregatedNPY:
    return USCropsAggregatedNPY(
        mode="all",
        root=datapath.resolve(),
        yield_csv=yield_csv,
        year=None,
        sequencelength=ctx.sequencelength,
        randomchoice=ctx.rc,
        interp=ctx.interp,
        seed=ctx.seed,
        feature_layout=ctx.feature_layout,
        target_column=ctx.target_column,
        harvest_years=harvest_years,
        npy_cache_size=128,
        min_coverage_ratio=min_coverage_ratio,
        max_coverage_ratio=max_coverage_ratio,
    )


def _forecasts_dataframe_for_year(
    predictions: dict[tuple[str, int], float],
    yield_lookup: dict[tuple[str, int], float],
    harvest_year: int,
) -> pd.DataFrame:
    rows = []
    for (muni_code, year), forecast in predictions.items():
        if year != harvest_year:
            continue
        rows.append(
            {
                "municipality_code": muni_code,
                "forecast": forecast,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["municipality_code", "forecast"])

    df = pd.DataFrame(rows).sort_values("municipality_code")
    df["municipality_code"] = df["municipality_code"].astype(str)

    labels = {
        muni: yield_lookup[(muni, harvest_year)]
        for muni in df["municipality_code"]
        if (muni, harvest_year) in yield_lookup
    }
    df["actual_yield"] = df["municipality_code"].map(labels)
    df["error"] = df["forecast"] - df["actual_yield"]
    df["abs_error"] = df["error"].abs()
    return df


def run_incomplete_series_evaluation(
    ctx: ModelContext,
    *,
    datapath: Path,
    yield_csv: Path,
    harvest_years: list[int],
    leave_out_years: list[int] | None = None,
    pred_dir: Path | None = None,
    min_coverage_ratio: float | None = DEFAULT_MIN_COVERAGE_RATIO,
    max_coverage_ratio: float | None = DEFAULT_MAX_COVERAGE_RATIO,
) -> tuple[Path, dict[int, Path], dict[int, Path], dict[int, dict[int, Path]]]:
    """
    Evaluate k=1..6 incomplete seasons.

    Writes:
      - predictions/incomplete_series/k{k}/yield_forecasts_{year}.csv
      - predictions/yield_forecasts_{year}.csv for k=6 (full series)
      - predictions/incomplete_series_evaluation.csv (all harvest years pooled)
      - predictions/incomplete_series_evaluation_{year}.csv for each leave-out year
    """
    pred_dir = pred_dir or predictions_dir(ctx.run_dir, create=True)
    incomplete_root = pred_dir / "incomplete_series"
    summary_path = pred_dir / "incomplete_series_evaluation.csv"
    leave_out_years = sorted(
        {
            int(y)
            for y in (leave_out_years or [])
            if int(y) in set(int(h) for h in harvest_years)
        }
    )

    dataset = _build_dataset(
        ctx,
        datapath,
        yield_csv,
        harvest_years=harvest_years,
        min_coverage_ratio=min_coverage_ratio,
        max_coverage_ratio=max_coverage_ratio,
    )
    municipality_list = dataset.municipality_list
    if not municipality_list:
        raise RuntimeError("No municipalities found for incomplete-series evaluation")

    yield_lookup = _load_yield_lookup(
        yield_csv,
        ctx.target_column,
        min_coverage_ratio=min_coverage_ratio,
        max_coverage_ratio=max_coverage_ratio,
    )

    inference_args = inference_args_from_run_config(ctx.run_config)
    inference_batch_size = inference_batch_size_from_run_config(ctx.run_config)

    all_results: list[dict] = []
    leave_out_results: dict[int, list[dict]] = {year: [] for year in leave_out_years}
    yield_forecasts_by_year: dict[int, Path] = {}
    incomplete_forecasts: dict[int, dict[int, Path]] = {
        year: {} for year in harvest_years
    }

    for num_periods in NUM_PERIODS_LIST:
        print(f"\n{'=' * 60}")
        print(f"Incomplete series: {num_periods} season month(s)")
        print(f"{'=' * 60}")

        predictions, failed_entries = batched_predict_with_periods(
            ctx.model,
            municipality_list,
            dataset,
            num_periods,
            ctx.chunk_size,
            ctx.device,
            reference_date=ctx.reference_date,
            args=inference_args,
            batch_size=inference_batch_size,
            desc=f"Predicting (k={num_periods})",
        )

        y_pred = []
        y_true = []
        for entry in municipality_list:
            if entry in predictions and entry in yield_lookup:
                pred = pd.to_numeric(predictions[entry], errors="coerce")
                true_val = pd.to_numeric(yield_lookup[entry], errors="coerce")
                if np.isfinite(pred) and np.isfinite(true_val):
                    y_pred.append(pred)
                    y_true.append(true_val)

        if y_pred:
            metrics = regression_metrics(
                np.array(y_pred), np.array(y_true), target_column=ctx.target_column
            )
            print(
                f"  Metrics ({len(y_pred)} municipality-years): "
                f"RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, R²={metrics['r2']:.4f}"
            )
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
            print(f"  Failed entries: {len(failed_entries)}")

        out_dir = pred_dir if num_periods == 6 else incomplete_root / f"k{num_periods}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for harvest_year in harvest_years:
            year_df = _forecasts_dataframe_for_year(
                predictions, yield_lookup, harvest_year
            )
            if not year_df.empty:
                csv_path = out_dir / f"yield_forecasts_{harvest_year}.csv"
                year_df.to_csv(csv_path, index=False)
                incomplete_forecasts[harvest_year][num_periods] = csv_path
                if num_periods == 6:
                    yield_forecasts_by_year[harvest_year] = csv_path

            if harvest_year in leave_out_results:
                row = _metrics_row_from_forecasts(
                    year_df,
                    num_periods=num_periods,
                    target_column=ctx.target_column,
                )
                leave_out_results[harvest_year].append(row)
                print(
                    f"  Leave-out {harvest_year} (k={num_periods}, "
                    f"n={row['num_municipalities']}): "
                    f"RMSE={row['rmse']:.2f}, MAE={row['mae']:.2f}, R²={row['r2']:.4f}"
                )

    summary_df = pd.DataFrame(all_results).sort_values("num_periods")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved incomplete-series summary to {summary_path}")

    leave_out_summaries: dict[int, Path] = {}
    for year, rows in leave_out_results.items():
        if not rows:
            continue
        year_path = pred_dir / f"incomplete_series_evaluation_{year}.csv"
        pd.DataFrame(rows).sort_values("num_periods").to_csv(year_path, index=False)
        leave_out_summaries[year] = year_path
        print(f"Saved leave-out incomplete-series summary to {year_path}")

    return (
        summary_path,
        leave_out_summaries,
        yield_forecasts_by_year,
        incomplete_forecasts,
    )


def run_guarapuava_heatmaps(
    ctx: ModelContext,
    *,
    datapath: Path,
    tiffpath: Path,
    holdout_year: int,
    municipality_code: str = DEFAULT_GUARAPUAVA_CODE,
    figures_dir: Path | None = None,
) -> dict[str, Path]:
    """Full-series and k=1..6 yield heatmaps for one municipality and harvest year."""
    year_range = harvest_to_year_range(holdout_year)
    season_datapath = (
        datapath / year_range if (datapath / year_range).is_dir() else datapath
    )
    reference_date = season_start_from_year_range(year_range)

    if figures_dir is None:
        figures_dir = intramunicipal_heatmap_dir(ctx.run_dir, create=True)
    full_dir = figures_dir
    incomplete_dir = figures_dir / "incomplete_series"
    full_dir.mkdir(parents=True, exist_ok=True)
    incomplete_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}
    period_list = [None, *NUM_PERIODS_LIST]

    for num_periods in period_list:
        label = "full" if num_periods is None else f"k{num_periods}"
        k_label = (
            f" (first {num_periods} season month{'s' if num_periods != 1 else ''})"
            if num_periods is not None
            else ""
        )
        print(f"\nGuarapuava heatmap{ k_label}...")

        bundle = reconstruct_spatial_predictions(
            ctx.model,
            municipality_code,
            season_datapath,
            tiffpath,
            holdout_year,
            ctx.sequencelength,
            ctx.rc,
            ctx.interp,
            ctx.chunk_size,
            ctx.device,
            seed=ctx.seed,
            year_range=year_range,
            reference_date=reference_date,
            input_dim=ctx.input_dim,
            num_periods=num_periods,
        )
        if bundle[0] is None:
            print(f"  Skipped {label}: inference failed")
            continue

        (
            prediction_map,
            ndvi_map,
            ndvi_map_median,
            ndvi_map_per_month,
            tiles,
            tile_height,
            tile_width,
        ) = bundle

        if num_periods is None:
            ok = save_individual_municipality_figures(
                municipality_code,
                prediction_map,
                ndvi_map,
                ndvi_map_median,
                ndvi_map_per_month,
                tiles,
                tile_height,
                tile_width,
                full_dir,
                tiffpath=tiffpath,
                year=holdout_year,
                target_unit=ctx.target_unit,
                aggregation=ctx.aggregation,
            )
            if ok:
                from municipality_labels import municipality_output_stem

                stem = municipality_output_stem(municipality_code, tiff_root=tiffpath)
                saved["full_heatmap"] = full_dir / f"{stem}_heatmap.png"
                saved["full_ndvi"] = full_dir / f"{stem}_ndvi_heatmap.png"
        else:
            suffix = f"_k{num_periods}"
            ok = save_yield_heatmap_only(
                municipality_code,
                prediction_map,
                tiles,
                tile_height,
                tile_width,
                incomplete_dir,
                tiffpath=tiffpath,
                year=holdout_year,
                suffix=suffix,
                title_suffix=(
                    f"first {num_periods} season month"
                    + ("s" if num_periods != 1 else "")
                ),
                target_unit=ctx.target_unit,
                aggregation=ctx.aggregation,
            )
            if ok:
                from municipality_labels import municipality_output_stem

                stem = municipality_output_stem(municipality_code, tiff_root=tiffpath)
                saved[f"k{num_periods}"] = (
                    incomplete_dir / f"{stem}_heatmap{suffix}.png"
                )

    return saved


def run_talhao_evaluations(
    ctx: ModelContext,
    *,
    datapath: Path,
    tiffpath: Path,
    harvest_years: list[int],
    talhoes_path: Path | None = None,
    baseline_csv: Path | None = None,
    pred_dir: Path | None = None,
) -> dict[int, Path]:
    pred_dir = pred_dir or predictions_dir(ctx.run_dir, create=True)
    talhoes_path = talhoes_path or _default_talhoes_path()
    baseline_csv = baseline_csv or DEFAULT_BASELINE_CSV

    outputs: dict[int, Path] = {}
    for harvest_year in harvest_years:
        year_range = harvest_to_year_range(harvest_year)
        tag = year_range.replace("-", "_")
        output_csv = pred_dir / f"talhoes_forecasts_{tag}.csv"
        print(f"\nTalhão evaluation for harvest {harvest_year} ({year_range})...")
        run_talhao_predictions(
            checkpoint=ctx.checkpoint,
            datapath=datapath,
            tiffpath=tiffpath,
            year_range=year_range,
            talhoes_path=talhoes_path,
            baseline_csv=baseline_csv,
            output_csv=output_csv,
            sequencelength=ctx.sequencelength,
            chunk_size=ctx.chunk_size,
            device=ctx.device,
            seed=ctx.seed,
            feature_layout=ctx.feature_layout,
            rc=ctx.rc,
            interp=ctx.interp,
            model=ctx.model,
            aggregation=ctx.aggregation,
            run_config=ctx.run_config,
        )
        outputs[harvest_year] = output_csv
    return outputs


def run_municipal_error_heatmap(
    ctx: ModelContext,
    *,
    forecasts_csv: Path,
    holdout_year: int,
    state: str = "PR",
    output_path: Path | None = None,
    dpi: int = 150,
) -> Path:
    if output_path is None:
        output_path = (
            municipal_heatmaps_dir(ctx.run_dir, create=True)
            / f"map_pr_error_{holdout_year}.png"
        )

    try:
        from geobr import read_municipality
    except ImportError as exc:
        raise ImportError(
            "Install geobr and geopandas for municipal maps: "
            "pip install geobr geopandas"
        ) from exc

    df = pd.read_csv(forecasts_csv)
    if "error" not in df.columns and {"forecast", "actual_yield"}.issubset(df.columns):
        df["error"] = df["forecast"] - df["actual_yield"]

    state_arg = state.strip().upper()
    code_state = int(state_arg) if state_arg.isdigit() else state_arg
    mun = read_municipality(code_muni=code_state, year=2020, simplified=True)
    mun = mun.copy()
    mun["muni_key"] = mun["code_muni"].astype(int).astype(str).str.zfill(7)

    plot_choropleth_column(
        df,
        mun,
        "error",
        output_path,
        state=state,
        csv_name=forecasts_csv.name,
        title=f"Paraná — erro de previsão (t) vs produção {holdout_year}",
        dpi=dpi,
    )
    return output_path


def _default_talhoes_path() -> Path:
    gpkg = REPO_ROOT / "benchmark" / "talhoes_baseline.gpkg"
    geojson = REPO_ROOT / "benchmark" / "talhoes_baseline.geojson"
    return gpkg if gpkg.is_file() else geojson


def run_productivity_scatter_plots(
    ctx: ModelContext,
    *,
    yield_forecasts_by_year: dict[int, Path],
    yield_csv: Path,
    dpi: int = 150,
) -> dict[int, dict[str, Path]]:
    """Predicted vs IBGE productivity scatter plots for each harvest year with forecasts."""
    if not yield_forecasts_by_year:
        return {}

    print(f"\n{'=' * 60}")
    print("Productivity scatter plots (predicted vs IBGE yield_t_ha)")
    print(f"{'=' * 60}")

    all_saved: dict[int, dict[str, Path]] = {}
    for harvest_year in sorted(yield_forecasts_by_year):
        forecasts_csv = Path(yield_forecasts_by_year[harvest_year])
        if not forecasts_csv.is_file():
            print(f"  Skipping {harvest_year}: missing {forecasts_csv}")
            continue
        print(f"\nHarvest {harvest_year}:")
        all_saved[harvest_year] = run_productivity_scatter_for_forecasts(
            forecasts_csv,
            yield_csv=yield_csv,
            target=ctx.target,
            harvest_year=harvest_year,
            dpi=dpi,
        )
    return all_saved


def generate_results(
    checkpoint: Path | str,
    *,
    datapath: Path | str | None = None,
    yield_csv: Path | str | None = None,
    tiffpath: Path | str | None = None,
    holdout_year: int = 2021,
    harvest_years: list[int] | None = None,
    guarapuava_code: str = DEFAULT_GUARAPUAVA_CODE,
    talhoes_path: Path | str | None = None,
    baseline_csv: Path | str | None = None,
    sequencelength: int | None = None,
    seed: int | None = None,
    feature_layout: str | None = None,
    device: str | None = None,
    chunk_size: int | None = None,
    reference_date: date | str | None = None,
    rc: bool | None = None,
    interp: bool | None = None,
    skip_incomplete_series: bool = False,
    skip_guarapuava: bool = False,
    skip_talhoes: bool = False,
    skip_municipal_map: bool = False,
    skip_productivity_scatter: bool = False,
    map_dpi: int = 150,
    verbose: bool = True,
    min_coverage_ratio: float | None = DEFAULT_MIN_COVERAGE_RATIO,
    max_coverage_ratio: float | None = DEFAULT_MAX_COVERAGE_RATIO,
) -> GenerateResultsOutput:
    """
    Load one checkpoint and produce all post-training result artifacts.

    Returns paths to every written file. Municipal forecasts for the full series
    (k=6) are saved under predictions/yield_forecasts_{year}.csv — equivalent to
    running predict_yield.py per harvest year.
    """
    from env_config import resolve_datapath

    yield_csv_path = Path(yield_csv) if yield_csv else YIELD_CSV
    datapath_resolved = resolve_datapath(datapath)
    tiffpath_resolved = Path(tiffpath) if tiffpath else DEFAULT_TIFFPATH

    ctx = load_model_context(
        checkpoint,
        datapath=datapath_resolved,
        yield_csv=yield_csv_path,
        sequencelength=sequencelength,
        seed=seed,
        feature_layout=feature_layout,
        device=device,
        chunk_size=chunk_size,
        reference_date=reference_date,
        rc=rc,
        interp=interp,
    )

    years = resolve_harvest_years(ctx, harvest_years, yield_csv_path)
    leave_out_years = resolve_leave_out_years(ctx, holdout_year)
    if verbose:
        print(f"Checkpoint: {ctx.checkpoint}")
        print(f"Run dir:    {ctx.run_dir}")
        print(f"Harvest years: {years}")
        print(f"Holdout year:  {holdout_year}")
        print(f"Leave-out years: {leave_out_years}")
        print(f"Pixel chunk size: {ctx.chunk_size}")

    output = GenerateResultsOutput()

    if not skip_incomplete_series:
        (
            output.incomplete_series_summary,
            output.incomplete_series_by_leave_out_year,
            output.yield_forecasts_by_year,
            output.incomplete_forecasts,
        ) = run_incomplete_series_evaluation(
            ctx,
            datapath=datapath_resolved,
            yield_csv=yield_csv_path,
            harvest_years=years,
            leave_out_years=leave_out_years,
            min_coverage_ratio=min_coverage_ratio,
            max_coverage_ratio=max_coverage_ratio,
        )

    holdout_csv = output.yield_forecasts_by_year.get(holdout_year)
    if holdout_csv is None and not skip_municipal_map:
        candidate = predictions_dir(ctx.run_dir) / f"yield_forecasts_{holdout_year}.csv"
        if candidate.is_file():
            holdout_csv = candidate

    if not skip_guarapuava:
        output.guarapuava_heatmaps = run_guarapuava_heatmaps(
            ctx,
            datapath=datapath_resolved,
            tiffpath=tiffpath_resolved,
            holdout_year=holdout_year,
            municipality_code=guarapuava_code,
        )

    if not skip_talhoes:
        output.talhao_forecasts = run_talhao_evaluations(
            ctx,
            datapath=datapath_resolved,
            tiffpath=tiffpath_resolved,
            harvest_years=years,
            talhoes_path=Path(talhoes_path) if talhoes_path else None,
            baseline_csv=Path(baseline_csv) if baseline_csv else None,
        )

    if not skip_municipal_map:
        if holdout_csv is None or not Path(holdout_csv).is_file():
            raise FileNotFoundError(
                f"No forecasts CSV for holdout year {holdout_year}. "
                "Run incomplete-series step first or place "
                f"predictions/yield_forecasts_{holdout_year}.csv in the run folder."
            )
        output.municipal_error_map = run_municipal_error_heatmap(
            ctx,
            forecasts_csv=Path(holdout_csv),
            holdout_year=holdout_year,
            dpi=map_dpi,
        )

    if not skip_productivity_scatter and output.yield_forecasts_by_year:
        output.productivity_scatter_plots = run_productivity_scatter_plots(
            ctx,
            yield_forecasts_by_year=output.yield_forecasts_by_year,
            yield_csv=yield_csv_path,
            dpi=map_dpi,
        )

    if verbose:
        print("\n" + "=" * 60)
        print("RESULTS PIPELINE COMPLETE")
        print("=" * 60)
        if output.incomplete_series_summary:
            print(f"  Incomplete summary: {output.incomplete_series_summary}")
        for year, path in sorted(output.incomplete_series_by_leave_out_year.items()):
            print(f"  Incomplete summary (leave-out {year}): {path}")
        for year, path in sorted(output.yield_forecasts_by_year.items()):
            print(f"  Yield forecasts {year}: {path}")
        if output.municipal_error_map:
            print(f"  Municipal error map: {output.municipal_error_map}")
        for year, variants in sorted(output.productivity_scatter_plots.items()):
            for variant_key, path in sorted(variants.items()):
                print(f"  Productivity scatter {year} ({variant_key}): {path}")
        for year, path in sorted(output.talhao_forecasts.items()):
            print(f"  Talhões {year}: {path}")
        if output.guarapuava_heatmaps:
            print(f"  Guarapuava heatmaps: {len(output.guarapuava_heatmaps)} files")

    return output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--datapath", type=Path, default=None)
    p.add_argument("--yield-csv", type=Path, default=None)
    p.add_argument("--tiffpath", type=Path, default=None)
    p.add_argument("--holdout-year", type=int, default=2021)
    p.add_argument(
        "--harvest-years",
        type=int,
        nargs="*",
        default=None,
        help="Harvest years to evaluate (default: from training config or yield CSV)",
    )
    p.add_argument(
        "--guarapuava-code",
        type=str,
        default=DEFAULT_GUARAPUAVA_CODE,
        help=f"IBGE code for intramunicipal heatmaps (default: {DEFAULT_GUARAPUAVA_CODE})",
    )
    p.add_argument("--talhoes", type=Path, default=None)
    p.add_argument("--baseline-csv", type=Path, default=None)
    p.add_argument("-seq", "--sequencelength", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--feature-layout", type=str, default=None)
    p.add_argument("-d", "--device", type=str, default=None)
    p.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Pixels per GPU forward pass (default: pixel_chunk_size * chunks_per_grad from training config)",
    )
    p.add_argument("--reference-date", type=str, default=None)
    p.add_argument("--rc", action="store_true", default=None)
    p.add_argument("--interp", action="store_true", default=None)
    p.add_argument("--map-dpi", type=int, default=150)
    p.add_argument("--skip-incomplete-series", action="store_true")
    p.add_argument("--skip-guarapuava", action="store_true")
    p.add_argument("--skip-talhoes", action="store_true")
    p.add_argument("--skip-municipal-map", action="store_true")
    p.add_argument("--skip-productivity-scatter", action="store_true")
    p.add_argument(
        "--min-coverage-ratio",
        type=float,
        default=DEFAULT_MIN_COVERAGE_RATIO,
        help=(
            "Keep rows with coverage_ratio >= this (default: no filter). "
            "Example: --min-coverage-ratio 0.8 --max-coverage-ratio 1.2"
        ),
    )
    p.add_argument(
        "--max-coverage-ratio",
        type=float,
        default=DEFAULT_MAX_COVERAGE_RATIO,
        help="Keep rows with coverage_ratio <= this (default: no filter)",
    )
    p.add_argument(
        "--no-coverage-filter",
        action="store_true",
        help="Do not filter yield CSV rows by coverage_ratio (default behavior)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    min_cov = None if args.no_coverage_filter else args.min_coverage_ratio
    max_cov = None if args.no_coverage_filter else args.max_coverage_ratio
    generate_results(
        args.checkpoint,
        datapath=args.datapath,
        yield_csv=args.yield_csv,
        tiffpath=args.tiffpath,
        holdout_year=args.holdout_year,
        harvest_years=args.harvest_years,
        guarapuava_code=args.guarapuava_code,
        talhoes_path=args.talhoes,
        baseline_csv=args.baseline_csv,
        sequencelength=args.sequencelength,
        seed=args.seed,
        feature_layout=args.feature_layout,
        device=args.device,
        chunk_size=args.chunk_size,
        reference_date=args.reference_date,
        rc=args.rc,
        interp=args.interp,
        skip_incomplete_series=args.skip_incomplete_series,
        skip_guarapuava=args.skip_guarapuava,
        skip_talhoes=args.skip_talhoes,
        skip_municipal_map=args.skip_municipal_map,
        skip_productivity_scatter=args.skip_productivity_scatter,
        map_dpi=args.map_dpi,
        min_coverage_ratio=min_cov,
        max_coverage_ratio=max_cov,
    )


if __name__ == "__main__":
    main()
