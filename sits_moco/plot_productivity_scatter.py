"""
Scatter plots: predicted municipal productivity vs IBGE (yield_t_ha).

How predicted productivity is derived from model forecasts depends on the training target:

  - productivity  → forecast is t/ha (z-score head denormalized with μ, σ)
  - total         → forecast / area_planted_ha (official IBGE area)
                    and forecast / pixel_area_ha (S2 footprint)
  - total_adj     → forecast / pixel_area_ha

Example::

    python plot_productivity_scatter.py \\
        --forecasts-csv results/.../predictions/yield_forecasts_2021.csv \\
        --yield-csv files/pam_soy_pr_2019_2025.csv \\
        --target total \\
        --harvest-year 2021
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_paths import (
    load_latest_run_config,
    productivity_scatter_dir,
    run_dir_from_forecasts_csv,
)
from utils_aggregated import regression_metrics, resolve_inference_target

IBGE_PRODUCTIVITY_COL = "yield_t_ha"
OFFICIAL_AREA_COL = "area_planted_ha"
PIXEL_AREA_COL = "pixel_area_ha"


@dataclass(frozen=True)
class ProductivityVariant:
    key: str
    area_column: str | None
    y_label: str


def productivity_variants_for_target(target: str) -> list[ProductivityVariant]:
    """Return one or more predicted-productivity definitions for a training target."""
    if target == "productivity":
        return [
            ProductivityVariant(
                key="direct",
                area_column=None,
                y_label="Predicted productivity (t/ha)",
            )
        ]
    if target == "total":
        return [
            ProductivityVariant(
                key="per_official_area",
                area_column=OFFICIAL_AREA_COL,
                y_label="Predicted / official IBGE area (t/ha)",
            ),
            ProductivityVariant(
                key="per_pixel_area",
                area_column=PIXEL_AREA_COL,
                y_label="Predicted / pixel area (t/ha)",
            ),
        ]
    if target == "total_adj":
        return [
            ProductivityVariant(
                key="per_pixel_area",
                area_column=PIXEL_AREA_COL,
                y_label="Predicted / pixel area (t/ha)",
            )
        ]
    raise ValueError(
        f"Unknown target {target!r}; expected productivity, total, or total_adj"
    )


def build_productivity_dataframe(
    forecasts_df: pd.DataFrame,
    yield_df: pd.DataFrame,
    *,
    harvest_year: int,
    variant: ProductivityVariant,
) -> pd.DataFrame:
    """Merge forecasts with IBGE labels and compute predicted / actual productivity."""
    required_forecast_cols = {"municipality_code", "forecast"}
    missing = required_forecast_cols - set(forecasts_df.columns)
    if missing:
        raise ValueError(
            f"Forecasts CSV missing columns {sorted(missing)}; "
            f"available: {list(forecasts_df.columns)}"
        )

    area_cols = [OFFICIAL_AREA_COL, PIXEL_AREA_COL]
    required_yield_cols = {"municipality_code", "year", IBGE_PRODUCTIVITY_COL, *area_cols}
    missing_yield = required_yield_cols - set(yield_df.columns)
    if missing_yield:
        raise ValueError(
            f"Yield CSV missing columns {sorted(missing_yield)}; "
            f"available: {list(yield_df.columns)}"
        )

    fc = forecasts_df.copy()
    fc["municipality_code"] = fc["municipality_code"].astype(str)

    ydf = yield_df.loc[yield_df["year"] == harvest_year].copy()
    ydf["municipality_code"] = ydf["municipality_code"].astype(str)
    merge_cols = ["municipality_code", IBGE_PRODUCTIVITY_COL, *area_cols]
    if "split" in ydf.columns:
        merge_cols.append("split")
    if "municipality_name" in ydf.columns:
        merge_cols.append("municipality_name")

    merged = fc.merge(
        ydf[merge_cols],
        on="municipality_code",
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError(
            f"No municipalities matched between forecasts and yield CSV for year {harvest_year}"
        )

    forecast = pd.to_numeric(merged["forecast"], errors="coerce")
    actual = pd.to_numeric(merged[IBGE_PRODUCTIVITY_COL], errors="coerce")

    if variant.area_column is None:
        predicted = forecast
    else:
        area = pd.to_numeric(merged[variant.area_column], errors="coerce")
        valid_area = area.notna() & (area > 0)
        predicted = forecast / area
        predicted = predicted.where(valid_area)

    out = merged.assign(
        harvest_year=harvest_year,
        predicted_productivity_t_ha=predicted,
        actual_productivity_t_ha=actual,
        productivity_error=predicted - actual,
    )
    valid = (
        out["predicted_productivity_t_ha"].notna()
        & out["actual_productivity_t_ha"].notna()
        & np.isfinite(out["predicted_productivity_t_ha"])
        & np.isfinite(out["actual_productivity_t_ha"])
    )
    return out.loc[valid].reset_index(drop=True)


def plot_productivity_scatter(
    df: pd.DataFrame,
    output_path: Path,
    *,
    title: str | None = None,
    x_label: str = "IBGE productivity (t/ha)",
    y_label: str = "Predicted productivity (t/ha)",
    dpi: int = 150,
) -> dict[str, float]:
    """Save a predicted-vs-actual productivity scatter plot; return regression metrics."""
    if df.empty:
        raise ValueError("Cannot plot productivity scatter from an empty dataframe")

    x = df["actual_productivity_t_ha"].to_numpy(dtype=float)
    y = df["predicted_productivity_t_ha"].to_numpy(dtype=float)
    metrics = regression_metrics(y, x, target_column=IBGE_PRODUCTIVITY_COL)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(x, y, alpha=0.65, edgecolors="white", linewidths=0.4, s=36)

    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = 0.05 * (hi - lo) if hi > lo else 0.5
    lim_lo, lim_hi = lo - pad, hi + pad
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=1.0, label="1:1")

    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(loc="upper left")

    mape_str = (
        f"{metrics['mape']:.1f}%"
        if metrics["mape"] is not None and np.isfinite(metrics["mape"])
        else "n/a"
    )
    stats = (
        f"n = {len(x)}\n"
        f"R² = {metrics['r2']:.3f}\n"
        f"RMSE = {metrics['rmse']:.2f} t/ha\n"
        f"MAE = {metrics['mae']:.2f} t/ha\n"
        f"MAPE = {mape_str}"
    )
    ax.text(
        0.98,
        0.02,
        stats,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85),
    )

    if title:
        ax.set_title(title, fontsize=11)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return metrics


def run_productivity_scatter_for_forecasts(
    forecasts_csv: Path,
    *,
    yield_csv: Path,
    target: str,
    harvest_year: int,
    output_dir: Path | None = None,
    dpi: int = 150,
) -> dict[str, Path]:
    """
    Build and save productivity scatter plot(s) for one forecasts CSV.

    Returns mapping variant_key → PNG path (and writes companion CSVs).
    """
    forecasts_csv = Path(forecasts_csv)
    yield_csv = Path(yield_csv)
    forecasts_df = pd.read_csv(forecasts_csv)
    yield_df = pd.read_csv(yield_csv)

    if output_dir is None:
        run_dir = run_dir_from_forecasts_csv(forecasts_csv)
        if run_dir is None:
            output_dir = forecasts_csv.parent / "productivity_scatter"
        else:
            output_dir = productivity_scatter_dir(run_dir, create=True)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}
    for variant in productivity_variants_for_target(target):
        df = build_productivity_dataframe(
            forecasts_df,
            yield_df,
            harvest_year=harvest_year,
            variant=variant,
        )
        csv_path = output_dir / f"productivity_{harvest_year}_{variant.key}.csv"
        df.to_csv(csv_path, index=False)

        title = f"Productivity — {harvest_year} — {variant.y_label}"
        png_path = output_dir / f"productivity_scatter_{harvest_year}_{variant.key}.png"
        plot_productivity_scatter(
            df,
            png_path,
            title=title,
            y_label=variant.y_label,
            dpi=dpi,
        )
        saved[variant.key] = png_path
        print(f"  Saved {variant.key}: {png_path} ({len(df)} municipalities)")
    return saved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--forecasts-csv", type=Path, required=True)
    p.add_argument("--yield-csv", type=Path, default=None)
    p.add_argument("--target", type=str, default=None, choices=("productivity", "total", "total_adj"))
    p.add_argument("--harvest-year", type=int, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    target = args.target
    yield_csv = args.yield_csv

    run_dir = run_dir_from_forecasts_csv(args.forecasts_csv)
    if run_dir is not None:
        run_config = load_latest_run_config(run_dir)
        if target is None:
            target, _, _, _ = resolve_inference_target(run_config)
        if yield_csv is None:
            cli_yield = (run_config.get("cli") or {}).get("yield_csv")
            yield_csv = Path(cli_yield) if cli_yield else Path("files/pam_soy_pr_2019_2025.csv")
    else:
        if target is None:
            raise ValueError("--target is required when forecasts CSV is outside a run folder")
        if yield_csv is None:
            yield_csv = Path("files/pam_soy_pr_2019_2025.csv")

    print(f"Target: {target}")
    print(f"Harvest year: {args.harvest_year}")
    run_productivity_scatter_for_forecasts(
        args.forecasts_csv,
        yield_csv=yield_csv,
        target=target,
        harvest_year=args.harvest_year,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
