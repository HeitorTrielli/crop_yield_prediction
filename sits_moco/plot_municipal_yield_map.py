#!/usr/bin/env python3
"""
Choropleth map: municipal forecasts vs labels on a Brazilian state (default Paraná).

Typical workflow (after training):
  1) Generate forecasts + merge ground truth with predict_yield.py (same checkpoint & sequencelength as training).
  2) Run this script on the CSV.

Requires: pip install geopandas matplotlib geobr

Example (Seed6003 model, eval split, Paraná-only map of forecast error in tons):

  python predict_yield.py ^
    --checkpoint results/Total_20_22_23_24_Seed6003/training/model_best.pth ^
    --datapath files/npy/2020-2024 ^
    --yield-csv files/pam_soy_pr_2019_2025.csv ^
    --eval-only ^
    --sequencelength 45 ^
    --output-csv presentation/pr_forecasts_seed6003_eval.csv

  python plot_municipal_yield_map.py ^
    --checkpoint results/.../training/model_best.pth ^
    --forecasts-csv results/.../predictions/yield_forecasts_eval.csv ^
    --state PR ^
    --column error

  # One CSV → four maps (previsão, observado, |erro| t, |erro| relativo à observada %%):
  python plot_municipal_yield_map.py ^
    --forecasts-csv presentation/pr_forecasts_seed6003_eval.csv ^
    --state PR ^
    --all-maps ^
    --output presentation/map_pr_2021_seed6003.png ^
    --title "Paraná — 2021 — Seed6003"

  # Side-by-side previsão | observado (escala de cor partilhada; default YlGnBu):
  python plot_municipal_yield_map.py ^
    --forecasts-csv presentation/forecasts.csv --state PR ^
    --composite-forecast-actual --output presentation/map_pr.png

Adjust --datapath and --sequencelength to match how you trained / where .npy files live.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from run_paths import (
    municipal_heatmaps_dir,
    run_dir_from_forecasts_csv,
    run_dir_from_path,
)

try:
    import geopandas as gpd
    from geobr import read_municipality
except ImportError as e:
    raise SystemExit(
        "Install dependencies: pip install geopandas geobr matplotlib pandas\n" + str(e)
    ) from e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot municipal yield forecasts / errors on a state choropleth (geobr boundaries)."
    )
    p.add_argument(
        "--forecasts-csv",
        type=Path,
        required=True,
        help="CSV from predict_yield.py (needs municipality_code + forecast; "
        "actual_yield & error columns recommended)",
    )
    p.add_argument(
        "--state",
        type=str,
        default="PR",
        help="IBGE state: two-letter code (e.g. PR) or numeric code (e.g. 41 for Paraná). Default: PR",
    )
    p.add_argument(
        "--mesh-year",
        type=int,
        default=2020,
        help="Year of municipal boundaries in geobr (default: 2020)",
    )
    p.add_argument(
        "--column",
        type=str,
        default="error",
        choices=(
            "error",
            "forecast",
            "actual_yield",
            "abs_error",
            "pct_error",
            "abs_pct_error",
            "rel_abs_err_pct",
        ),
        help="Column to paint municipalities (default: error = forecast - actual if present)",
    )
    p.add_argument(
        "--relative-error-floor-tons",
        type=float,
        default=0.0,
        help=(
            "For rel_abs_err_pct: use denominator max(actual_yield, this floor in tons). "
            "Set e.g. 500–1000 to avoid huge %% where observed production is very small (0 = no floor)."
        ),
    )
    p.add_argument(
        "--all-maps",
        action="store_true",
        help=(
            "Write four PNGs next to --output: *_forecast, *_actual_yield, *_abs_error (t), "
            "*_rel_abs_err_pct (|error|/max(observed, floor) as %%); needs forecast + actual_yield. "
            "Ignores --column. See --relative-error-floor-tons."
        ),
    )
    p.add_argument(
        "--composite-forecast-actual",
        action="store_true",
        help=(
            "Write one wide PNG: forecast | actual_yield side by side, shared color scale (tons). "
            "Uses --composite-cmap (not viridis). Requires forecast + actual_yield. "
            "If used without --all-maps, only this file is written."
        ),
    )
    p.add_argument(
        "--composite-cmap",
        type=str,
        default="YlOrRd",
        help=(
            "Matplotlib colormap for --composite-forecast-actual (default: YlOrRd). "
            "Alternatives: Reds, OrRd, YlGnBu, viridis."
        ),
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint path; with --output omitted, maps go to {run_dir}/figures/",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PNG path (stem used for --all-maps suffixes). "
            "Default: {run_dir}/figures/municipal_heatmaps/municipal_yield_map.png "
            "from --checkpoint or forecasts CSV under predictions/"
        ),
    )
    p.add_argument("--title", type=str, default=None, help="Figure title")
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PNG resolution (default: 150)",
    )
    p.add_argument(
        "--simplified",
        action="store_true",
        default=True,
        help="Use simplified municipality polygons (default: True)",
    )
    p.add_argument(
        "--no-simplified",
        action="store_false",
        dest="simplified",
        help="Use full-resolution polygons",
    )
    p.add_argument(
        "--compare-forecasts-csv",
        type=Path,
        default=None,
        help="Second model forecasts CSV (com chuva) for --report-yield-maps.",
    )
    p.add_argument(
        "--report-yield-maps",
        action="store_true",
        help=(
            "Write one PNG with three yield panels (sem/com chuva + PAM) and a "
            "shared colorbar below. Requires --compare-forecasts-csv."
        ),
    )
    p.add_argument(
        "--report-error-maps",
        action="store_true",
        help=(
            "Write one PNG (3×3) comparing sem/com chuva on abs error (t), "
            "relative error (%%) and symmetric sMAPE (%%), plus a delta column "
            "(com − sem); one colorbar per row. Requires --compare-forecasts-csv."
        ),
    )
    return p.parse_args()


def _normalize_muni_code(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    return s.str.zfill(7)


def _ensure_derived_columns(
    df: pd.DataFrame, *, relative_error_floor_tons: float = 0.0
) -> pd.DataFrame:
    """Add error / abs_error / pct_error / rel_abs_err_pct when forecast and actual_yield exist."""
    out = df.copy()
    if "forecast" in out.columns and "actual_yield" in out.columns:
        fc = pd.to_numeric(out["forecast"], errors="coerce")
        ac = pd.to_numeric(out["actual_yield"], errors="coerce")
        if "error" not in out.columns:
            out["error"] = fc - ac
        if "abs_error" not in out.columns:
            out["abs_error"] = (fc - ac).abs()
        acv = ac.to_numpy(dtype=float)
        fcv = fc.to_numpy(dtype=float)
        # Percent vs actual: 100 * (forecast - actual) / actual; NaN where |actual| ~ 0
        if "pct_error" not in out.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                pct = np.where(np.abs(acv) > 1e-9, 100.0 * (fcv - acv) / acv, np.nan)
            out["pct_error"] = pct
        if "abs_pct_error" not in out.columns:
            out["abs_pct_error"] = pd.to_numeric(
                out["pct_error"], errors="coerce"
            ).abs()
        # |forecast - actual| / max(|actual|, floor) as percent
        ft = max(float(relative_error_floor_tons), 1e-9)
        denom = np.maximum(np.abs(acv), ft)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_abs_pct = 100.0 * np.abs(fcv - acv) / denom
        out["rel_abs_err_pct"] = rel_abs_pct
        if "smape_pct" not in out.columns:
            sden = np.abs(fcv) + np.abs(acv)
            with np.errstate(divide="ignore", invalid="ignore"):
                smape = np.where(sden > 1e-9, 100.0 * np.abs(fcv - acv) / sden, np.nan)
            out["smape_pct"] = smape
    return out


def _cmap_and_limits(column: str, vmin: float, vmax: float) -> tuple[str, float, float]:
    if column in ("error", "pct_error"):
        cmap = "RdBu_r"
        if vmin < 0 < vmax:
            m = max(abs(vmin), abs(vmax))
            vmin, vmax = -m, m
        return cmap, vmin, vmax
    if column in (
        "forecast",
        "actual_yield",
        "abs_error",
        "abs_pct_error",
        "rel_abs_err_pct",
        "smape_pct",
    ):
        return "YlOrRd", vmin, vmax
    return "viridis", vmin, vmax


def _composite_output_path(base: Path) -> Path:
    base = Path(base)
    stem = base.stem
    suf = base.suffix if base.suffix else ".png"
    return base.parent / f"{stem}_forecast_vs_actual{suf}"


def plot_composite_forecast_actual(
    df: pd.DataFrame,
    mun: gpd.GeoDataFrame,
    output_path: Path,
    *,
    state: str,
    csv_name: str,
    title: str | None,
    dpi: int,
    cmap: str,
) -> None:
    """Side-by-side choropleths with shared vmin/vmax and a single colorbar."""
    if "forecast" not in df.columns or "actual_yield" not in df.columns:
        raise SystemExit(
            "Composite map requires columns 'forecast' and 'actual_yield' in the CSV."
        )
    plot_df = df.copy()
    plot_df["muni_key"] = _normalize_muni_code(plot_df["municipality_code"])
    fc = pd.to_numeric(plot_df["forecast"], errors="coerce")
    ac = pd.to_numeric(plot_df["actual_yield"], errors="coerce")
    df_plot = plot_df.assign(
        _fc=fc,
        _ac=ac,
    )[
        ["muni_key", "_fc", "_ac"]
    ].drop_duplicates(subset=["muni_key"])

    gdf = mun.merge(df_plot, on="muni_key", how="inner")
    if len(gdf) == 0:
        raise SystemExit(
            "No municipalities matched between CSV and geobr for composite map."
        )

    fc_vals = np.asarray(gdf["_fc"], dtype=float)
    ac_vals = np.asarray(gdf["_ac"], dtype=float)
    pool = np.concatenate([fc_vals, ac_vals])
    pool = pool[np.isfinite(pool)]
    if pool.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.nanmin(pool))
        vmax = float(np.nanmax(pool))
    if vmin >= vmax:
        vmax = vmin + 1.0

    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # Typography: neutral sans stack + slightly larger hierarchy than matplotlib defaults
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Segoe UI",
            "Helvetica Neue",
            "Arial",
            "DejaVu Sans",
            "sans-serif",
        ],
        "axes.titleweight": "600",
        "axes.titlepad": 10,
        "figure.titleweight": "650",
    }

    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(18, 8.2),
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.14, wspace=0.06)

        for ax, col, subt in zip(
            axes,
            ("_fc", "_ac"),
            ("Previsão", "Observado"),
            strict=True,
        ):
            gdf.plot(
                column=col,
                ax=ax,
                cmap=cmap,
                legend=False,
                vmin=vmin,
                vmax=vmax,
                edgecolor="#2d2d2d",
                linewidth=0.12,
                missing_kwds={"color": "#d9d9d9"},
            )
            ax.set_axis_off()
            ax.set_title(subt, fontsize=14, pad=10)

        supt = title if title else f"Previsão vs observado ({state}) — {csv_name}"
        fig.suptitle(supt, fontsize=16, y=0.96)

        # One shared colorbar for both panels (no duplicated scales)
        cbar = fig.colorbar(
            sm,
            ax=list(axes),
            orientation="horizontal",
            fraction=0.065,
            pad=0.02,
            aspect=42,
            shrink=0.92,
        )
        cbar.set_label("Produção (t)", fontsize=12, labelpad=8)
        cbar.ax.tick_params(labelsize=10)
        # Readable thousands for large tonnage (e.g. 350000 → 350 000)
        tons_fmt = mticker.FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", " "))
        cbar.ax.xaxis.set_major_formatter(tons_fmt)

    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved composite map: {out}")
    print(
        f"  cmap={cmap!r}, shared scale [{vmin:.1f}, {vmax:.1f}] t, {len(gdf)} municipalities"
    )


def _gdf_for_column(
    df: pd.DataFrame, mun: gpd.GeoDataFrame, column: str
) -> gpd.GeoDataFrame:
    plot_df = df.copy()
    plot_df["muni_key"] = _normalize_muni_code(plot_df["municipality_code"])
    values = pd.to_numeric(plot_df[column], errors="coerce")
    df_plot = plot_df.assign(_plot_value=values)[
        ["muni_key", "_plot_value"]
    ].drop_duplicates(subset=["muni_key"])
    gdf = mun.merge(df_plot, on="muni_key", how="inner")
    if len(gdf) == 0:
        raise SystemExit(
            f"No municipalities matched for column '{column}'. "
            "Check municipality_code uses IBGE 7-digit codes for this state."
        )
    return gdf


def plot_composite_yield_trio(
    df_sem: pd.DataFrame,
    df_com: pd.DataFrame,
    mun: gpd.GeoDataFrame,
    output_path: Path,
    *,
    title: str | None,
    dpi: int,
    cmap: str = "YlOrRd",
) -> None:
    """Three-panel yield comparison with one shared horizontal colorbar below."""
    vmin, vmax = _yield_scale_from_dfs([df_sem, df_com])
    panels: list[tuple[pd.DataFrame, str, str]] = [
        (df_sem, "forecast", "Sem dados de chuva"),
        (df_com, "forecast", "Com dados de chuva"),
        (df_sem, "actual_yield", "Observado (PAM)"),
    ]
    gdfs = [_gdf_for_column(frame, mun, col) for frame, col, _ in panels]

    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Segoe UI",
            "Helvetica Neue",
            "Arial",
            "DejaVu Sans",
            "sans-serif",
        ],
        "axes.titleweight": "600",
        "axes.titlepad": 10,
        "figure.titleweight": "650",
    }

    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 3, figsize=(20, 7.5), constrained_layout=False)
        fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.16, wspace=0.05)

        for ax, gdf, (_, _, subt) in zip(axes, gdfs, panels, strict=True):
            gdf.plot(
                column="_plot_value",
                ax=ax,
                cmap=cmap,
                legend=False,
                vmin=vmin,
                vmax=vmax,
                edgecolor="#2d2d2d",
                linewidth=0.12,
                missing_kwds={"color": "#d9d9d9"},
            )
            ax.set_axis_off()
            ax.set_title(subt, fontsize=13, pad=8)

        if title:
            fig.suptitle(title, fontsize=15, y=0.98)

        cbar = fig.colorbar(
            sm,
            ax=list(axes),
            orientation="horizontal",
            fraction=0.07,
            pad=0.06,
            aspect=50,
            shrink=0.95,
        )
        cbar.set_label("Produção municipal (t)", fontsize=12, labelpad=8)
        cbar.ax.tick_params(labelsize=10)
        tons_fmt = mticker.FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", " "))
        cbar.ax.xaxis.set_major_formatter(tons_fmt)

    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved composite yield trio: {out}")
    print(
        f"  cmap={cmap!r}, shared scale [{vmin:.1f}, {vmax:.1f}] t, "
        f"{len(gdfs[0])} municipalities"
    )


def _scale_from_dfs_column(dfs: list[pd.DataFrame], column: str) -> tuple[float, float]:
    return _yield_scale_from_dfs(dfs, columns=(column,))


def _error_delta_frame(
    df_sem: pd.DataFrame, df_com: pd.DataFrame, column: str
) -> pd.DataFrame:
    """Per-municipality delta (com − sem) for one error metric."""
    key = "municipality_code"
    merged = df_sem[[key, column]].merge(
        df_com[[key, column]], on=key, suffixes=("_sem", "_com")
    )
    delta = merged[key].to_frame()
    delta[column] = merged[f"{column}_com"] - merged[f"{column}_sem"]
    return delta


def _symmetric_scale(vals: np.ndarray) -> tuple[float, float]:
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return -1.0, 1.0
    m = float(np.max(np.abs(finite)))
    if m <= 0:
        m = 1.0
    return -m, m


def plot_composite_error_grid(
    df_sem: pd.DataFrame,
    df_com: pd.DataFrame,
    mun: gpd.GeoDataFrame,
    output_path: Path,
    *,
    title: str | None,
    dpi: int,
    relative_error_floor_tons: float,
    cmap: str = "YlOrRd",
) -> None:
    """3×3 error comparison: abs (t), relative %, sMAPE % — sem, com, delta (com − sem)."""
    df_sem = _ensure_derived_columns(
        df_sem, relative_error_floor_tons=relative_error_floor_tons
    )
    df_com = _ensure_derived_columns(
        df_com, relative_error_floor_tons=relative_error_floor_tons
    )
    panels: list[tuple[str, str, str, str]] = [
        ("abs_error", "Erro absoluto", "Erro absoluto (t)", "Δ erro absoluto (t)"),
        (
            "abs_pct_error",
            "Erro relativo",
            "|erro| / observado (%)",
            "Δ erro relativo (p.p.)",
        ),
        ("smape_pct", "sMAPE (simétrico)", "sMAPE (%)", "Δ sMAPE (p.p.)"),
    ]

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Segoe UI",
            "Helvetica Neue",
            "Arial",
            "DejaVu Sans",
            "sans-serif",
        ],
        "axes.titleweight": "600",
        "axes.titlepad": 8,
        "figure.titleweight": "650",
    }

    with plt.rc_context(rc):
        fig = plt.figure(figsize=(14.2, 12.5))
        outer = gridspec.GridSpec(
            3,
            1,
            figure=fig,
            left=0.03,
            right=0.97,
            top=0.90,
            bottom=0.04,
            hspace=0.20,
        )
        if title:
            fig.suptitle(title, fontsize=14, y=0.97)
        fig.text(
            0.18,
            0.915,
            "Sem dados de chuva",
            ha="center",
            fontsize=11,
            fontweight="600",
        )
        fig.text(
            0.50,
            0.915,
            "Com dados de chuva",
            ha="center",
            fontsize=11,
            fontweight="600",
        )
        fig.text(
            0.82,
            0.915,
            "Δ (com − sem)",
            ha="center",
            fontsize=11,
            fontweight="600",
        )

        tons_fmt = mticker.FuncFormatter(
            lambda x, _: f"{x:,.0f}".replace(",", "\u202f")
        )
        pct_fmt = mticker.FuncFormatter(lambda x, _: f"{x:.0f}")

        for row, (column, row_label, cbar_label, delta_cbar_label) in enumerate(panels):
            inner = gridspec.GridSpecFromSubplotSpec(
                2,
                3,
                subplot_spec=outer[row],
                height_ratios=[1.0, 0.07],
                width_ratios=[1.0, 1.0, 1.0],
                hspace=0.06,
                wspace=0.04,
            )
            vmin, vmax = _scale_from_dfs_column([df_sem, df_com], column)
            norm = Normalize(vmin=vmin, vmax=vmax)
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])

            df_delta = _error_delta_frame(df_sem, df_com, column)
            delta_vals = pd.to_numeric(df_delta[column], errors="coerce").to_numpy(
                dtype=float
            )
            dmin, dmax = _symmetric_scale(delta_vals)
            delta_norm = Normalize(vmin=dmin, vmax=dmax)
            delta_sm = ScalarMappable(norm=delta_norm, cmap="RdBu_r")
            delta_sm.set_array([])

            row_frames = (df_sem, df_com, df_delta)
            row_vmins = (vmin, vmin, dmin)
            row_vmaxs = (vmax, vmax, dmax)
            row_cmaps = (cmap, cmap, "RdBu_r")

            for col_idx, frame in enumerate(row_frames):
                ax = fig.add_subplot(inner[0, col_idx])
                gdf = _gdf_for_column(frame, mun, column)
                gdf.plot(
                    column="_plot_value",
                    ax=ax,
                    cmap=row_cmaps[col_idx],
                    legend=False,
                    vmin=row_vmins[col_idx],
                    vmax=row_vmaxs[col_idx],
                    edgecolor="#2d2d2d",
                    linewidth=0.10,
                    missing_kwds={"color": "#d9d9d9"},
                )
                ax.set_axis_off()
                if col_idx == 0:
                    ax.text(
                        0.02,
                        0.98,
                        row_label,
                        transform=ax.transAxes,
                        fontsize=10,
                        fontweight="600",
                        va="top",
                        ha="left",
                        bbox=dict(
                            boxstyle="round,pad=0.2", facecolor="white", alpha=0.88
                        ),
                    )

            cax_sem_com = fig.add_subplot(inner[1, 0:2])
            cbar = fig.colorbar(sm, cax=cax_sem_com, orientation="horizontal")
            cax_sem_com.set_xlabel(cbar_label, fontsize=9, labelpad=2)
            cax_sem_com.tick_params(labelsize=8, pad=1)
            fmt = tons_fmt if column == "abs_error" else pct_fmt
            cax_sem_com.xaxis.set_major_formatter(fmt)

            cax_delta = fig.add_subplot(inner[1, 2])
            delta_cbar = fig.colorbar(delta_sm, cax=cax_delta, orientation="horizontal")
            cax_delta.set_xlabel(delta_cbar_label, fontsize=9, labelpad=2)
            cax_delta.tick_params(labelsize=8, pad=1)
            cax_delta.xaxis.set_major_formatter(fmt)

    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved composite error grid: {out}")
    print(
        f"  metrics: abs_error, abs_pct_error, smape_pct + delta (com-sem); "
        f"cmap={cmap!r}, delta=RdBu_r"
    )


def _yield_scale_from_dfs(
    dfs: list[pd.DataFrame], columns: tuple[str, ...] = ("forecast", "actual_yield")
) -> tuple[float, float]:
    """Shared vmin/vmax (tons) for comparable production choropleths."""
    chunks: list[np.ndarray] = []
    for frame in dfs:
        for col in columns:
            if col not in frame.columns:
                continue
            vals = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                chunks.append(vals)
    if not chunks:
        return 0.0, 1.0
    pool = np.concatenate(chunks)
    vmin = float(np.min(pool))
    vmax = float(np.max(pool))
    if vmin >= vmax:
        vmax = vmin + 1.0
    return vmin, vmax


def _bundle_output_paths(base: Path) -> tuple[Path, Path, Path, Path]:
    base = Path(base)
    stem = base.stem
    suf = base.suffix if base.suffix else ".png"
    parent = base.parent
    return (
        parent / f"{stem}_forecast{suf}",
        parent / f"{stem}_actual_yield{suf}",
        parent / f"{stem}_abs_error{suf}",
        parent / f"{stem}_rel_abs_err_pct{suf}",
    )


def plot_choropleth_column(
    df: pd.DataFrame,
    mun: gpd.GeoDataFrame,
    column: str,
    output_path: Path,
    *,
    state: str,
    csv_name: str,
    title: str | None,
    dpi: int,
    legend_label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
) -> int:
    """Merge CSV values on municipalities and save one PNG. Returns number of painted polygons."""
    if column not in df.columns:
        raise SystemExit(f"Column '{column}' not in CSV. Available: {list(df.columns)}")

    plot_df = df.copy()
    plot_df["muni_key"] = _normalize_muni_code(plot_df["municipality_code"])
    values = pd.to_numeric(plot_df[column], errors="coerce")
    df_plot = plot_df.assign(_plot_value=values)[
        ["muni_key", "_plot_value"]
    ].drop_duplicates(subset=["muni_key"])

    gdf = mun.merge(df_plot, on="muni_key", how="inner")
    if len(gdf) == 0:
        raise SystemExit(
            "No municipalities matched between CSV and geobr. "
            "Check that municipality_code uses IBGE 7-digit codes for this state."
        )

    missing = len(df_plot) - len(gdf)
    if missing > 0:
        print(
            f"  Note: {missing} CSV row(s) had no matching polygon (wrong state or code)."
        )

    col = "_plot_value"
    arr = gdf[col].to_numpy(dtype=float)
    finite = np.isfinite(arr)
    if vmin is None or vmax is None:
        if finite.any():
            vmin = float(np.nanmin(arr)) if vmin is None else vmin
            vmax = float(np.nanmax(arr)) if vmax is None else vmax
        else:
            vmin = 0.0 if vmin is None else vmin
            vmax = 1.0 if vmax is None else vmax
    if cmap is None:
        cmap, vmin, vmax = _cmap_and_limits(column, vmin, vmax)
    elif column in ("error", "pct_error") and vmin < 0 < vmax:
        m = max(abs(vmin), abs(vmax))
        vmin, vmax = -m, m
    if vmin >= vmax:
        vmax = vmin + 1.0

    fig, ax = plt.subplots(figsize=(11, 11))
    leg = legend_label if legend_label is not None else column
    gdf.plot(
        column=col,
        cmap=cmap,
        legend=True,
        legend_kwds={"label": leg},
        ax=ax,
        edgecolor="#333333",
        linewidth=0.2,
        vmin=vmin,
        vmax=vmax,
        missing_kwds={"color": "lightgrey"},
    )
    ax.set_axis_off()
    ax.set_title(title or f"Municipal {column} ({state}) — {csv_name}", fontsize=13)

    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved map: {out}")
    print(f"  Painted {len(gdf)} municipalities; column={column}")
    return len(gdf)


def main() -> None:
    args = parse_args()
    csv_path = args.forecasts_csv.resolve()
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    if args.output is None:
        run_dir = None
        if args.checkpoint is not None:
            run_dir = run_dir_from_path(args.checkpoint)
        else:
            run_dir = run_dir_from_forecasts_csv(csv_path)
        if run_dir is not None:
            args.output = (
                municipal_heatmaps_dir(run_dir, create=True) / "municipal_yield_map.png"
            )
        else:
            args.output = Path("municipal_yield_map.png")
    else:
        args.output = Path(args.output)

    df = pd.read_csv(csv_path)
    if "municipality_code" not in df.columns:
        raise SystemExit("CSV must contain column 'municipality_code'")

    if args.all_maps:
        df = _ensure_derived_columns(
            df, relative_error_floor_tons=args.relative_error_floor_tons
        )
        if "forecast" not in df.columns or "actual_yield" not in df.columns:
            raise SystemExit(
                "--all-maps requires columns 'forecast' and 'actual_yield' in the CSV."
            )
        if "abs_error" not in df.columns:
            raise SystemExit(
                "Could not derive abs_error (need forecast and actual_yield)."
            )
        if "rel_abs_err_pct" not in df.columns:
            raise SystemExit(
                "Could not derive rel_abs_err_pct (need forecast and actual_yield)."
            )
        print(
            "--all-maps: ignoring --column; writing "
            "forecast, actual_yield, abs_error (t), rel_abs_err_pct "
            f"(|error|/max(observed, {args.relative_error_floor_tons:g}) as %)."
        )
    else:
        # Derive error columns if missing but needed for selected column
        if args.column == "error" and "error" not in df.columns:
            df = _ensure_derived_columns(
                df, relative_error_floor_tons=args.relative_error_floor_tons
            )
            if "error" not in df.columns:
                raise SystemExit(
                    "Column 'error' missing and cannot derive it (need forecast + actual_yield)."
                )

        if args.column == "abs_error" and "abs_error" not in df.columns:
            df = _ensure_derived_columns(
                df, relative_error_floor_tons=args.relative_error_floor_tons
            )
            if "abs_error" not in df.columns:
                raise SystemExit("Cannot compute abs_error from this CSV.")

        if (
            args.column in ("pct_error", "abs_pct_error")
            and args.column not in df.columns
        ):
            df = _ensure_derived_columns(
                df, relative_error_floor_tons=args.relative_error_floor_tons
            )
            if args.column not in df.columns:
                raise SystemExit(
                    f"Cannot compute {args.column} from this CSV (need forecast + actual_yield)."
                )

        if args.column == "rel_abs_err_pct" and args.column not in df.columns:
            df = _ensure_derived_columns(
                df, relative_error_floor_tons=args.relative_error_floor_tons
            )
            if args.column not in df.columns:
                raise SystemExit(
                    "Cannot compute rel_abs_err_pct from this CSV (need forecast + actual_yield)."
                )

        if args.column not in df.columns:
            raise SystemExit(
                f"Column '{args.column}' not in CSV. Available: {list(df.columns)}"
            )

    df = _ensure_derived_columns(
        df, relative_error_floor_tons=args.relative_error_floor_tons
    )

    state_arg = args.state.strip().upper()
    code_state = int(state_arg) if state_arg.isdigit() else state_arg

    print(f"Loading municipalities for state={code_state}, year={args.mesh_year}...")
    mun: gpd.GeoDataFrame = read_municipality(
        code_muni=code_state,
        year=args.mesh_year,
        simplified=args.simplified,
    )
    if mun is None or len(mun) == 0:
        raise SystemExit("geobr returned no municipalities for this state/year.")

    mun = mun.copy()
    mun["muni_key"] = mun["code_muni"].astype(int).astype(str).str.zfill(7)

    if args.report_yield_maps or args.report_error_maps:
        if args.compare_forecasts_csv is None:
            need = (
                "--report-yield-maps"
                if args.report_yield_maps
                else "--report-error-maps"
            )
            raise SystemExit(f"{need} requires --compare-forecasts-csv")
        csv_com = Path(args.compare_forecasts_csv).resolve()
        if not csv_com.is_file():
            raise SystemExit(f"Compare CSV not found: {csv_com}")
        df_com = pd.read_csv(csv_com)
        if "municipality_code" not in df_com.columns:
            raise SystemExit("Compare CSV must contain column 'municipality_code'")
        for frame in (df, df_com):
            if "forecast" not in frame.columns or "actual_yield" not in frame.columns:
                raise SystemExit(
                    "Report maps require forecast and actual_yield in both CSVs."
                )
        title_base = args.title or "Paraná — safra 2021"
        if args.report_yield_maps:
            plot_composite_yield_trio(
                df,
                df_com,
                mun,
                args.output,
                title=title_base,
                dpi=args.dpi,
                cmap="YlOrRd",
            )
        if args.report_error_maps:
            plot_composite_error_grid(
                df,
                df_com,
                mun,
                args.output,
                title=title_base,
                dpi=args.dpi,
                relative_error_floor_tons=max(
                    float(args.relative_error_floor_tons), 1e-9
                ),
                cmap="YlOrRd",
            )
        return

    if args.composite_forecast_actual:
        plot_composite_forecast_actual(
            df,
            mun,
            _composite_output_path(args.output),
            state=args.state,
            csv_name=csv_path.name,
            title=args.title,
            dpi=args.dpi,
            cmap=args.composite_cmap,
        )
        if not args.all_maps:
            return

    if args.all_maps:
        p_forecast, p_actual, p_abs, p_pct = _bundle_output_paths(args.output)
        yield_vmin, yield_vmax = _yield_scale_from_dfs([df])
        yield_cmap = "YlOrRd"
        bundle = (
            (
                "forecast",
                p_forecast,
                (
                    f"{args.title} — previsão (t)"
                    if args.title
                    else f"Municipal forecast ({args.state}) — {csv_path.name}"
                ),
                "Produção (t)",
                yield_vmin,
                yield_vmax,
                yield_cmap,
            ),
            (
                "actual_yield",
                p_actual,
                (
                    f"{args.title} — produção observada (t)"
                    if args.title
                    else f"Municipal actual_yield ({args.state}) — {csv_path.name}"
                ),
                "Produção (t)",
                yield_vmin,
                yield_vmax,
                yield_cmap,
            ),
            (
                "abs_error",
                p_abs,
                (
                    f"{args.title} — |previsão − observado| (t)"
                    if args.title
                    else f"Municipal abs_error ({args.state}) — {csv_path.name}"
                ),
                None,
                None,
                None,
                None,
            ),
            (
                "rel_abs_err_pct",
                p_pct,
                (
                    f"{args.title} — |erro| relativo à observada (%)"
                    if args.title
                    else f"Municipal rel_abs_err_pct ({args.state}) — {csv_path.name}"
                ),
                "|forecast−actual| / max(observed, floor) (%)",
                None,
                None,
                None,
            ),
        )
        for col, outp, tit, leg, vmin, vmax, cmap in bundle:
            plot_choropleth_column(
                df,
                mun,
                col,
                outp,
                state=args.state,
                csv_name=csv_path.name,
                title=tit,
                dpi=args.dpi,
                legend_label=leg,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
        return

    legend_single = None
    if args.column == "pct_error":
        legend_single = "pct_error (% vs actual)"
    elif args.column == "abs_pct_error":
        legend_single = "abs_pct_error (% vs actual)"
    elif args.column == "rel_abs_err_pct":
        legend_single = f"|forecast−actual| / max(observed, {args.relative_error_floor_tons:g} t) (%)"

    plot_choropleth_column(
        df,
        mun,
        args.column,
        args.output,
        state=args.state,
        csv_name=csv_path.name,
        title=args.title,
        dpi=args.dpi,
        legend_label=legend_single,
    )


if __name__ == "__main__":
    main()
