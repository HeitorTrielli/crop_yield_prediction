#!/usr/bin/env python3
"""
Yearly municipal productivity histograms: IBGE (Paraná) vs best MoCo model.

Picks the MoCo yield run with the highest test R², loads its municipal
predictions, and draws:

  1. A 2-row grid (IBGE top, MoCo predictions bottom), one column per year
  2. An overlay variant (IBGE + predictions in the same axes)

Examples::

    python scripts/plot_yearly_productivity_histograms.py
    python scripts/plot_yearly_productivity_histograms.py --ibge-set matched
    python scripts/plot_yearly_productivity_histograms.py --run-dir results/.../spectral_xavier
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from run_paths import run_dir_from_path  # noqa: E402

DEFAULT_YIELD_CSV = _ROOT / "files" / "pam_soy_pr_2019_2025.csv"
DEFAULT_COVERAGE_CSV = _ROOT / "files" / "pam_soy_pr_2019_2025_coverage_0.8_1.2.csv"
DEFAULT_RESULTS = _ROOT / "results"
IBGE_COL = "yield_t_ha"
PRED_COL = "predicted_productivity_t_ha"
ACTUAL_COL = "actual_productivity_t_ha"
IBGE_COLOR = "#4c78a8"
PRED_COLOR = "#f58518"
BIN_WIDTH = 0.25
XMIN = 0.25
XMAX = 5.25


def _finite(series: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return vals[np.isfinite(vals)]


def find_best_moco_run(results_root: Path) -> tuple[Path, float]:
    """Return (run_dir, test_r2) for the MoCo yield checkpoint with max test R²."""
    best_dir: Path | None = None
    best_r2 = float("-inf")
    results_root = results_root.resolve()
    for testlog in results_root.rglob("testlog.csv"):
        try:
            rel_parts = testlog.resolve().relative_to(results_root).parts
        except ValueError:
            rel_parts = testlog.parts
        # Match MoCo in the experiment/layout folder names only — not the
        # repo path (this project lives under sits_moco/).
        names = [p.lower() for p in rel_parts]
        if not any("moco" in p for p in names):
            continue
        if any(p.startswith("p_moco") for p in names):
            continue
        try:
            df = pd.read_csv(testlog)
        except Exception:
            continue
        if df.empty or "r2" not in df.columns:
            continue
        r2 = pd.to_numeric(df.iloc[0]["r2"], errors="coerce")
        if pd.isna(r2):
            continue
        run_dir = run_dir_from_path(testlog)
        if float(r2) > best_r2:
            best_r2 = float(r2)
            best_dir = run_dir
    if best_dir is None:
        raise SystemExit(
            f"No MoCo yield run with testlog.csv found under {results_root}"
        )
    return best_dir, best_r2


def load_predictions_by_year(run_dir: Path) -> dict[int, np.ndarray]:
    scatter = run_dir / "figures" / "productivity_scatter"
    out: dict[int, np.ndarray] = {}
    if scatter.is_dir():
        for csv in sorted(scatter.glob("productivity_*_direct.csv")):
            year = int(csv.stem.split("_")[1])
            df = pd.read_csv(csv)
            if PRED_COL in df.columns:
                out[year] = _finite(df[PRED_COL])
            elif "forecast" in df.columns:
                out[year] = _finite(df["forecast"])
        if out:
            return out

    pred_dir = run_dir / "predictions"
    for csv in sorted(pred_dir.glob("yield_forecasts_*.csv")):
        year = int(csv.stem.rsplit("_", 1)[-1])
        df = pd.read_csv(csv)
        if "forecast" not in df.columns:
            continue
        out[year] = _finite(df["forecast"])
    if not out:
        raise SystemExit(
            f"No productivity predictions found under {run_dir} "
            "(looked for figures/productivity_scatter and predictions/yield_forecasts_*.csv)"
        )
    return out


def load_ibge_by_year(
    yield_csv: Path,
    years: list[int],
    *,
    municipality_filter: dict[int, set[str]] | None = None,
) -> dict[int, np.ndarray]:
    df = pd.read_csv(yield_csv)
    df["municipality_code"] = df["municipality_code"].astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    out: dict[int, np.ndarray] = {}
    for year in years:
        sub = df.loc[df["year"] == year]
        if municipality_filter is not None:
            keep = municipality_filter.get(year)
            if keep is not None:
                sub = sub.loc[sub["municipality_code"].isin(keep)]
        out[year] = _finite(sub[IBGE_COL])
    return out


def prediction_muni_filter(run_dir: Path) -> dict[int, set[str]]:
    scatter = run_dir / "figures" / "productivity_scatter"
    out: dict[int, set[str]] = {}
    if not scatter.is_dir():
        return out
    for csv in scatter.glob("productivity_*_direct.csv"):
        year = int(csv.stem.split("_")[1])
        df = pd.read_csv(csv)
        if "municipality_code" not in df.columns:
            continue
        out[year] = set(df["municipality_code"].astype(str))
    return out


def _annotate(ax, vals: np.ndarray) -> None:
    if vals.size == 0:
        return
    ax.axvline(float(np.median(vals)), color="0.15", ls="--", lw=1.0)
    ax.axvline(float(np.mean(vals)), color="0.40", ls=":", lw=1.0)


def _style_ax(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(XMIN, XMAX)


def _percent_weights(vals: np.ndarray) -> np.ndarray | None:
    """Each observation contributes 100/n so bar heights sum to 100."""
    if vals.size == 0:
        return None
    return np.full(vals.size, 100.0 / vals.size)


def plot_two_row(
    ibge: dict[int, np.ndarray],
    preds: dict[int, np.ndarray],
    *,
    years: list[int],
    bins: np.ndarray,
    out_path: Path,
    title: str,
    ibge_label: str,
    pred_label: str,
    percent: bool,
) -> None:
    n = len(years)
    fig, axes = plt.subplots(2, n, figsize=(3.15 * n, 6.4), sharex=True, squeeze=False)

    for col, year in enumerate(years):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        top_vals = ibge.get(year, np.array([], dtype=float))
        bot_vals = preds.get(year, np.array([], dtype=float))
        ax_top.hist(
            top_vals,
            bins=bins,
            color=IBGE_COLOR,
            edgecolor="white",
            linewidth=0.4,
            weights=_percent_weights(top_vals) if percent else None,
            alpha=0.92,
        )
        ax_bot.hist(
            bot_vals,
            bins=bins,
            color=PRED_COLOR,
            edgecolor="white",
            linewidth=0.4,
            weights=_percent_weights(bot_vals) if percent else None,
            alpha=0.92,
        )
        _annotate(ax_top, top_vals)
        _annotate(ax_bot, bot_vals)
        ax_top.set_title(str(year), fontsize=11)
        _style_ax(ax_top)
        _style_ax(ax_bot)
        ax_bot.set_xlabel("Productivity (t/ha)")
        if col == 0:
            suffix = " (%)" if percent else ""
            ax_top.set_ylabel(f"{ibge_label}{suffix}")
            ax_bot.set_ylabel(f"{pred_label}{suffix}")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_overlay(
    ibge: dict[int, np.ndarray],
    preds: dict[int, np.ndarray],
    *,
    years: list[int],
    bins: np.ndarray,
    out_path: Path,
    title: str,
    ibge_label: str,
    pred_label: str,
    percent: bool,
) -> None:
    n = len(years)
    ncols = 3 if n > 3 else n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows), sharex=True, squeeze=False
    )
    for i, year in enumerate(years):
        ax = axes[i // ncols, i % ncols]
        top_vals = ibge.get(year, np.array([], dtype=float))
        bot_vals = preds.get(year, np.array([], dtype=float))
        ax.hist(
            top_vals,
            bins=bins,
            color=IBGE_COLOR,
            edgecolor="white",
            linewidth=0.3,
            weights=_percent_weights(top_vals) if percent else None,
            alpha=0.65,
            label=ibge_label,
        )
        ax.hist(
            bot_vals,
            bins=bins,
            color=PRED_COLOR,
            edgecolor="white",
            linewidth=0.3,
            weights=_percent_weights(bot_vals) if percent else None,
            alpha=0.55,
            label=pred_label,
        )
        ax.set_title(str(year), fontsize=11)
        _style_ax(ax)
        ax.set_xlabel("Productivity (t/ha)")
        if percent and i % ncols == 0:
            ax.set_ylabel("%")
        if top_vals.size:
            ax.axvline(float(np.median(top_vals)), color=IBGE_COLOR, ls="--", lw=1.0)
        if bot_vals.size:
            ax.axvline(float(np.median(bot_vals)), color=PRED_COLOR, ls="--", lw=1.0)
        if i == 0:
            ax.legend(fontsize=8, frameon=False, loc="upper left")

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def summarize(
    ibge: dict[int, np.ndarray],
    preds: dict[int, np.ndarray],
    years: list[int],
) -> pd.DataFrame:
    rows = []
    for year in years:
        a = ibge.get(year, np.array([], dtype=float))
        p = preds.get(year, np.array([], dtype=float))
        rows.append(
            {
                "year": year,
                "ibge_n": int(a.size),
                "ibge_mean": float(np.mean(a)) if a.size else np.nan,
                "ibge_median": float(np.median(a)) if a.size else np.nan,
                "ibge_std": float(np.std(a, ddof=1)) if a.size > 1 else np.nan,
                "pred_n": int(p.size),
                "pred_mean": float(np.mean(p)) if p.size else np.nan,
                "pred_median": float(np.median(p)) if p.size else np.nan,
                "pred_std": float(np.std(p, ddof=1)) if p.size > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def histogram_payload(
    ibge: dict[int, np.ndarray],
    preds: dict[int, np.ndarray],
    years: list[int],
    bins: np.ndarray,
) -> dict:
    """Compact counts for canvases / downstream plots (0.5 t/ha bins)."""
    coarse = np.arange(0.0, 5.5 + 1e-9, 0.5)
    labels = [f"{coarse[i]:.1f}–{coarse[i + 1]:.1f}" for i in range(len(coarse) - 1)]
    years_out = []
    for year in years:
        a = ibge.get(year, np.array([], dtype=float))
        p = preds.get(year, np.array([], dtype=float))
        ibge_counts, _ = np.histogram(a, bins=coarse)
        pred_counts, _ = np.histogram(p, bins=coarse)
        years_out.append(
            {
                "year": int(year),
                "ibge": ibge_counts.astype(int).tolist(),
                "pred": pred_counts.astype(int).tolist(),
            }
        )
    return {"bin_labels": labels, "bin_width": 0.5, "years": years_out}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Yearly IBGE vs best-MoCo municipal productivity histograms."
    )
    p.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Root used to auto-pick the MoCo run with highest test R²",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Override: use this yield-run directory instead of auto-picking",
    )
    p.add_argument(
        "--yield-csv",
        type=Path,
        default=DEFAULT_YIELD_CSV,
        help="PAM municipal yield CSV (all PR municipalities)",
    )
    p.add_argument(
        "--coverage-csv",
        type=Path,
        default=DEFAULT_COVERAGE_CSV,
        help="Coverage-filtered PAM CSV (used when --ibge-set coverage)",
    )
    p.add_argument(
        "--ibge-set",
        choices=("all", "coverage", "matched"),
        default="all",
        help="IBGE sample: all PR munis, coverage 0.8–1.2, or munis with MoCo forecasts",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--counts",
        action="store_true",
        help="Use municipality counts instead of percent of municipalities",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_dir is not None:
        run_dir = run_dir_from_path(args.run_dir)
        testlog = run_dir / "training" / "testlog.csv"
        if testlog.is_file():
            r2 = float(pd.to_numeric(pd.read_csv(testlog).iloc[0]["r2"], errors="coerce"))
        else:
            r2 = float("nan")
    else:
        run_dir, r2 = find_best_moco_run(args.results_root)

    preds = load_predictions_by_year(run_dir)
    years = sorted(preds)
    if args.ibge_set == "coverage":
        ibge_csv = args.coverage_csv
        muni_filter = None
    elif args.ibge_set == "matched":
        ibge_csv = args.yield_csv
        muni_filter = prediction_muni_filter(run_dir)
    else:
        ibge_csv = args.yield_csv
        muni_filter = None

    ibge = load_ibge_by_year(ibge_csv, years, municipality_filter=muni_filter)
    ibge_label = "IBGE"
    pred_label = "Predicted"
    r2_str = f"{r2:.3f}" if np.isfinite(r2) else "n/a"
    title = "Municipal soybean productivity — Paraná"
    bins = np.arange(XMIN, XMAX + 1e-9, BIN_WIDTH)
    percent = not args.counts

    if args.output_dir is None:
        out_dir = run_dir / "figures" / "productivity_histograms"
    else:
        out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize(ibge, preds, years)
    summary_path = out_dir / "yearly_productivity_histogram_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Best MoCo run: {run_dir}")
    print(f"Test R²: {r2_str}")
    print(f"IBGE set: {args.ibge_set}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Wrote {summary_path}")

    payload = histogram_payload(ibge, preds, years, bins)
    payload_path = out_dir / "yearly_productivity_histogram_bins.json"
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {payload_path}")

    plot_two_row(
        ibge,
        preds,
        years=years,
        bins=bins,
        out_path=out_dir / "yearly_productivity_histograms.png",
        title=title,
        ibge_label=ibge_label,
        pred_label=pred_label,
        percent=percent,
    )
    plot_overlay(
        ibge,
        preds,
        years=years,
        bins=bins,
        out_path=out_dir / "yearly_productivity_histograms_overlay.png",
        title=title,
        ibge_label=ibge_label,
        pred_label=pred_label,
        percent=percent,
    )


if __name__ == "__main__":
    main()
