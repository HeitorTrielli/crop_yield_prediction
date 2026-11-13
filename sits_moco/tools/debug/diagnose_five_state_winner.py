#!/usr/bin/env python3
"""Diagnose five-state winner (decoder muni_rank wide trial_005).

Uses exported holdout scatters: R² / bias / dispersion by year×state,
prediction-band collapse, and residual structure.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "loo_pixel_results" / "pr_rs_sp_go_sc_decoder_muni_rank_wide_trial_005"
OUT = EXP / "diagnostics"
YEARS = range(2020, 2025)
PRED_LO, PRED_HI = 3.50, 3.75
STATE_RE = re.compile(r"-\s*([A-Z]{2})\s*$")


def load_all() -> pd.DataFrame:
    rows = []
    for year in YEARS:
        p = (
            EXP
            / f"loo{year}"
            / "muni_mean"
            / "figures"
            / "productivity_scatter"
            / f"productivity_{year}_direct.csv"
        )
        df = pd.read_csv(p)
        df["harvest_year"] = year
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["municipality_code"] = out["municipality_code"].astype(str)
    out["state"] = out["municipality_name"].astype(str).str.extract(STATE_RE, expand=False)
    out["actual"] = out["actual_productivity_t_ha"].astype(float)
    out["pred"] = out["predicted_productivity_t_ha"].astype(float)
    out["err"] = out["pred"] - out["actual"]
    return out


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    actual = np.asarray(actual, float)
    pred = np.asarray(pred, float)
    m = np.isfinite(actual) & np.isfinite(pred)
    actual, pred = actual[m], pred[m]
    if len(actual) < 5:
        return {"n": int(len(actual))}
    a_std = float(np.std(actual))
    p_std = float(np.std(pred))
    return {
        "n": int(len(actual)),
        "r2": float(r2_score(actual, pred)),
        "mae": float(mean_absolute_error(actual, pred)),
        "bias": float(np.mean(pred - actual)),
        "actual_mean": float(np.mean(actual)),
        "pred_mean": float(np.mean(pred)),
        "actual_std": a_std,
        "pred_std": p_std,
        "std_ratio": p_std / a_std if a_std > 1e-8 else float("nan"),
        "spearman": float(pd.Series(actual).corr(pd.Series(pred), method="spearman")),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_all()
    df.to_csv(OUT / "all_holdout_predictions.csv", index=False)

    # ---- overall / by year / by state / year×state ----
    overall = metrics(df["actual"], df["pred"])
    by_year = (
        df.groupby("harvest_year")
        .apply(lambda g: pd.Series(metrics(g["actual"], g["pred"])), include_groups=False)
        .reset_index()
    )
    by_state = (
        df.groupby("state")
        .apply(lambda g: pd.Series(metrics(g["actual"], g["pred"])), include_groups=False)
        .reset_index()
        .sort_values("r2", ascending=False)
    )
    by_ys = (
        df.groupby(["harvest_year", "state"])
        .apply(lambda g: pd.Series(metrics(g["actual"], g["pred"])), include_groups=False)
        .reset_index()
    )
    by_year.to_csv(OUT / "metrics_by_year.csv", index=False)
    by_state.to_csv(OUT / "metrics_by_state.csv", index=False)
    by_ys.to_csv(OUT / "metrics_by_year_state.csv", index=False)

    # ---- prediction collapse band ----
    in_band = df["pred"].between(PRED_LO, PRED_HI)
    band = df[in_band]
    collapse = {
        "n_all": int(len(df)),
        "n_band": int(in_band.sum()),
        "frac_band": float(in_band.mean()),
        "all": overall,
        "band": metrics(band["actual"], band["pred"]),
        "pred_mode_rounded": df["pred"].round(2).value_counts().head(10).to_dict(),
    }

    # duplicate near-identical preds
    rounded = df["pred"].round(3)
    dup = (
        df.assign(pred_r=rounded)
        .groupby("pred_r")
        .agg(
            n=("actual", "size"),
            actual_std=("actual", "std"),
            actual_min=("actual", "min"),
            actual_max=("actual", "max"),
            n_states=("state", "nunique"),
            n_years=("harvest_year", "nunique"),
        )
        .reset_index()
        .query("n >= 15")
        .sort_values("n", ascending=False)
    )
    dup.to_csv(OUT / "duplicate_pred_clusters.csv", index=False)

    # ---- residual by state in weak years ----
    weak = by_ys[by_ys["harvest_year"].isin([2021, 2024])].sort_values("r2")
    strong = by_ys[by_ys["harvest_year"].isin([2022, 2023])].sort_values(
        "r2", ascending=False
    )

    summary = {
        "model": "pr_rs_sp_go_sc_decoder_muni_rank_wide_trial_005",
        "overall": overall,
        "collapse": collapse,
        "by_year": by_year.to_dict(orient="records"),
        "by_state": by_state.to_dict(orient="records"),
        "weak_year_state_worst": weak.head(10).to_dict(orient="records"),
        "strong_year_state_best": strong.head(10).to_dict(orient="records"),
        "n_dup_clusters_ge15": int(len(dup)),
    }
    with open(OUT / "diagnosis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ---- plots ----
    # 1) year×state R2 heatmap
    states = sorted(df["state"].dropna().unique())
    mat = np.full((len(YEARS), len(states)), np.nan)
    for i, year in enumerate(YEARS):
        for j, st in enumerate(states):
            sub = by_ys[(by_ys["harvest_year"] == year) & (by_ys["state"] == st)]
            if len(sub):
                mat[i, j] = sub["r2"].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-0.2, vmax=0.8, aspect="auto")
    ax.set_xticks(range(len(states)))
    ax.set_xticklabels(states)
    ax.set_yticks(range(len(list(YEARS))))
    ax.set_yticklabels(list(YEARS))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Five-state winner · holdout R² by year × state")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="R²")
    fig.tight_layout()
    fig.savefig(OUT / "r2_heatmap_year_state.png", dpi=160)
    plt.close(fig)

    # 2) scatter actual vs pred colored by state, faceted by year
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2), sharex=True, sharey=True)
    cmap = {s: c for s, c in zip(states, plt.cm.tab10.colors)}
    for ax, year in zip(axes, YEARS):
        sub = df[df["harvest_year"] == year]
        for st in states:
            m = sub["state"] == st
            ax.scatter(
                sub.loc[m, "actual"],
                sub.loc[m, "pred"],
                s=8,
                alpha=0.55,
                c=[cmap[st]],
                label=st,
                rasterized=True,
                edgecolors="none",
            )
        lo = min(sub["actual"].min(), sub["pred"].min())
        hi = max(sub["actual"].max(), sub["pred"].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        r = metrics(sub["actual"], sub["pred"])
        ax.set_title(f"{year}\nR²={r['r2']:.2f} σp/σa={r['std_ratio']:.2f}", fontsize=9)
        ax.set_xlabel("actual")
        if year == 2020:
            ax.set_ylabel("pred")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("Five-state winner · holdout scatters", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "scatter_by_year_state.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 3) pred vs actual std by year
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(list(YEARS)))
    ax.bar(x - 0.15, by_year["actual_std"], width=0.3, label="actual σ", color="#4c78a8")
    ax.bar(x + 0.15, by_year["pred_std"], width=0.3, label="pred σ", color="#e45756")
    ax.set_xticks(x)
    ax.set_xticklabels(by_year["harvest_year"].astype(int))
    ax.set_ylabel("std (t/ha)")
    ax.set_title("Dispersion: does the model under-spread?")
    ax.legend(frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "dispersion_by_year.png", dpi=160)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote diagnostics to {OUT}")


if __name__ == "__main__":
    main()
