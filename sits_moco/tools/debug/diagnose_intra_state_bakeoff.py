#!/usr/bin/env python3
"""Diagnose each finished trial in the intra-state bakeoff.

Builds holdout prediction tables from yield_forecasts_*.csv and writes the same
style of year×state metrics / plots as diagnose_five_state_winner.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.aux_losses import ibge_state_from_code  # noqa: E402
from yield_utils import within_state_year_metrics  # noqa: E402

STUDY = (
    ROOT
    / "results"
    / "tuning"
    / "productivity_muni_mean_dual_stnet_pr_rs_sp_go_sc_intra_state_bakeoff_year_loo"
)
UF = {"35": "SP", "41": "PR", "42": "SC", "43": "RS", "52": "GO"}
PRED_LO, PRED_HI = 3.50, 3.75


def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    if len(actual) < 2:
        return {"n": int(len(actual)), "r2": np.nan, "mae": np.nan, "spearman": np.nan}
    return {
        "n": int(len(actual)),
        "r2": float(r2_score(actual, pred)),
        "mae": float(mean_absolute_error(actual, pred)),
        "spearman": float(spearmanr(actual, pred).correlation),
        "actual_std": float(np.std(actual)),
        "pred_std": float(np.std(pred)),
        "std_ratio": float(np.std(pred) / max(np.std(actual), 1e-8)),
    }


def load_trial_preds(trial_dir: Path) -> pd.DataFrame | None:
    rows = []
    for year in range(2020, 2025):
        path = (
            trial_dir
            / f"loo{year}"
            / "predictions_muni_mean"
            / f"yield_forecasts_{year}.csv"
        )
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        rows.append(
            pd.DataFrame(
                {
                    "municipality_code": df["municipality_code"].astype(str),
                    "harvest_year": year,
                    "actual": df["actual_yield"].astype(float),
                    "pred": df["forecast"].astype(float),
                }
            )
        )
    if not rows:
        return None
    out = pd.concat(rows, ignore_index=True)
    out["state"] = out["municipality_code"].map(ibge_state_from_code).map(UF)
    return out


def diagnose_df(df: pd.DataFrame, out_dir: Path, label: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "all_holdout_predictions.csv", index=False)

    overall = _metrics(df["actual"].to_numpy(), df["pred"].to_numpy())
    ws = within_state_year_metrics(
        df["pred"].to_numpy(),
        df["actual"].to_numpy(),
        df["municipality_code"].tolist(),
        df["harvest_year"].tolist(),
        min_count=20,
    )
    by_state = []
    for st, g in df.groupby("state"):
        m = _metrics(g["actual"].to_numpy(), g["pred"].to_numpy())
        m["state"] = st
        by_state.append(m)
    by_ys = []
    for (y, st), g in df.groupby(["harvest_year", "state"]):
        m = _metrics(g["actual"].to_numpy(), g["pred"].to_numpy())
        m["harvest_year"] = int(y)
        m["state"] = st
        by_ys.append(m)
    by_ys_df = pd.DataFrame(by_ys)
    by_ys_df.to_csv(out_dir / "metrics_by_year_state.csv", index=False)
    pd.DataFrame(by_state).to_csv(out_dir / "metrics_by_state.csv", index=False)

    band = df[(df["pred"] >= PRED_LO) & (df["pred"] <= PRED_HI)]
    summary = {
        "label": label,
        "overall": overall,
        "within_state": ws,
        "by_state": by_state,
        "collapse": {
            "n_band": int(len(band)),
            "frac_band": float(len(band) / max(len(df), 1)),
            "band": _metrics(band["actual"].to_numpy(), band["pred"].to_numpy())
            if len(band) >= 2
            else {},
        },
        "weak_year_state": by_ys_df.sort_values("r2").head(8).to_dict(orient="records"),
    }
    with open(out_dir / "diagnosis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # heatmap
    states = ["GO", "PR", "RS", "SC", "SP"]
    years = sorted(df["harvest_year"].unique())
    mat = np.full((len(years), len(states)), np.nan)
    for i, y in enumerate(years):
        for j, st in enumerate(states):
            sub = by_ys_df[(by_ys_df["harvest_year"] == y) & (by_ys_df["state"] == st)]
            if len(sub):
                mat[i, j] = sub["r2"].iloc[0]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-0.2, vmax=0.8, aspect="auto")
    ax.set_xticks(range(len(states)))
    ax.set_xticklabels(states)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels([str(y) for y in years])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"{label} · holdout R² by year × state")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "r2_heatmap_year_state.png", dpi=160)
    plt.close(fig)
    return summary


def main() -> None:
    results = []
    if not STUDY.is_dir():
        print(f"Study not found: {STUDY}")
        return
    for trial_dir in sorted(STUDY.glob("trial_*")):
        df = load_trial_preds(trial_dir)
        if df is None or df["harvest_year"].nunique() < 5:
            n = 0 if df is None else int(df["harvest_year"].nunique())
            print(f"{trial_dir.name}: incomplete ({n}/5 folds with forecasts)")
            continue
        out = trial_dir / "diagnostics"
        # params hint
        params_path = trial_dir / "params.json"
        label = trial_dir.name
        if params_path.is_file():
            params = json.loads(params_path.read_text(encoding="utf-8"))
            aux = params.get("aux_loss", "?")
            w = params.get("aux_loss_weight", "?")
            demean = params.get("mse_within_state_year", False)
            label = f"{trial_dir.name} · {aux}@{w}" + (" · demean" if demean else "")
        summary = diagnose_df(df, out, label)
        results.append(
            {
                "trial": trial_dir.name,
                "label": label,
                "pooled_r2": summary["overall"]["r2"],
                "within_state_r2": summary["within_state"]["within_state_r2"],
                "within_state_spearman": summary["within_state"][
                    "within_state_spearman"
                ],
                "by_state": {
                    r["state"]: {"r2": r["r2"], "spearman": r["spearman"]}
                    for r in summary["by_state"]
                },
            }
        )
        print(
            f"{label}: pooled_r2={summary['overall']['r2']:.3f} "
            f"ws_r2={summary['within_state']['within_state_r2']:.3f} "
            f"ws_sp={summary['within_state']['within_state_spearman']:.3f}"
        )

    out_json = STUDY / "bakeoff_diagnosis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
