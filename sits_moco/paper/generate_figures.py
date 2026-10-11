"""Build paper figures from Paraná year-LOO trial_005 (pixel_by_pixel)."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOO = ROOT / "loo_pixel_results/pr_only_trial_005"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
YEARS = [2020, 2021, 2022, 2023, 2024]

# --- copy maps / heatmap from representative folds ---
copies = {
    "map_error_2021.png": LOO
    / "loo2021/pixel_by_pixel/figures/municipal_heatmaps/map_pr_error_2021.png",
    "map_error_2022.png": LOO
    / "loo2022/pixel_by_pixel/figures/municipal_heatmaps/map_pr_error_2022.png",
    "guarapuava_heatmap.png": LOO
    / "loo2021/pixel_by_pixel/figures/intramunicipal_heatmap/guarapuava_pr_heatmap.png",
}
for dst_name, src in copies.items():
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copy2(src, OUT / dst_name)
    print("copied", dst_name)

# --- LOO scatter panel: all five holdout years ---
lims = [0.5, 5.5]


def load_year(year: int) -> tuple[np.ndarray, np.ndarray]:
    path = (
        LOO
        / f"loo{year}/pixel_by_pixel/figures/productivity_scatter"
        / f"productivity_{year}_direct.csv"
    )
    if not path.exists():
        # fall back to yield_forecasts
        path = LOO / f"loo{year}/pixel_by_pixel/predictions/yield_forecasts_{year}.csv"
        pred, act = [], []
        with path.open() as f:
            for r in csv.DictReader(f):
                pred.append(float(r["forecast"]))
                act.append(float(r["actual_yield"]))
        return np.asarray(act), np.asarray(pred)
    act, pred = [], []
    with path.open() as f:
        for r in csv.DictReader(f):
            act.append(float(r["actual_productivity_t_ha"]))
            pred.append(float(r["predicted_productivity_t_ha"]))
    return np.asarray(act), np.asarray(pred)


fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
axes_flat = list(axes.ravel())
for i, year in enumerate(YEARS):
    ax = axes_flat[i]
    act, pred = load_year(year)
    ax.scatter(act, pred, s=8, alpha=0.55, c="#1f4e79", edgecolors="none")
    ax.plot(lims, lims, color="0.4", lw=0.8, ls="--")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_box_aspect(1)
    title = f"{year} (drought LOO)" if year == 2022 else f"{year} (LOO)"
    ax.set_title(title, fontsize=9, pad=2)
    ax.tick_params(labelsize=7, length=2.5, pad=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if i % 3 == 0:
        ax.set_ylabel("Predicted (t/ha)", fontsize=8, labelpad=1)
    if i >= 3:
        ax.set_xlabel("Actual (t/ha)", fontsize=8, labelpad=1)

axes_flat[-1].axis("off")
fig.tight_layout()
fig.savefig(OUT / "scatter_panel.pdf", bbox_inches="tight", pad_inches=0.04)
fig.savefig(OUT / "scatter_panel.png", dpi=200, bbox_inches="tight", pad_inches=0.04)
plt.close(fig)
print("wrote scatter_panel")
print("done ->", OUT)
