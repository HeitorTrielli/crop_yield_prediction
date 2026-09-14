"""Copy run figures and generate learning-curve / incomplete-series plots."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / (
    "results/tuning/productivity_top5_spectral_xavier_full/trial_003"
)
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# --- copy priority PNGs ---
copies = {
    "scatter_2021.png": RUN / "figures/productivity_scatter/productivity_scatter_2021_direct.png",
    "scatter_2020.png": RUN / "figures/productivity_scatter/productivity_scatter_2020_direct.png",
    "scatter_2022.png": RUN / "figures/productivity_scatter/productivity_scatter_2022_direct.png",
    "scatter_2023.png": RUN / "figures/productivity_scatter/productivity_scatter_2023_direct.png",
    "scatter_2024.png": RUN / "figures/productivity_scatter/productivity_scatter_2024_direct.png",
    "map_error_2021.png": RUN / "figures/municipal_heatmaps/map_pr_error_2021.png",
    "guarapuava_heatmap.png": RUN / "figures/intramunicipal_heatmap/guarapuava_pr_heatmap.png",
}
for dst_name, src in copies.items():
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copy2(src, OUT / dst_name)
    print("copied", dst_name)

# --- learning curves ---
rows = list(csv.DictReader((RUN / "training/trainlog.csv").open()))
epochs = [int(r["epoch"]) for r in rows]
rmse = [float(r["rmse"]) for r in rows]
r2 = [float(r["r2"]) for r in rows]
trainloss = [float(r["trainloss"]) for r in rows]
valloss = [float(r["valloss"]) for r in rows]
best_ep = max(range(len(r2)), key=lambda i: r2[i])

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
axes[0].plot(epochs, rmse, color="#1f4e79", lw=1.6)
axes[0].axvline(epochs[best_ep], color="#c45c26", ls="--", lw=1, label=f"best (ep {epochs[best_ep]})")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Val RMSE (t/ha)")
axes[0].legend(fontsize=8, frameon=False)
axes[0].set_title("(a) Validation RMSE")

axes[1].plot(epochs, r2, color="#1f4e79", lw=1.6)
axes[1].axvline(epochs[best_ep], color="#c45c26", ls="--", lw=1)
axes[1].axhline(0, color="0.5", lw=0.8)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel(r"Val $R^2$")
axes[1].set_title(r"(b) Validation $R^2$")

axes[2].plot(epochs, trainloss, color="#1f4e79", lw=1.4, label="train")
axes[2].plot(epochs, valloss, color="#c45c26", lw=1.4, label="val")
axes[2].axvline(epochs[best_ep], color="0.4", ls="--", lw=1)
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("MSE loss")
axes[2].legend(fontsize=8, frameon=False)
axes[2].set_title("(c) Training / validation loss")

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "learning_curves.pdf", bbox_inches="tight")
fig.savefig(OUT / "learning_curves.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote learning_curves")

# --- incomplete series ---
inc = list(csv.DictReader((RUN / "predictions/incomplete_series_evaluation.csv").open()))
k = [int(r["num_periods"]) for r in inc]
inc_r2 = [float(r["r2"]) for r in inc]
inc_rmse = [float(r["rmse"]) for r in inc]
inc_mape = [float(r["mape"]) for r in inc]

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
axes[0].plot(k, inc_r2, marker="o", color="#1f4e79", lw=1.6)
axes[0].axhline(0, color="0.5", lw=0.8)
axes[0].set_xlabel("Season months available ($k$)")
axes[0].set_ylabel(r"$R^2$")
axes[0].set_xticks(k)
axes[0].set_title(r"(a) $R^2$ vs season length")

axes[1].plot(k, inc_rmse, marker="o", color="#1f4e79", lw=1.6)
axes[1].set_xlabel("Season months available ($k$)")
axes[1].set_ylabel("RMSE (t/ha)")
axes[1].set_xticks(k)
axes[1].set_title("(b) RMSE vs season length")

axes[2].plot(k, inc_mape, marker="o", color="#1f4e79", lw=1.6)
axes[2].set_xlabel("Season months available ($k$)")
axes[2].set_ylabel("MAPE (%)")
axes[2].set_xticks(k)
axes[2].set_title("(c) MAPE vs season length")

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "incomplete_series.pdf", bbox_inches="tight")
fig.savefig(OUT / "incomplete_series.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote incomplete_series")

# Combined multi-year scatter panel from CSVs (cleaner for paper than 5 separate)
cov_rows = list(csv.DictReader((ROOT / "files/pam_soy_pr_2019_2025_coverage_0.8_1.2.csv").open()))
allowed = {(str(r["municipality_code"]), int(r["year"])) for r in cov_rows}
sc = RUN / "figures/productivity_scatter"
fig, axes = plt.subplots(1, 5, figsize=(12.5, 2.6), sharex=True, sharey=True)
for ax, year in zip(axes, range(2020, 2025)):
    rows = [
        r
        for r in csv.DictReader((sc / f"productivity_{year}_direct.csv").open())
        if (str(r["municipality_code"]), year) in allowed
    ]
    pred = np.array([float(r["predicted_productivity_t_ha"]) for r in rows])
    act = np.array([float(r["actual_productivity_t_ha"]) for r in rows])
    splits = [r["split"] for r in rows]
    for split, color, marker in [
        ("train", "#1f4e79", "o"),
        ("valid", "#c45c26", "s"),
        ("test", "#2a7f62", "^"),
        ("eval", "#2a7f62", "^"),
    ]:
        m = np.array([s == split for s in splits])
        if m.any():
            ax.scatter(act[m], pred[m], s=10, alpha=0.55, c=color, marker=marker, label=split, edgecolors="none")
    lims = [0.5, 5.5]
    ax.plot(lims, lims, color="0.4", lw=0.9, ls="--")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_title(str(year), fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if year == 2020:
        ax.set_ylabel("Predicted (t/ha)")
    ax.set_xlabel("Actual (t/ha)")
handles, labels = axes[1].get_legend_handles_labels()
# unique
seen = {}
for h, l in zip(handles, labels):
    seen[l] = h
fig.legend(seen.values(), seen.keys(), loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.08))
fig.tight_layout()
fig.savefig(OUT / "scatter_panel.pdf", bbox_inches="tight")
fig.savefig(OUT / "scatter_panel.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote scatter_panel")
print("done ->", OUT)
