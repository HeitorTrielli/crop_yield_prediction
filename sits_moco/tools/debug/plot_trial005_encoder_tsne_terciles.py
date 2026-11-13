#!/usr/bin/env python3
"""Clearer t-SNE views: actual yield terciles + per-year panels.

Reuses tsne_coordinates.csv from plot_trial005_encoder_tsne.py (no re-encode).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.debug.analyze_trial005_encoder_vs_decoder import (  # noqa: E402
    PRED_HI,
    PRED_LO,
)

DEFAULT_OUT = ROOT / "loo_pixel_results" / "pr_only_trial_005" / "encoder_tsne"

TERCILE_COLORS = {
    "baixo": "#4c78a8",
    "médio": "#f58518",
    "alto": "#54a24b",
}
TERCILE_ORDER = ("baixo", "médio", "alto")


def assign_terciles(y: np.ndarray) -> tuple[np.ndarray, dict]:
    q1, q2 = np.quantile(y, [1 / 3, 2 / 3])
    labels = np.full(len(y), "médio", dtype=object)
    labels[y <= q1] = "baixo"
    labels[y > q2] = "alto"
    return labels, {"q33": float(q1), "q66": float(q2)}


def neighbor_purity(xy: np.ndarray, labels: np.ndarray, k: int = 15) -> float:
    nn = NearestNeighbors(n_neighbors=k + 1).fit(xy)
    idx = nn.kneighbors(return_distance=False)[:, 1:]
    same = labels[idx] == labels[:, None]
    return float(np.mean(same))


def scatter_terciles(ax, xy, labels, *, title, sizes=None, alphas=None, order=None):
    order = order or TERCILE_ORDER
    sizes = np.full(len(xy), 14.0) if sizes is None else sizes
    alphas = np.full(len(xy), 0.75) if alphas is None else alphas
    # draw low alpha first so highlighted points sit on top
    for name in order:
        m = labels == name
        if not np.any(m):
            continue
        ax.scatter(
            xy[m, 0],
            xy[m, 1],
            s=sizes[m],
            c=TERCILE_COLORS[name],
            alpha=float(np.mean(alphas[m])),
            edgecolors="none",
            rasterized=True,
            label=name,
            zorder=2 if np.mean(alphas[m]) > 0.5 else 1,
        )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--coord-csv",
        type=Path,
        default=None,
        help="Defaults to <output-dir>/tsne_coordinates.csv",
    )
    p.add_argument("--title-prefix", type=str, default="Fused encoder t-SNE")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT = Path(args.output_dir)
    COORD = Path(args.coord_csv) if args.coord_csv else OUT / "tsne_coordinates.csv"
    df = pd.read_csv(COORD)
    actual = df["actual"].to_numpy(dtype=np.float64)
    pred = df["decoder_prediction"].to_numpy(dtype=np.float64)
    year = df["harvest_year"].to_numpy(dtype=np.int64)
    xy = df[["tsne_fused_1", "tsne_fused_2"]].to_numpy(dtype=np.float64)
    in_band = (pred >= PRED_LO) & (pred <= PRED_HI)

    # Global terciles on full sample (same cuts everywhere)
    terc_all, cuts_all = assign_terciles(actual)
    # Within-band terciles (only among people with similar pred)
    terc_band, cuts_band = assign_terciles(actual[in_band])

    purity = {
        "all_actual_tercile_15nn": neighbor_purity(xy, terc_all, k=15),
        "all_pred_tercile_15nn": neighbor_purity(xy, assign_terciles(pred)[0], k=15),
        "band_actual_tercile_15nn": neighbor_purity(
            xy[in_band], terc_band, k=15
        ),
        "band_pred_tercile_15nn": neighbor_purity(
            xy[in_band], assign_terciles(pred[in_band])[0], k=15
        ),
    }
    # chance baseline for 3 balanced classes ≈ 1/3
    purity["chance_3class"] = 1.0 / 3.0

    # ---- Fig 1: full sample vs band, discrete terciles ----
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))

    scatter_terciles(
        axes[0],
        xy,
        terc_all,
        title=f"All munis · actual tercile\n"
        f"(cuts {cuts_all['q33']:.2f} / {cuts_all['q66']:.2f} t/ha)",
    )
    axes[0].legend(frameon=False, fontsize=8, loc="best", title="actual")

    # Band highlighted with band-specific terciles; grey outside
    axes[1].scatter(
        xy[~in_band, 0],
        xy[~in_band, 1],
        s=8,
        c="0.82",
        alpha=0.3,
        edgecolors="none",
        rasterized=True,
        label="fora do band",
    )
    band_labels_full = np.full(len(df), "", dtype=object)
    band_labels_full[in_band] = terc_band
    for name in TERCILE_ORDER:
        m = band_labels_full == name
        axes[1].scatter(
            xy[m, 0],
            xy[m, 1],
            s=16,
            c=TERCILE_COLORS[name],
            alpha=0.85,
            edgecolors="none",
            rasterized=True,
            label=name,
        )
    axes[1].set_title(
        f"Pred band [3.50, 3.75] · actual tercile inside band\n"
        f"(cuts {cuts_band['q33']:.2f} / {cuts_band['q66']:.2f}; n={in_band.sum()})"
    )
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    for spine in axes[1].spines.values():
        spine.set_visible(False)

    # Band only, larger
    scatter_terciles(
        axes[2],
        xy[in_band],
        terc_band,
        title="Band only · actual tercile\n"
        f"15-NN purity={purity['band_actual_tercile_15nn']:.2f} "
        f"(chance≈{purity['chance_3class']:.2f})",
    )
    axes[2].legend(frameon=False, fontsize=8, loc="best", title="actual")

    fig.suptitle(
        "Fused encoder t-SNE · discrete actual yield (baixo / médio / alto)\n"
        "Continuous colormap hid this; terciles make mixing vs separation readable",
        fontsize=11,
    )
    fig.tight_layout()
    path1 = OUT / "tsne_fused_actual_terciles.png"
    fig.savefig(path1, dpi=180)
    plt.close(fig)
    print(f"Wrote {path1}")

    # ---- Fig 2: one panel per year (full sample, global terciles) ----
    years = sorted(np.unique(year))
    fig, axes = plt.subplots(1, len(years), figsize=(3.2 * len(years), 3.6), sharex=False)
    for ax, y in zip(axes, years):
        m = year == y
        # faint all points for context
        ax.scatter(
            xy[~m, 0],
            xy[~m, 1],
            s=4,
            c="0.88",
            alpha=0.25,
            edgecolors="none",
            rasterized=True,
        )
        scatter_terciles(
            ax,
            xy[m],
            terc_all[m],
            title=f"{y} (n={m.sum()})",
        )
        # local purity within year
        pur = neighbor_purity(xy[m], terc_all[m], k=10)
        ax.text(
            0.02,
            0.02,
            f"10-NN purity={pur:.2f}",
            transform=ax.transAxes,
            fontsize=7,
            va="bottom",
        )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=TERCILE_COLORS[n],
            markersize=7,
            label=n,
        )
        for n in TERCILE_ORDER
    ]
    fig.legend(handles=handles, frameon=False, fontsize=9, loc="upper center", ncol=3)
    fig.suptitle(
        "Within each year: does fused t-SNE separate baixo / médio / alto actual?\n"
        f"Global cuts {cuts_all['q33']:.2f} / {cuts_all['q66']:.2f} t/ha · "
        f"all-sample 15-NN purity={purity['all_actual_tercile_15nn']:.2f}",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    path2 = OUT / "tsne_fused_actual_terciles_by_year.png"
    fig.savefig(path2, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path2}")

    # ---- Fig 3: within-year band only (the honest hard case) ----
    fig, axes = plt.subplots(1, len(years), figsize=(3.2 * len(years), 3.6))
    year_stats = []
    for ax, y in zip(axes, years):
        m = (year == y) & in_band
        if m.sum() < 5:
            ax.set_title(f"{y} (n={m.sum()})")
            ax.axis("off")
            continue
        # terciles within that year's band
        local_terc, local_cuts = assign_terciles(actual[m])
        scatter_terciles(ax, xy[m], local_terc, title=f"{y} band (n={m.sum()})")
        pur = neighbor_purity(xy[m], local_terc, k=min(10, max(3, m.sum() // 5)))
        ax.text(
            0.02,
            0.02,
            f"purity={pur:.2f}\n"
            f"act σ={actual[m].std():.2f}\npred σ={pred[m].std():.2f}",
            transform=ax.transAxes,
            fontsize=7,
            va="bottom",
        )
        year_stats.append(
            {
                "year": int(y),
                "n_band": int(m.sum()),
                "actual_std": float(actual[m].std()),
                "pred_std": float(pred[m].std()),
                "nn_purity_actual_tercile": float(pur),
                "cuts": local_cuts,
            }
        )
    fig.legend(handles=handles, frameon=False, fontsize=9, loc="upper center", ncol=3)
    fig.suptitle(
        "Hard case: same-pred band, split by year, colored by local actual tercile\n"
        "If colors mix inside each blob → encoder also struggles on residual yield",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    path3 = OUT / "tsne_fused_band_actual_terciles_by_year.png"
    fig.savefig(path3, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path3}")

    summary = {
        "cuts_all": cuts_all,
        "cuts_band": cuts_band,
        "neighbor_purity": purity,
        "by_year_band": year_stats,
        "figures": [str(path1), str(path2), str(path3)],
        "note": (
            "Continuous viridis on the band plot mixes midtones; "
            "terciles make high/mid/low readable. Neighbor purity near chance "
            "⇒ little local actual structure in 2D t-SNE."
        ),
    }
    with open(OUT / "tsne_tercile_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
