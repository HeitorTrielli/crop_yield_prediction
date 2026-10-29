#!/usr/bin/env python3
"""Quantify whether pred-band inputs are similar or very different (no torch)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

EXP = Path(__file__).resolve().parents[2] / "loo_pixel_results" / "pr_only_trial_005"
FEAT = EXP / "muni_mean_pred_band_inputs" / "muni_year_input_features.csv"
OUT = EXP / "muni_mean_pred_band_inputs"
PRED_LO, PRED_HI = 3.50, 3.75
COLS = [
    "ndvi_peak",
    "ndvi_mean",
    "ndvi_window",
    "n_valid_days",
    "rain_sum_mm",
    "rain_dec_feb_mm",
    "eto_sum_mm",
    "tmax_mean",
    "tmin_mean",
    "rs_mean",
    "dry_spell_max",
    "clay_pct",
    "silt_pct",
    "sand_pct",
    "soc_t_ha",
]


def zscore(x):
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (x - mu) / sd


def sample_pairs(n, rng, n_pairs=15000):
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    return i[keep], j[keep]


def pair_metrics(x, actual, pred, rng):
    i, j = sample_pairs(len(x), rng)
    d = x[i] - x[j]
    l2 = np.linalg.norm(d, axis=1)
    ni = np.linalg.norm(x[i], axis=1) + 1e-8
    nj = np.linalg.norm(x[j], axis=1) + 1e-8
    cos = 1.0 - np.sum(x[i] * x[j], axis=1) / (ni * nj)
    d_act = np.abs(actual[i] - actual[j])
    d_pred = np.abs(pred[i] - pred[j])
    # also: same-year pairs only
    return {
        "l2_median": float(np.median(l2)),
        "l2_mean": float(np.mean(l2)),
        "l2_p10": float(np.percentile(l2, 10)),
        "l2_p90": float(np.percentile(l2, 90)),
        "cosine_median": float(np.median(cos)),
        "cosine_mean": float(np.mean(cos)),
        "frac_cos_lt_0.05": float(np.mean(cos < 0.05)),
        "frac_cos_lt_0.10": float(np.mean(cos < 0.10)),
        "frac_cos_lt_0.20": float(np.mean(cos < 0.20)),
        "frac_cos_gt_0.50": float(np.mean(cos > 0.50)),
        "spearman_l2_dact": float(stats.spearmanr(l2, d_act).correlation),
        "spearman_l2_dpred": float(stats.spearmanr(l2, d_pred).correlation),
    }


def main():
    rng = np.random.default_rng(0)
    df = pd.read_csv(FEAT)
    band = df[df.predicted_productivity_t_ha.between(PRED_LO, PRED_HI)].copy()
    other = df[~df.predicted_productivity_t_ha.between(PRED_LO, PRED_HI)].copy()

    # Fit z-score on ALL so distances are comparable
    x_all = zscore(df[COLS].to_numpy(float))
    band_idx = df.predicted_productivity_t_ha.between(PRED_LO, PRED_HI).to_numpy()
    x_band = x_all[band_idx]
    x_other = x_all[~band_idx]

    stats_all = pair_metrics(
        x_all,
        df.actual_productivity_t_ha.to_numpy(),
        df.predicted_productivity_t_ha.to_numpy(),
        rng,
    )
    stats_band = pair_metrics(
        x_band,
        band.actual_productivity_t_ha.to_numpy(),
        band.predicted_productivity_t_ha.to_numpy(),
        rng,
    )
    stats_other = pair_metrics(
        x_other,
        other.actual_productivity_t_ha.to_numpy(),
        other.predicted_productivity_t_ha.to_numpy(),
        rng,
    )

    # Within-band vs random: how much of global input variance remains?
    # Mahalanobis-ish: mean ||z|| of band points from global center
    center_dist_band = np.linalg.norm(x_band, axis=1)
    center_dist_all = np.linalg.norm(x_all, axis=1)

    # Feature std retained in band
    std_ratio = {
        c: float(band[c].std() / (df[c].std() + 1e-12)) for c in COLS
    }

    # Compare to "truly similar" baseline: nearest neighbors in input space
    # For each band point, distance to 5th NN among band
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=6, metric="euclidean").fit(x_band)
    dists, _ = nn.kneighbors(x_band)
    nn5 = dists[:, 5]  # 5th neighbor (0 is self)
    nn_all = NearestNeighbors(n_neighbors=6, metric="euclidean").fit(x_all)
    dists_all, _ = nn_all.kneighbors(x_all)
    nn5_all = dists_all[:, 5]

    # Random pair from full data as reference for "different"
    # Typical same-year different munis
    summary = {
        "n_all": int(len(df)),
        "n_band": int(len(band)),
        "pairwise": {"all": stats_all, "band": stats_band, "outside": stats_other},
        "band_over_all_median_l2": stats_band["l2_median"] / stats_all["l2_median"],
        "band_over_outside_median_l2": stats_band["l2_median"] / stats_other["l2_median"],
        "median_norm_from_center_band": float(np.median(center_dist_band)),
        "median_norm_from_center_all": float(np.median(center_dist_all)),
        "median_5nn_l2_band": float(np.median(nn5)),
        "median_5nn_l2_all": float(np.median(nn5_all)),
        "feature_std_band_over_all": std_ratio,
        "mean_feature_std_retained": float(np.mean(list(std_ratio.values()))),
        "interpretation_thresholds": {
            "similar_if_band_median_l2_lt_0.6_of_all": stats_band["l2_median"]
            < 0.6 * stats_all["l2_median"],
            "very_different_if_band_median_l2_gt_0.85_of_all": stats_band["l2_median"]
            > 0.85 * stats_all["l2_median"],
        },
    }

    # Percentile: where does band pairwise median sit in the global pairwise L2 distribution?
    i, j = sample_pairs(len(x_all), rng, 30000)
    global_l2 = np.linalg.norm(x_all[i] - x_all[j], axis=1)
    summary["band_median_l2_percentile_in_global_pairs"] = float(
        100 * np.mean(global_l2 <= stats_band["l2_median"])
    )

    with open(OUT / "input_similarity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("\n=== Plain-language ===")
    ratio = summary["band_over_all_median_l2"]
    print(f"Band pairwise median L2 / all pairwise median L2 = {ratio:.2f}")
    print(
        f"Band median L2 is at percentile {summary['band_median_l2_percentile_in_global_pairs']:.0f} "
        f"of global random pairs"
    )
    print(f"Mean feature std retained in band: {summary['mean_feature_std_retained']:.2f}")
    if ratio > 0.85:
        print("=> Inputs in the pred band are STILL VERY DIFFERENT (nearly as spread as all munis).")
    elif ratio > 0.6:
        print("=> Inputs in the pred band are SOMEWHAT SIMILAR but still substantially spread.")
    else:
        print("=> Inputs in the pred band are SIMILAR (tight cluster relative to all munis).")


if __name__ == "__main__":
    main()
