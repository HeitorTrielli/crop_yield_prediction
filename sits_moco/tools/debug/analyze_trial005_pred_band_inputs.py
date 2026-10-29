#!/usr/bin/env python3
"""Compare DualSTNet inputs for PR trial_005 muni-mean LOO prediction band.

Municipalities with predicted yield near the scatterplot ceiling (~3.5–3.75 t/ha)
are stratified by actual IBGE yield. If inputs look similar across strata, the
model is collapsing because the signal is weak; if inputs differ but predictions
do not, the model fails to use that variation.

Outputs under loo_pixel_results/pr_only_trial_005/muni_mean_pred_band_inputs/
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "loo_pixel_results" / "pr_only_trial_005"
NPY_ROOT = Path.home() / "sits_moco_data" / "npy_muni_mean_soy"
OUT_DIR = EXP / "muni_mean_pred_band_inputs"

PRED_LO, PRED_HI = 3.50, 3.75
NO_DATA = -9999.0
SEASON = {
    2020: "2019-2020",
    2021: "2020-2021",
    2022: "2021-2022",
    2023: "2022-2023",
    2024: "2023-2024",
}
FEATURE_COLS = [
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


def load_forecasts() -> pd.DataFrame:
    frames = []
    for y in range(2020, 2025):
        p = (
            EXP
            / f"loo{y}"
            / "muni_mean"
            / "figures"
            / "productivity_scatter"
            / f"productivity_{y}_direct.csv"
        )
        df = pd.read_csv(p)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["municipality_code"] = (
        out["municipality_code"].astype(str).str.replace(r"\.0$", "", regex=True)
    )
    return out


def _valid_spec(spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """spec [T,10] scaled reflectance; returns ndvi [T], valid [T]."""
    invalid = (spec == NO_DATA) | (spec == 0) | ~np.isfinite(spec)
    refl = np.where(invalid.any(axis=1)[:, None], np.nan, spec * 1e-4)
    red = refl[:, 2]
    nir = refl[:, 6]
    ndvi = (nir - red) / (nir + red + 1e-8)
    valid = np.isfinite(ndvi) & ((nir + red) > 0.01)
    return ndvi, valid


def extract_features(code: str, year: int) -> dict[str, float] | None:
    season = SEASON[year]
    base = NPY_ROOT / season / code
    cube_path = base / f"{code}.npy"
    clim_path = base / f"{code}_climate_daily.npy"
    soil_path = base / f"{code}_soil.npy"
    if not cube_path.is_file():
        return None

    cube = np.load(cube_path)
    if cube.ndim != 3 or cube.shape[0] < 1 or cube.shape[2] < 11:
        return None
    arr = cube[0]  # muni mean: single pseudo-pixel [T, C]
    doy = arr[:, 10]
    ndvi, valid = _valid_spec(arr[:, :10].astype(np.float32))
    ndvi_m = np.where(valid, ndvi, np.nan)
    with np.errstate(all="ignore"):
        ndvi_peak = float(np.nanmax(ndvi_m)) if np.any(valid) else np.nan
        ndvi_mean = float(np.nanmean(ndvi_m)) if np.any(valid) else np.nan
        window = valid & (doy >= 60) & (doy <= 135)
        ndvi_window = float(np.nanmean(np.where(window, ndvi, np.nan)))

    feats: dict[str, float] = {
        "ndvi_peak": ndvi_peak,
        "ndvi_mean": ndvi_mean,
        "ndvi_window": ndvi_window,
        "n_valid_days": float(valid.sum()),
        "n_obs_dates": float(arr.shape[0]),
    }

    if clim_path.is_file():
        clim = np.load(clim_path).astype(np.float32)
        if clim.ndim == 3:
            clim = clim[0]
        ok = np.isfinite(clim).all(axis=1) & (clim != NO_DATA).all(axis=1)
        c = np.where(ok[:, None], clim, np.nan)
        pr, eto, rs, tmax, tmin = c.T[:5]
        # Oct=0..30, Nov=31..60, Dec=61..91, Jan=92..122, Feb=123..150/151, Mar=rest
        # Dec–Feb ≈ DOY 61–150 (0-indexed day-of-window)
        dec_feb = slice(61, 151)
        with np.errstate(all="ignore"):
            feats["rain_sum_mm"] = float(np.nansum(pr))
            feats["rain_dec_feb_mm"] = float(np.nansum(pr[dec_feb]))
            feats["eto_sum_mm"] = float(np.nansum(eto))
            feats["tmax_mean"] = float(np.nanmean(tmax))
            feats["tmin_mean"] = float(np.nanmean(tmin))
            feats["rs_mean"] = float(np.nanmean(rs))
            # dry spell: consecutive days with pr < 1 mm
            dry = (pr < 1.0) & np.isfinite(pr)
            run = 0
            best = 0
            for d in dry:
                if d:
                    run += 1
                    best = max(best, run)
                else:
                    run = 0
            feats["dry_spell_max"] = float(best)
    else:
        for k in (
            "rain_sum_mm",
            "rain_dec_feb_mm",
            "eto_sum_mm",
            "tmax_mean",
            "tmin_mean",
            "rs_mean",
            "dry_spell_max",
        ):
            feats[k] = np.nan

    if soil_path.is_file():
        soil = np.load(soil_path).astype(np.float32).reshape(-1)
        feats["clay_pct"] = float(soil[0])
        feats["silt_pct"] = float(soil[1])
        feats["sand_pct"] = float(soil[2])
        feats["soc_t_ha"] = float(soil[3])
    else:
        for k in ("clay_pct", "silt_pct", "sand_pct", "soc_t_ha"):
            feats[k] = np.nan
    return feats


def zscore(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = df[cols].to_numpy(dtype=np.float64)
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (x - mu) / sd


def plot_feature_vs_actual(df: pd.DataFrame, band: pd.DataFrame) -> None:
    cols = [
        "ndvi_peak",
        "ndvi_window",
        "ndvi_mean",
        "rain_sum_mm",
        "rain_dec_feb_mm",
        "dry_spell_max",
        "tmax_mean",
        "clay_pct",
        "soc_t_ha",
    ]
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for ax, col in zip(axes.ravel(), cols):
        ax.scatter(
            df["actual_productivity_t_ha"],
            df[col],
            s=8,
            alpha=0.25,
            c="0.65",
            label="all munis",
            rasterized=True,
        )
        ax.scatter(
            band["actual_productivity_t_ha"],
            band[col],
            s=14,
            alpha=0.7,
            c="#c44e52",
            label="pred 3.50–3.75",
            rasterized=True,
        )
        ax.axvline(3.6, color="0.2", ls="--", lw=0.7)
        ax.set_xlabel("Actual yield (t/ha)")
        ax.set_ylabel(col)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 0].legend(fontsize=8, frameon=False, markerscale=1.5)
    fig.suptitle(
        "PR trial_005 muni-mean LOO · inputs vs actual yield\n"
        "Red = municipalities with predicted yield in [3.50, 3.75] t/ha",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "inputs_vs_actual_highlight_pred_band.png", dpi=160)
    plt.close(fig)


def plot_band_by_actual_stratum(band: pd.DataFrame) -> None:
    b = band.copy()
    b["actual_stratum"] = pd.qcut(
        b["actual_productivity_t_ha"],
        q=3,
        labels=["low actual", "mid actual", "high actual"],
    )
    cols = [
        "ndvi_peak",
        "ndvi_window",
        "rain_sum_mm",
        "rain_dec_feb_mm",
        "dry_spell_max",
        "tmax_mean",
        "clay_pct",
        "soc_t_ha",
        "n_valid_days",
    ]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10.5))
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for ax, col in zip(axes.ravel(), cols):
        data = [
            b.loc[b["actual_stratum"] == lab, col].dropna().to_numpy()
            for lab in ["low actual", "mid actual", "high actual"]
        ]
        bp = ax.boxplot(
            data,
            labels=["low", "mid", "high"],
            patch_artist=True,
            showfliers=False,
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        # Kruskal-Wallis
        ok = [d for d in data if len(d) >= 3]
        p = stats.kruskal(*ok).pvalue if len(ok) == 3 else np.nan
        ax.set_title(f"{col}\nKW p={p:.2g}", fontsize=10)
        ax.set_xlabel("Actual-yield tercile (within pred band)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(
        f"Input distributions inside pred band [{PRED_LO}, {PRED_HI}] t/ha\n"
        "Same predicted yield · split by actual IBGE yield tercile",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pred_band_inputs_by_actual_tercile.png", dpi=160)
    plt.close(fig)
    return b


def plot_pred_vs_actual(df: pd.DataFrame, band: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(
        df["actual_productivity_t_ha"],
        df["predicted_productivity_t_ha"],
        s=10,
        alpha=0.35,
        c="0.55",
        rasterized=True,
    )
    ax.scatter(
        band["actual_productivity_t_ha"],
        band["predicted_productivity_t_ha"],
        s=16,
        alpha=0.85,
        c="#c44e52",
        label=f"pred ∈ [{PRED_LO}, {PRED_HI}]",
        rasterized=True,
    )
    lims = [
        min(df["actual_productivity_t_ha"].min(), df["predicted_productivity_t_ha"].min()),
        max(df["actual_productivity_t_ha"].max(), df["predicted_productivity_t_ha"].max()),
    ]
    ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
    ax.axhline(3.6, color="#c44e52", ls=":", lw=1.0, label="~3.6 t/ha line")
    ax.set_xlabel("Actual yield (t/ha)")
    ax.set_ylabel("Predicted yield (t/ha)")
    ax.set_title("PR trial_005 muni-mean · all LOO years 2020–2024")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scatter_pred_vs_actual_band.png", dpi=160)
    plt.close(fig)


def plot_pca(band: pd.DataFrame) -> dict:
    from sklearn.decomposition import PCA

    cols = [c for c in FEATURE_COLS if band[c].notna().mean() > 0.9]
    x = zscore(band, cols)
    mask = np.isfinite(x).all(axis=1)
    x = x[mask]
    y = band.loc[mask, "actual_productivity_t_ha"].to_numpy()
    pred = band.loc[mask, "predicted_productivity_t_ha"].to_numpy()
    pca = PCA(n_components=2, random_state=0)
    z = pca.fit_transform(x)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc0 = axes[0].scatter(z[:, 0], z[:, 1], c=y, cmap="viridis", s=22, alpha=0.85)
    sc1 = axes[1].scatter(z[:, 0], z[:, 1], c=pred, cmap="coolwarm", s=22, alpha=0.85)
    fig.colorbar(sc0, ax=axes[0], label="Actual yield (t/ha)")
    fig.colorbar(sc1, ax=axes[1], label="Predicted yield (t/ha)")
    for ax in axes:
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_title("Colored by actual yield")
    axes[1].set_title("Colored by predicted yield")
    fig.suptitle(
        f"PCA of z-scored inputs · munis with pred ∈ [{PRED_LO}, {PRED_HI}] t/ha",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pred_band_input_pca.png", dpi=160)
    plt.close(fig)

    # Does PC1 track actual yield?
    r_act = float(stats.spearmanr(z[:, 0], y).correlation)
    r_pred = float(stats.spearmanr(z[:, 0], pred).correlation)
    return {
        "pca_var_pc1": float(pca.explained_variance_ratio_[0]),
        "pca_var_pc2": float(pca.explained_variance_ratio_[1]),
        "spearman_pc1_vs_actual": r_act,
        "spearman_pc1_vs_pred": r_pred,
        "n_pca": int(len(y)),
        "feature_cols_pca": cols,
    }


def summarize(df: pd.DataFrame, band: pd.DataFrame, pca_info: dict) -> dict:
    # Within-band: spearman of each feature with actual (and with pred)
    feat_stats = []
    for col in FEATURE_COLS:
        sub = band[[col, "actual_productivity_t_ha", "predicted_productivity_t_ha"]].dropna()
        if len(sub) < 20:
            continue
        r_a = float(stats.spearmanr(sub[col], sub["actual_productivity_t_ha"]).correlation)
        r_p = float(stats.spearmanr(sub[col], sub["predicted_productivity_t_ha"]).correlation)
        # Compare feature means low vs high actual tercile
        q = sub["actual_productivity_t_ha"].quantile([1 / 3, 2 / 3])
        lo = sub[sub["actual_productivity_t_ha"] <= q.iloc[0]][col]
        hi = sub[sub["actual_productivity_t_ha"] >= q.iloc[1]][col]
        p_mw = float(stats.mannwhitneyu(lo, hi, alternative="two-sided").pvalue)
        feat_stats.append(
            {
                "feature": col,
                "spearman_vs_actual_in_band": r_a,
                "spearman_vs_pred_in_band": r_p,
                "mean_low_actual": float(lo.mean()),
                "mean_high_actual": float(hi.mean()),
                "mannwhitney_low_vs_high_p": p_mw,
            }
        )

    # Input similarity: pairwise L2 in z-space for pairs with similar pred
    # but large actual gap vs similar actual
    cols = [c for c in FEATURE_COLS if band[c].notna().mean() > 0.9]
    b = band.dropna(subset=cols).reset_index(drop=True)
    xz = zscore(b, cols)
    rng = np.random.default_rng(0)
    n = len(b)
    # sample pairs
    n_pairs = min(8000, n * (n - 1) // 2)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    d_in = np.linalg.norm(xz[i] - xz[j], axis=1)
    d_act = np.abs(
        b.loc[i, "actual_productivity_t_ha"].to_numpy()
        - b.loc[j, "actual_productivity_t_ha"].to_numpy()
    )
    # correlation: larger actual gap → larger input distance?
    r_dist = float(stats.spearmanr(d_in, d_act).correlation)

    # Exact duplicate prediction clusters
    vc = df["predicted_productivity_t_ha"].value_counts()
    top_dup = vc[vc >= 5].head(8)
    dup_rows = []
    for pred_val, cnt in top_dup.items():
        sub = df[np.isclose(df["predicted_productivity_t_ha"], pred_val)]
        # join features
        keys = sub[["municipality_code", "harvest_year", "actual_productivity_t_ha"]]
        merged = keys.merge(
            b[
                ["municipality_code", "harvest_year"]
                + cols
            ],
            on=["municipality_code", "harvest_year"],
            how="left",
        )
        feat_mat = merged[cols].to_numpy(dtype=np.float64)
        if np.isfinite(feat_mat).all() and len(feat_mat) >= 2:
            # std across munis sharing exact pred
            feat_std = np.nanstd(feat_mat, axis=0)
            dup_rows.append(
                {
                    "pred": float(pred_val),
                    "n": int(cnt),
                    "actual_min": float(sub["actual_productivity_t_ha"].min()),
                    "actual_max": float(sub["actual_productivity_t_ha"].max()),
                    "actual_std": float(sub["actual_productivity_t_ha"].std()),
                    "mean_feature_std_z": float(np.nanmean(feat_std)),
                    "ndvi_peak_std": float(np.nanstd(merged["ndvi_peak"])),
                    "rain_sum_std": float(np.nanstd(merged["rain_sum_mm"])),
                    "clay_std": float(np.nanstd(merged["clay_pct"])),
                }
            )

    summary = {
        "n_all": int(len(df)),
        "n_band": int(len(band)),
        "pred_band": [PRED_LO, PRED_HI],
        "band_actual": {
            "min": float(band["actual_productivity_t_ha"].min()),
            "max": float(band["actual_productivity_t_ha"].max()),
            "mean": float(band["actual_productivity_t_ha"].mean()),
            "std": float(band["actual_productivity_t_ha"].std()),
        },
        "band_pred": {
            "min": float(band["predicted_productivity_t_ha"].min()),
            "max": float(band["predicted_productivity_t_ha"].max()),
            "mean": float(band["predicted_productivity_t_ha"].mean()),
            "std": float(band["predicted_productivity_t_ha"].std()),
        },
        "by_year": band.groupby("harvest_year")
        .agg(
            n=("predicted_productivity_t_ha", "count"),
            act_min=("actual_productivity_t_ha", "min"),
            act_max=("actual_productivity_t_ha", "max"),
            act_std=("actual_productivity_t_ha", "std"),
        )
        .reset_index()
        .to_dict(orient="records"),
        "feature_stats_in_band": feat_stats,
        "pairwise_input_dist_vs_actual_gap_spearman": r_dist,
        "duplicate_pred_clusters": dup_rows,
        "pca": pca_info,
        "pred_mode_rounded": df["predicted_productivity_t_ha"]
        .round(2)
        .value_counts()
        .head(8)
        .to_dict(),
    }
    return summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_forecasts()
    print(f"Loaded {len(df)} muni-years")

    rows = []
    for r in df.itertuples(index=False):
        feats = extract_features(str(r.municipality_code), int(r.harvest_year))
        if feats is None:
            continue
        feats.update(
            {
                "municipality_code": str(r.municipality_code),
                "municipality_name": getattr(r, "municipality_name", ""),
                "harvest_year": int(r.harvest_year),
                "predicted_productivity_t_ha": float(r.predicted_productivity_t_ha),
                "actual_productivity_t_ha": float(r.actual_productivity_t_ha),
            }
        )
        rows.append(feats)
    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(OUT_DIR / "muni_year_input_features.csv", index=False)
    print(f"Features for {len(feat_df)} / {len(df)} rows -> {OUT_DIR}")

    band = feat_df[
        feat_df["predicted_productivity_t_ha"].between(PRED_LO, PRED_HI)
    ].copy()
    print(f"Band [{PRED_LO}, {PRED_HI}]: n={len(band)}")

    plot_pred_vs_actual(feat_df, band)
    plot_feature_vs_actual(feat_df, band)
    plot_band_by_actual_stratum(band)
    pca_info = plot_pca(band)
    summary = summarize(feat_df, band, pca_info)

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # compact print
    print("\n=== Within-band Spearman(feature, actual) ===")
    for row in sorted(
        summary["feature_stats_in_band"],
        key=lambda r: abs(r["spearman_vs_actual_in_band"]),
        reverse=True,
    ):
        print(
            f"  {row['feature']:<18} ρ_act={row['spearman_vs_actual_in_band']:+.3f}  "
            f"ρ_pred={row['spearman_vs_pred_in_band']:+.3f}  "
            f"MW p(low vs high act)={row['mannwhitney_low_vs_high_p']:.2g}"
        )
    print(
        f"\nPairwise: spearman(input L2, |Δactual|) = "
        f"{summary['pairwise_input_dist_vs_actual_gap_spearman']:+.3f}"
    )
    print(f"PCA PC1 vs actual ρ={pca_info['spearman_pc1_vs_actual']:+.3f}")
    print(f"Wrote figures + summary to {OUT_DIR}")


if __name__ == "__main__":
    main()
