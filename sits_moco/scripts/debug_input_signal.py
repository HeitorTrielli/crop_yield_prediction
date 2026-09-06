"""Diagnose whether model inputs can resolve high productivity.

For each municipality-season, compute summary features from the raw .npy
(after the same nodata->0, *1e-4 handling as PixelTransform) and join with
IBGE yields. If features are flat above ~3.5 t/ha, the input data cannot
separate mid from high yields (NDVI saturation / too few observations).

Run:  python scripts/debug_input_signal.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
NPY_ROOT = REPO / "files" / "npy"
YIELD_CSV = REPO / "files" / "pam_soy_pr_2019_2025_coverage_0.8_1.2.csv"
OUT_DIR = REPO / "results" / "debug_inputs"
MAX_PIXELS = 1500
RNG = np.random.default_rng(0)

SEASON_TO_YEAR = {
    "2019-2020": 2020,
    "2020-2021": 2021,
    "2021-2022": 2022,
    "2022-2023": 2023,
    "2023-2024": 2024,
}


def muni_features(path: Path) -> dict | None:
    arr = np.load(path, mmap_mode="r")
    n, t, c = arr.shape
    if n == 0 or c < 13:
        return None
    if n > MAX_PIXELS:
        idx = np.sort(RNG.choice(n, MAX_PIXELS, replace=False))
        sub = np.asarray(arr[idx])
    else:
        sub = np.asarray(arr)

    doy = sub[:, :, 10]
    spec = sub[:, :, :10].astype(np.float32)
    invalid = (spec == -9999) | ~np.isfinite(spec)
    frac_invalid = float(invalid.any(axis=2).mean())
    spec = np.where(invalid, 0.0, spec) * 1e-4  # same as PixelTransform

    red = spec[:, :, 2]
    nir = spec[:, :, 6]
    ndvi = (nir - red) / (nir + red + 1e-8)
    valid_day = ~invalid.any(axis=2) & (nir + red > 0.01)

    ndvi_masked = np.where(valid_day, ndvi, np.nan)
    with np.errstate(all="ignore"):
        px_peak = np.nanmax(ndvi_masked, axis=1)
        px_mean = np.nanmean(ndvi_masked, axis=1)
    # NDVI in the peak vegetative window (season DOY 60-135 ~ Dec-Feb)
    window = (doy >= 60) & (doy <= 135) & valid_day
    ndvi_win = np.where(window, ndvi, np.nan)
    with np.errstate(all="ignore"):
        px_win = np.nanmean(ndvi_win, axis=1)

    rain = sub[:, :, 11]
    dry = sub[:, :, 12]
    rain_valid = rain != -9999
    rain_end = np.nanmax(np.where(rain_valid, rain, np.nan), axis=1)
    dry_max = np.nanmax(np.where(rain_valid, dry, np.nan), axis=1)

    return {
        "n_pixels": int(n),
        "t_days": int(t),
        "doy_min": float(np.nanmin(doy)),
        "doy_max": float(np.nanmax(doy)),
        "valid_days_mean": float(valid_day.sum(axis=1).mean()),
        "frac_invalid_obs": frac_invalid,
        "ndvi_peak": float(np.nanmean(px_peak)),
        "ndvi_peak_p90": float(np.nanpercentile(px_peak, 90)),
        "ndvi_mean": float(np.nanmean(px_mean)),
        "ndvi_window": float(np.nanmean(px_win)),
        "frac_window_covered": float(window.any(axis=1).mean()),
        "rain_cum_end": float(np.nanmean(rain_end)),
        "dry_streak_max": float(np.nanmean(dry_max)),
    }


def main() -> None:
    ydf = pd.read_csv(YIELD_CSV)
    ydf["municipality_code"] = ydf["municipality_code"].astype(str)
    ylookup = {
        (str(r.municipality_code), int(r.year)): float(r.yield_t_ha)
        for r in ydf.itertuples()
    }

    rows = []
    for season, year in SEASON_TO_YEAR.items():
        sdir = NPY_ROOT / season
        if not sdir.is_dir():
            print(f"skip {season}: missing")
            continue
        munis = sorted(os.listdir(sdir))
        print(f"{season}: {len(munis)} municipalities")
        for i, muni in enumerate(munis):
            p = sdir / muni / f"{muni}.npy"
            if not p.is_file():
                continue
            y = ylookup.get((muni, year))
            if y is None:
                continue
            try:
                feats = muni_features(p)
            except Exception as e:  # noqa: BLE001
                print(f"  {muni}: ERROR {e}")
                continue
            if feats is None:
                continue
            feats.update({"muni": muni, "year": year, "yield_t_ha": y})
            rows.append(feats)
            if (i + 1) % 100 == 0:
                print(f"  ... {i + 1}/{len(munis)}")

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "muni_input_features.csv", index=False)
    print(f"\nSaved {len(df)} rows -> {OUT_DIR / 'muni_input_features.csv'}")

    feats = [
        "ndvi_peak", "ndvi_peak_p90", "ndvi_mean", "ndvi_window",
        "rain_cum_end", "dry_streak_max", "valid_days_mean", "t_days",
        "frac_invalid_obs", "frac_window_covered",
    ]

    print("\n=== Spearman correlation with yield_t_ha ===")
    print(f"{'feature':<20}{'all':>8}{'y<3.5':>8}{'y>=3.5':>8}")
    lo = df[df.yield_t_ha < 3.5]
    hi = df[df.yield_t_ha >= 3.5]
    for f in feats:
        c_all = df[f].corr(df.yield_t_ha, method="spearman")
        c_lo = lo[f].corr(lo.yield_t_ha, method="spearman")
        c_hi = hi[f].corr(hi.yield_t_ha, method="spearman")
        print(f"{f:<20}{c_all:>8.3f}{c_lo:>8.3f}{c_hi:>8.3f}")

    print("\n=== Feature means by yield bin ===")
    bins = [0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    df["ybin"] = pd.cut(df.yield_t_ha, bins)
    g = df.groupby("ybin", observed=True)[feats + ["yield_t_ha"]].mean()
    g["n"] = df.groupby("ybin", observed=True).size()
    print(g.round(3).to_string())

    print("\n=== t_days (stored obs dates) per season ===")
    print(df.groupby("year")["t_days"].describe()[["mean", "min", "max"]].round(1).to_string())

    # plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_feats = ["ndvi_peak", "ndvi_window", "ndvi_mean", "rain_cum_end",
                  "dry_streak_max", "valid_days_mean"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, f in zip(axes.ravel(), plot_feats):
        for yr, sub in df.groupby("year"):
            ax.scatter(sub.yield_t_ha, sub[f], s=6, alpha=0.4, label=str(yr))
        ax.set_xlabel("IBGE yield (t/ha)")
        ax.set_ylabel(f)
        ax.axvline(3.6, color="red", ls="--", lw=0.8)
    axes[0, 0].legend(markerscale=2, fontsize=8)
    fig.suptitle("Input features vs actual yield (red line = prediction cap 3.6)")
    fig.tight_layout()
    out = OUT_DIR / "input_features_vs_yield.png"
    fig.savefig(out, dpi=130)
    print(f"\nSaved plot -> {out}")


if __name__ == "__main__":
    sys.exit(main())
