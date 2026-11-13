"""Test whether RAW bands / non-NDVI indices carry high-yield signal.

Per municipality-season: mean reflectance of each S2 band over valid days in
the peak vegetative window (season DOY 60-135), plus derived indices and rain.
Then: Spearman correlations (all / low / high yield range) and a multivariate
GBM probe to see if any combination separates mid from high yields.

Band order (S2): 0=B02 1=B03 2=B04(red) 3=B05(RE1) 4=B06(RE2) 5=B07(RE3)
6=B08(NIR) 7=B8A 8=B11(SWIR1) 9=B12(SWIR2)

Run:  python scripts/debug_band_signal.py
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

BAND_NAMES = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

SEASON_TO_YEAR = {
    "2019-2020": 2020,
    "2020-2021": 2021,
    "2021-2022": 2022,
    "2022-2023": 2023,
    "2023-2024": 2024,
}


def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / (a + b + 1e-8)


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
    spec = np.where(invalid, 0.0, spec) * 1e-4  # same as PixelTransform

    valid_day = ~invalid.any(axis=2) & (spec.sum(axis=2) > 0.05)
    window = valid_day & (doy >= 60) & (doy <= 135)
    if not window.any():
        window = valid_day  # fallback: whole season
    w = window[:, :, None].astype(np.float32)
    wsum = w.sum(axis=(0, 1))
    if wsum[0] < 1:
        return None

    # municipality-level mean reflectance per band in window
    band_mean = (spec * w).sum(axis=(0, 1)) / np.maximum(wsum, 1.0)

    feats = {f"{BAND_NAMES[b]}_win": float(band_mean[b]) for b in range(10)}

    # indices computed per observation then averaged over window
    red, re1, re2, re3 = spec[:, :, 2], spec[:, :, 3], spec[:, :, 4], spec[:, :, 5]
    nir, nirn, swir1, swir2 = spec[:, :, 6], spec[:, :, 7], spec[:, :, 8], spec[:, :, 9]
    green = spec[:, :, 1]

    idx_map = {
        "NDVI": _safe_ratio(nir, red),
        "kNDVI": np.tanh(_safe_ratio(nir, red) ** 2),
        "EVI2": 2.5 * (nir - red) / (nir + 2.4 * red + 1.0),
        "NDRE1": _safe_ratio(nir, re1),
        "NDRE2": _safe_ratio(nir, re2),
        "CIRE": re3 / (re1 + 1e-8) - 1.0,
        "GNDVI": _safe_ratio(nir, green),
        "NDMI": _safe_ratio(nir, swir1),   # canopy moisture (B08 vs B11)
        "NBR2like": _safe_ratio(nirn, swir2),
        "NIRv": _safe_ratio(nir, red) * nir,
    }
    wmask = window.astype(np.float32)
    wtot = max(float(wmask.sum()), 1.0)
    for name, v in idx_map.items():
        feats[f"{name}_win"] = float((v * wmask).sum() / wtot)
        # p90 across pixels of per-pixel window mean: high-end sensitivity
        with np.errstate(all="ignore"):
            px = np.where(window, v, np.nan)
            px_mean = np.nanmean(px, axis=1)
        feats[f"{name}_p90"] = float(np.nanpercentile(px_mean, 90))

    rain = sub[:, :, 11]
    dry = sub[:, :, 12]
    rv = rain != -9999
    feats["rain_cum_end"] = float(np.nanmean(np.nanmax(np.where(rv, rain, np.nan), axis=1)))
    feats["dry_streak_max"] = float(np.nanmean(np.nanmax(np.where(rv, dry, np.nan), axis=1)))
    feats["valid_days_mean"] = float(valid_day.sum(axis=1).mean())
    return feats


def build_dataframe() -> pd.DataFrame:
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
            continue
        munis = sorted(os.listdir(sdir))
        print(f"{season}: {len(munis)} municipalities", flush=True)
        for muni in munis:
            p = sdir / muni / f"{muni}.npy"
            if not p.is_file():
                continue
            y = ylookup.get((muni, year))
            if y is None:
                continue
            try:
                feats = muni_features(p)
            except Exception as e:  # noqa: BLE001
                print(f"  {muni}: ERROR {e}", flush=True)
                continue
            if feats is None:
                continue
            feats.update({"muni": muni, "year": year, "yield_t_ha": y})
            rows.append(feats)
    return pd.DataFrame(rows)


def main() -> None:
    cache = OUT_DIR / "muni_band_features.csv"
    if cache.is_file():
        df = pd.read_csv(cache)
        print(f"Loaded cache ({len(df)} rows) from {cache}")
    else:
        df = build_dataframe()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        print(f"Saved {len(df)} rows -> {cache}")

    y = df["yield_t_ha"]
    feat_cols = [c for c in df.columns if c not in ("muni", "year", "yield_t_ha")]
    lo, hi = df[y < 3.5], df[y >= 3.5]

    print("\n=== Spearman correlation with yield (sorted by |y>=3.5|) ===")
    recs = []
    for f in feat_cols:
        recs.append({
            "feature": f,
            "all": df[f].corr(y, method="spearman"),
            "y<3.5": lo[f].corr(lo.yield_t_ha, method="spearman"),
            "y>=3.5": hi[f].corr(hi.yield_t_ha, method="spearman"),
        })
    cor = pd.DataFrame(recs).set_index("feature")
    cor = cor.reindex(cor["y>=3.5"].abs().sort_values(ascending=False).index)
    print(cor.round(3).to_string())

    # ---- multivariate probe -------------------------------------------------
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from scipy.stats import spearmanr

    X = df[feat_cols].fillna(0.0).to_numpy()
    groups = df["year"].to_numpy()  # leave-year-out CV: honest generalization
    gkf = GroupKFold(n_splits=5)

    print("\n=== GBM leave-year-out CV: predict yield from ALL input features ===")
    gbm = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, random_state=0)
    pred = cross_val_predict(gbm, X, y, cv=gkf, groups=groups)
    resid = y - pred
    r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"all rows:   R2={r2:.3f}  spearman={spearmanr(pred, y).statistic:.3f}"
          f"  pred_max={pred.max():.2f}  actual_max={y.max():.2f}")
    m = y >= 3.5
    print(f"y>=3.5:     spearman={spearmanr(pred[m], y[m]).statistic:.3f}"
          f"  (n={int(m.sum())})")
    n_above = int((pred > 4.0).sum())
    print(f"GBM predictions above 4.0: {n_above}  (actual: {int((y > 4.0).sum())})")

    print("\n=== GBM classifier: 3.0-3.5 vs >=4.0 (can inputs separate at all?) ===")
    mid = (y >= 3.0) & (y < 3.5)
    top = y >= 4.0
    sel = mid | top
    Xc, yc, gc = X[sel], top[sel].astype(int), groups[sel]
    clf = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                     random_state=0)
    from sklearn.metrics import roc_auc_score
    proba = cross_val_predict(clf, Xc, yc, cv=GroupKFold(n_splits=5), groups=gc,
                              method="predict_proba")[:, 1]
    print(f"AUC (mid vs high): {roc_auc_score(yc, proba):.3f}"
          f"  n_mid={int(mid.sum())} n_high={int(top.sum())}")

    # feature importances from a full fit (context only)
    gbm.fit(X, y)
    imp = pd.Series(gbm.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print("\nTop 15 GBM feature importances (full fit):")
    print(imp.head(15).round(3).to_string())


if __name__ == "__main__":
    sys.exit(main())
