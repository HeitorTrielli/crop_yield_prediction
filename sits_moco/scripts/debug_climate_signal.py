"""Do the unused climate channels (13-16: ETo/Rs/Tmax/Tmin) add high-yield signal?

Extract per municipality-season climate features from the 17-channel .npy,
merge with cached band features (muni_band_features.csv), and compare GBM
probes with vs without climate features.

Run:  python scripts/debug_climate_signal.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
NPY_ROOT = REPO / "files" / "npy"
OUT_DIR = REPO / "results" / "debug_inputs"
BAND_CSV = OUT_DIR / "muni_band_features.csv"
CACHE = OUT_DIR / "muni_climate_features.csv"

SEASON_TO_YEAR = {
    "2019-2020": 2020,
    "2020-2021": 2021,
    "2021-2022": 2022,
    "2022-2023": 2023,
    "2023-2024": 2024,
}

CH = {"rain": 11, "dry": 12, "eto": 13, "rs": 14, "tmax": 15, "tmin": 16}


def _series(sub: np.ndarray, ch: int) -> tuple[np.ndarray, np.ndarray]:
    """Median over pixels of a cumulative channel -> (doy, values) sorted."""
    v = sub[:, :, ch]
    v = np.where(v == -9999, np.nan, v)
    with np.errstate(all="ignore"):
        med = np.nanmedian(v, axis=0)
    doy = np.nanmedian(np.where(sub[:, :, 10] == -9999, np.nan, sub[:, :, 10]), axis=0)
    ok = np.isfinite(med) & np.isfinite(doy)
    order = np.argsort(doy[ok])
    return doy[ok][order], med[ok][order]


def _rate_in_window(doy: np.ndarray, cum: np.ndarray, lo: float, hi: float) -> float:
    """Daily rate of a cumulative series inside [lo, hi] season DOY."""
    if len(doy) < 2:
        return np.nan
    a = np.interp(lo, doy, cum)
    b = np.interp(hi, doy, cum)
    return float((b - a) / (hi - lo))


def muni_climate(path: Path) -> dict | None:
    arr = np.load(path, mmap_mode="r")
    n, t, c = arr.shape
    if n == 0 or c < 17:
        return None
    take = min(n, 800)
    idx = np.sort(np.random.default_rng(0).choice(n, take, replace=False)) if n > take else slice(None)
    sub = np.asarray(arr[idx])

    feats: dict[str, float] = {}
    series = {k: _series(sub, ch) for k, ch in CH.items()}

    for k in ("rain", "eto", "rs", "tmax", "tmin"):
        doy, cum = series[k]
        if len(cum) == 0:
            return None
        feats[f"{k}_total"] = float(cum[-1])
        # vegetative window (DOY 60-120) and grain-fill window (DOY 120-165)
        feats[f"{k}_rate_veg"] = _rate_in_window(doy, cum, 60, 120)
        feats[f"{k}_rate_fill"] = _rate_in_window(doy, cum, 120, 165)

    doy_d, dry = series["dry"]
    feats["dry_max"] = float(np.nanmax(dry)) if len(dry) else np.nan
    m = (doy_d >= 120) & (doy_d <= 165)
    feats["dry_max_fill"] = float(np.nanmax(dry[m])) if m.any() else np.nan

    # derived water balance / stress
    feats["water_balance"] = feats["rain_total"] - feats["eto_total"]
    feats["rain_eto_ratio"] = feats["rain_total"] / max(feats["eto_total"], 1e-6)
    feats["wb_rate_fill"] = feats["rain_rate_fill"] - feats["eto_rate_fill"]
    feats["diurnal_range"] = feats["tmax_total"] - feats["tmin_total"]
    return feats


def build() -> pd.DataFrame:
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
            try:
                feats = muni_climate(p)
            except Exception as e:  # noqa: BLE001
                print(f"  {muni}: ERROR {e}", flush=True)
                continue
            if feats is None:
                continue
            feats.update({"muni": muni, "year": year})
            rows.append(feats)
    return pd.DataFrame(rows)


def gbm_probe(df: pd.DataFrame, feat_cols: list[str], label: str) -> None:
    from scipy.stats import spearmanr
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict

    X = df[feat_cols].fillna(0.0).to_numpy()
    y = df["yield_t_ha"].to_numpy()
    groups = df["year"].to_numpy()
    gkf = GroupKFold(n_splits=5)

    gbm = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, random_state=0)
    pred = cross_val_predict(gbm, X, y, cv=gkf, groups=groups)
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    m = y >= 3.5
    mid = (y >= 3.0) & (y < 3.5)
    top = y >= 4.0
    sel = mid | top
    clf = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                     random_state=0)
    proba = cross_val_predict(clf, X[sel], top[sel].astype(int),
                              cv=GroupKFold(n_splits=5), groups=groups[sel],
                              method="predict_proba")[:, 1]
    auc = roc_auc_score(top[sel].astype(int), proba)
    print(f"{label:<28} R2={r2:.3f}  sp_all={spearmanr(pred, y).statistic:.3f}  "
          f"sp_hi={spearmanr(pred[m], y[m]).statistic:.3f}  "
          f"pred>4.0: {int((pred > 4.0).sum()):>3} (actual {int((y > 4.0).sum())})  "
          f"AUC mid/high={auc:.3f}")


def main() -> None:
    if CACHE.is_file():
        cdf = pd.read_csv(CACHE)
        print(f"Loaded cache ({len(cdf)} rows) from {CACHE}")
    else:
        cdf = build()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        cdf.to_csv(CACHE, index=False)
        print(f"Saved {len(cdf)} rows -> {CACHE}")

    bdf = pd.read_csv(BAND_CSV)
    bdf["muni"] = bdf["muni"].astype(str)
    cdf["muni"] = cdf["muni"].astype(str)
    df = bdf.merge(cdf, on=["muni", "year"], how="inner")
    print(f"Merged: {len(df)} rows")

    band_feats = [c for c in bdf.columns if c not in ("muni", "year", "yield_t_ha")]
    clim_feats = [c for c in cdf.columns if c not in ("muni", "year")]

    y = df["yield_t_ha"]
    lo, hi = df[y < 3.5], df[y >= 3.5]
    print("\n=== Climate features: Spearman with yield (sorted by |y>=3.5|) ===")
    recs = []
    for f in clim_feats:
        recs.append({
            "feature": f,
            "all": df[f].corr(y, method="spearman"),
            "y<3.5": lo[f].corr(lo.yield_t_ha, method="spearman"),
            "y>=3.5": hi[f].corr(hi.yield_t_ha, method="spearman"),
        })
    cor = pd.DataFrame(recs).set_index("feature")
    cor = cor.reindex(cor["y>=3.5"].abs().sort_values(ascending=False).index)
    print(cor.round(3).to_string())

    print("\n=== GBM leave-year-out probes ===")
    gbm_probe(df, band_feats, "bands+indices only")
    gbm_probe(df, clim_feats, "climate only")
    gbm_probe(df, band_feats + clim_feats, "bands + climate")

    # importances for combined
    from sklearn.ensemble import GradientBoostingRegressor
    gbm = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, random_state=0)
    gbm.fit(df[band_feats + clim_feats].fillna(0.0), y)
    imp = pd.Series(gbm.feature_importances_, index=band_feats + clim_feats)
    print("\nTop 15 importances (bands+climate full fit):")
    print(imp.sort_values(ascending=False).head(15).round(3).to_string())


if __name__ == "__main__":
    sys.exit(main())
