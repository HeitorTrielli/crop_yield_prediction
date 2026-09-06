"""Wide sweep: many band indices + temporal shape + GDD/climate features.

Tests which pre-calculated variables carry signal for HIGH yields (>=3.5 t/ha),
where the current inputs saturate. Adds to the previous probes:
- ~25 extra spectral indices (red-edge, SAVI family, SWIR/tillage, chlorophyll)
- temporal shape of the muni-median NDVI/EVI2 curve (green-up slope, peak DOY,
  AUC, senescence slope, season length)
- growing degree days  GDD = (Tmax+Tmin)/2 - 12  (soybean base temp per user)
  from cumulative tmax/tmin channels, plus photothermal quotient Rs/GDD.

Run:  python scripts/debug_extended_signal.py
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
CACHE = OUT_DIR / "muni_extended_features.csv"
MAX_PIXELS = 1000
T_BASE = 12.0  # soybean base temperature (user-specified)

SEASON_TO_YEAR = {
    "2019-2020": 2020,
    "2020-2021": 2021,
    "2021-2022": 2022,
    "2022-2023": 2023,
    "2023-2024": 2024,
}

WIN_LO, WIN_HI = 60, 135      # peak vegetative window (season DOY)
FILL_LO, FILL_HI = 120, 165   # grain-fill window


def _nr(a, b):
    return (a - b) / (a + b + 1e-8)


def spectral_indices(spec: np.ndarray) -> dict[str, np.ndarray]:
    """spec: (N,T,10) reflectance. Band order B02..B12."""
    B02, B03, B04, B05, B06, B07 = (spec[:, :, i] for i in range(6))
    B08, B8A, B11, B12 = (spec[:, :, i] for i in range(6, 10))
    eps = 1e-8
    out = {
        "NDVI": _nr(B08, B04),
        "EVI2": 2.5 * (B08 - B04) / (B08 + 2.4 * B04 + 1.0),
        "SAVI": 1.5 * (B08 - B04) / (B08 + B04 + 0.5),
        "MSAVI": (2 * B08 + 1 - np.sqrt(np.maximum((2 * B08 + 1) ** 2 - 8 * (B08 - B04), 0))) / 2,
        "WDRVI": (0.2 * B08 - B04) / (0.2 * B08 + B04 + eps),
        "GNDVI": _nr(B08, B03),
        "GCI": B08 / (B03 + eps) - 1,
        "RECI": B08 / (B05 + eps) - 1,
        "NDRE1": _nr(B08, B05),
        "NDRE2": _nr(B08, B06),
        "NDRE3": _nr(B08, B07),
        "CIRE": B07 / (B05 + eps) - 1,
        "MTCI": (B06 - B05) / (B05 - B04 + eps),
        "IRECI": (B07 - B04) * B06 / (B05 + eps),
        "S2REP": 705 + 35 * ((B04 + B07) / 2 - B05) / (B06 - B05 + eps),
        "MCARI": ((B05 - B04) - 0.2 * (B05 - B03)) * (B05 / (B04 + eps)),
        "ARVI": _nr(B08, 2 * B04 - B02),
        "VARI": (B03 - B04) / (B03 + B04 - B02 + eps),
        "CVI": B08 * B04 / (B03 ** 2 + eps),
        "PSRI": (B04 - B03) / (B06 + eps),
        "NDMI": _nr(B08, B11),
        "NMDI": (B08 - (B11 - B12)) / (B08 + (B11 - B12) + eps),
        "NDTI": _nr(B11, B12),
        "STI": B11 / (B12 + eps),
        "BSI": ((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02) + eps),
        "NIRv": _nr(B08, B04) * B08,
        "Brightness": np.sqrt((spec ** 2).mean(axis=2)),
        "R_B08_B11": B08 / (B11 + eps),
        "R_B12_B04": B12 / (B04 + eps),
        "R_B8A_B04": B8A / (B04 + eps),
    }
    return out


def _median_series(vals: np.ndarray, valid: np.ndarray, doy: np.ndarray):
    """Muni-level median curve over pixels -> (doy_sorted, median_vals)."""
    v = np.where(valid, vals, np.nan)
    with np.errstate(all="ignore"):
        med = np.nanmedian(v, axis=0)
    d = np.nanmedian(np.where(valid, doy, np.nan), axis=0)
    ok = np.isfinite(med) & np.isfinite(d)
    order = np.argsort(d[ok])
    return d[ok][order], med[ok][order]


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    x = x - x.mean()
    denom = (x ** 2).sum()
    return float((x * (y - y.mean())).sum() / denom) if denom > 0 else np.nan


def temporal_shape(doy: np.ndarray, ndvi: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    if len(doy) < 3:
        return {k: np.nan for k in
                ("greenup_slope", "peak_doy", "peak_val", "senescence_slope",
                 "ndvi_auc", "season_len", "ndvi_fill_mean")}
    i_peak = int(np.argmax(ndvi))
    out["peak_doy"] = float(doy[i_peak])
    out["peak_val"] = float(ndvi[i_peak])
    up = doy <= doy[i_peak]
    down = doy >= doy[i_peak]
    out["greenup_slope"] = _slope(doy[up], ndvi[up])
    out["senescence_slope"] = _slope(doy[down], ndvi[down])
    out["ndvi_auc"] = float(np.trapezoid(ndvi, doy))
    half = 0.5 * ndvi[i_peak]
    above = doy[ndvi >= half]
    out["season_len"] = float(above.max() - above.min()) if len(above) else np.nan
    m = (doy >= FILL_LO) & (doy <= FILL_HI)
    out["ndvi_fill_mean"] = float(ndvi[m].mean()) if m.any() else np.nan
    return out


def _cum_series(sub, ch):
    v = np.where(sub[:, :, ch] == -9999, np.nan, sub[:, :, ch])
    with np.errstate(all="ignore"):
        med = np.nanmedian(v, axis=0)
    d = np.nanmedian(np.where(sub[:, :, 10] == -9999, np.nan, sub[:, :, 10]), axis=0)
    ok = np.isfinite(med) & np.isfinite(d)
    order = np.argsort(d[ok])
    return d[ok][order], med[ok][order]


def _rate(doy, cum, lo, hi):
    if len(doy) < 2:
        return np.nan
    return float((np.interp(hi, doy, cum) - np.interp(lo, doy, cum)) / (hi - lo))


def muni_features(path: Path) -> dict | None:
    arr = np.load(path, mmap_mode="r")
    n, t, c = arr.shape
    if n == 0 or c < 17:
        return None
    if n > MAX_PIXELS:
        idx = np.sort(np.random.default_rng(0).choice(n, MAX_PIXELS, replace=False))
        sub = np.asarray(arr[idx])
    else:
        sub = np.asarray(arr)

    doy = sub[:, :, 10]
    spec = sub[:, :, :10].astype(np.float32)
    invalid = (spec == -9999) | ~np.isfinite(spec)
    spec = np.where(invalid, 0.0, spec) * 1e-4
    valid = ~invalid.any(axis=2) & (spec.sum(axis=2) > 0.05)
    window = valid & (doy >= WIN_LO) & (doy <= WIN_HI)
    if not window.any():
        window = valid
    wmask = window.astype(np.float32)
    wtot = max(float(wmask.sum()), 1.0)

    feats: dict[str, float] = {}
    idxs = spectral_indices(spec)
    for name, v in idxs.items():
        feats[f"{name}_win"] = float((np.nan_to_num(v) * wmask).sum() / wtot)

    # temporal shape from muni-median NDVI curve
    d_nd, ndvi_med = _median_series(idxs["NDVI"], valid, doy)
    feats.update(temporal_shape(d_nd, ndvi_med))

    # climate: GDD etc.
    d_tx, tx = _cum_series(sub, 15)   # tmax cumulative
    d_tn, tn = _cum_series(sub, 16)   # tmin cumulative
    d_rs, rs = _cum_series(sub, 14)
    d_rn, rn = _cum_series(sub, 11)
    if len(tx) and len(tn):
        # GDD_cum(t) = (tmax_cum + tmin_cum)/2 - T_BASE * days_elapsed
        common = np.union1d(d_tx, d_tn)
        txi = np.interp(common, d_tx, tx)
        tni = np.interp(common, d_tn, tn)
        gdd = (txi + tni) / 2.0 - T_BASE * common
        feats["gdd_total"] = float(gdd[-1])
        feats["gdd_rate_veg"] = _rate(common, gdd, WIN_LO, FILL_LO)
        feats["gdd_rate_fill"] = _rate(common, gdd, FILL_LO, FILL_HI)
        if len(rs):
            rs_fill = _rate(d_rs, rs, FILL_LO, FILL_HI)
            gdd_fill = feats["gdd_rate_fill"]
            feats["ptq_fill"] = rs_fill / gdd_fill if gdd_fill and gdd_fill > 0 else np.nan
            rs_veg = _rate(d_rs, rs, WIN_LO, FILL_LO)
            gdd_veg = feats["gdd_rate_veg"]
            feats["ptq_veg"] = rs_veg / gdd_veg if gdd_veg and gdd_veg > 0 else np.nan
        # diurnal amplitude proxy per day
        feats["tamp_rate_veg"] = _rate(common, txi - tni, WIN_LO, FILL_LO)
    if len(rn):
        feats["rain_total"] = float(rn[-1])
        feats["rain_rate_fill"] = _rate(d_rn, rn, FILL_LO, FILL_HI)

    feats["valid_days_mean"] = float(valid.sum(axis=1).mean())
    return feats


def build() -> pd.DataFrame:
    ydf = pd.read_csv(YIELD_CSV)
    ydf["municipality_code"] = ydf["municipality_code"].astype(str)
    ylookup = {(str(r.municipality_code), int(r.year)): float(r.yield_t_ha)
               for r in ydf.itertuples()}
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


def gbm_probe(df: pd.DataFrame, feat_cols: list[str], label: str) -> None:
    from scipy.stats import spearmanr
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict

    X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    y = df["yield_t_ha"].to_numpy()
    groups = df["year"].to_numpy()
    gbm = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, random_state=0)
    pred = cross_val_predict(gbm, X, y, cv=GroupKFold(n_splits=5), groups=groups)
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
    print(f"{label:<30} R2={r2:.3f}  sp_all={spearmanr(pred, y).statistic:.3f}  "
          f"sp_hi={spearmanr(pred[m], y[m]).statistic:.3f}  "
          f"pred>4.0: {int((pred > 4.0).sum()):>3} (actual {int((y > 4.0).sum())})  "
          f"AUC mid/high={auc:.3f}")


def main() -> None:
    if CACHE.is_file():
        df = pd.read_csv(CACHE)
        print(f"Loaded cache ({len(df)} rows) from {CACHE}")
    else:
        df = build()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE, index=False)
        print(f"Saved {len(df)} rows -> {CACHE}")

    y = df["yield_t_ha"]
    feat_cols = [c for c in df.columns if c not in ("muni", "year", "yield_t_ha")]
    lo, hi = df[y < 3.5], df[y >= 3.5]

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
    print("\n=== Spearman with yield, TOP 25 by |y>=3.5| ===")
    print(cor.head(25).round(3).to_string())

    spectral = [c for c in feat_cols if c.endswith("_win")]
    temporal = ["greenup_slope", "peak_doy", "peak_val", "senescence_slope",
                "ndvi_auc", "season_len", "ndvi_fill_mean"]
    climate = [c for c in feat_cols if c.startswith(("gdd", "ptq", "tamp", "rain"))]

    print("\n=== GBM leave-year-out probes ===")
    gbm_probe(df, spectral, "30 spectral indices")
    gbm_probe(df, temporal, "temporal shape only")
    gbm_probe(df, climate, "GDD/PTQ/rain only")
    gbm_probe(df, spectral + temporal, "spectral + temporal")
    gbm_probe(df, feat_cols, "everything")


if __name__ == "__main__":
    sys.exit(main())
