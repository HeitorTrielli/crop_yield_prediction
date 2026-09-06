"""Quantify how much signal the exp(NDVI) temporal pooling destroys.

Simulates what the model's pooled representation "sees" under different
temporal pooling schemes, then runs the same GBM leave-year-out probes:

  expw   : exp(NDVI)-weighted temporal mean (current model pooling)
  unif   : uniform mean over VALID days only (remove exp, keep cloud masking)
  unifall: uniform mean over ALL days incl. zeroed clouds (naive removal)
  bins   : means in 3 season windows (Oct-Nov / Dec-Jan / Feb-Mar)
           = temporal structure preserved (upper bound for seq models)

Features pooled: 10 bands + NDVI, EVI2, NDRE2, NDMI.

Run:  python scripts/debug_pooling_effect.py
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
CACHE = OUT_DIR / "muni_pooling_features.csv"
MAX_PIXELS = 1000

SEASON_TO_YEAR = {
    "2019-2020": 2020,
    "2020-2021": 2021,
    "2021-2022": 2022,
    "2022-2023": 2023,
    "2023-2024": 2024,
}

BIN_EDGES = [(1, 61), (61, 123), (123, 185)]  # season DOY: Oct-Nov, Dec-Jan, Feb-Mar
FEAT_NAMES = [f"B{i}" for i in range(10)] + ["NDVI", "EVI2", "NDRE2", "NDMI"]


def _nr(a, b):
    return (a - b) / (a + b + 1e-8)


def muni_features(path: Path) -> dict | None:
    arr = np.load(path, mmap_mode="r")
    n, t, c = arr.shape
    if n == 0 or c < 13:
        return None
    if n > MAX_PIXELS:
        idx = np.sort(np.random.default_rng(0).choice(n, MAX_PIXELS, replace=False))
        sub = np.asarray(arr[idx])
    else:
        sub = np.asarray(arr)

    doy = sub[:, :, 10]
    spec = sub[:, :, :10].astype(np.float32)
    invalid = (spec == -9999) | ~np.isfinite(spec)
    spec = np.where(invalid, 0.0, spec) * 1e-4  # exactly like PixelTransform

    # exact getWeight_batch logic (datasets/datautils.py)
    score = np.minimum(1.0, (spec[:, :, [0, 1, 2]].sum(axis=2) - 0.2) / 0.6)
    cloud = score * 100 > 20
    dark = spec[:, :, [6, 8, 9]].sum(axis=2) < 0.35
    ndvi_w = _nr(spec[:, :, 6], spec[:, :, 2])
    ndvi_w = np.where(cloud | dark, -1.0, ndvi_w).clip(-1, 1)
    w_exp = np.exp(ndvi_w)
    w_exp = w_exp / w_exp.sum(axis=1, keepdims=True)          # (N, T)

    valid = ~invalid.any(axis=2) & (spec.sum(axis=2) > 0.05)
    w_unif = valid.astype(np.float64)
    w_unif = w_unif / np.maximum(w_unif.sum(axis=1, keepdims=True), 1.0)
    w_all = np.full((sub.shape[0], t), 1.0 / t)

    # feature stack (N, T, F)
    ndvi = _nr(spec[:, :, 6], spec[:, :, 2])
    evi2 = 2.5 * (spec[:, :, 6] - spec[:, :, 2]) / (spec[:, :, 6] + 2.4 * spec[:, :, 2] + 1.0)
    ndre2 = _nr(spec[:, :, 6], spec[:, :, 4])
    ndmi = _nr(spec[:, :, 6], spec[:, :, 8])
    F = np.concatenate([spec, ndvi[..., None], evi2[..., None],
                        ndre2[..., None], ndmi[..., None]], axis=2)  # (N,T,14)

    feats: dict[str, float] = {}

    def pool(weights: np.ndarray, tag: str) -> None:
        # per-pixel weighted temporal mean -> muni mean  (mimics model pooling)
        pooled = (F * weights[..., None]).sum(axis=1)   # (N, 14)
        m = pooled.mean(axis=0)
        for j, name in enumerate(FEAT_NAMES):
            feats[f"{tag}_{name}"] = float(m[j])

    pool(w_exp, "expw")
    pool(w_unif, "unif")
    pool(w_all, "unifall")

    # temporal bins (uniform within each window, valid days only)
    for bi, (lo, hi) in enumerate(BIN_EDGES):
        inwin = valid & (doy >= lo) & (doy < hi)
        wb = inwin.astype(np.float64)
        tot = wb.sum()
        if tot < 1:
            for name in FEAT_NAMES:
                feats[f"bin{bi}_{name}"] = np.nan
            continue
        m = (F * wb[..., None]).sum(axis=(0, 1)) / tot
        for j, name in enumerate(FEAT_NAMES):
            feats[f"bin{bi}_{name}"] = float(m[j])

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
    print(f"{label:<38} R2={r2:.3f}  sp_all={spearmanr(pred, y).statistic:.3f}  "
          f"sp_hi={spearmanr(pred[m], y[m]).statistic:.3f}  AUC mid/high={auc:.3f}")


def main() -> None:
    if CACHE.is_file():
        df = pd.read_csv(CACHE)
        print(f"Loaded cache ({len(df)} rows) from {CACHE}")
    else:
        df = build()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE, index=False)
        print(f"Saved {len(df)} rows -> {CACHE}")

    cols = {tag: [c for c in df.columns if c.startswith(tag + "_")]
            for tag in ("expw", "unif", "unifall")}
    bin_cols = [c for c in df.columns if c.startswith("bin")]

    print("\n=== GBM leave-year-out: pooling scheme comparison ===")
    gbm_probe(df, cols["expw"], "exp(NDVI)-weighted (current model)")
    gbm_probe(df, cols["unif"], "uniform over valid days")
    gbm_probe(df, cols["unifall"], "uniform over all days (naive)")
    gbm_probe(df, bin_cols, "3 temporal bins (structure kept)")
    gbm_probe(df, cols["expw"] + bin_cols, "expw + temporal bins")


if __name__ == "__main__":
    sys.exit(main())
