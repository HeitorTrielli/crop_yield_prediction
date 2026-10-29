#!/usr/bin/env python3
"""Tabular year-LOO baselines vs MSTNet (PR, coverage 0.5–1.5).

Feature modes:
  indices — NDVI peak/mean/window + climate + soil
  raw     — per-band mean/peak/percentiles (B2–B12) + climate + soil (no VIs)

Models: ridge, rf, xgboost, mlp

Examples:
  python tools/debug/run_tabular_year_loo_baseline.py --feature-mode indices
  python tools/debug/run_tabular_year_loo_baseline.py --feature-mode raw --models mlp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPY = Path.home() / "sits_moco_data" / "npy_muni_mean_soy"
DEFAULT_YIELD = ROOT / "files" / "pam_soy_pr_mapbiomas_coverage_0.5_1.5.csv"
OUT_BY_MODE = {
    "indices": ROOT / "loo_pixel_results" / "pr_tabular_year_loo_baseline",
    "raw": ROOT / "loo_pixel_results" / "pr_tabular_raw_bands_year_loo_baseline",
}

NO_DATA = -9999.0
YEARS = (2020, 2021, 2022, 2023, 2024)
SEASON = {
    2020: "2019-2020",
    2021: "2020-2021",
    2022: "2021-2022",
    2023: "2022-2023",
    2024: "2023-2024",
}
BAND_NAMES = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")
BAND_STATS = ("mean", "peak", "p10", "p25", "p50", "p75", "p90")
BAND_PERCENTILES = (10, 25, 50, 75, 90)
CLIMATE_COLS = [
    "rain_sum_mm",
    "rain_dec_feb_mm",
    "eto_sum_mm",
    "tmax_mean",
    "tmin_mean",
    "rs_mean",
    "dry_spell_max",
]
SOIL_COLS = ["clay_pct", "silt_pct", "sand_pct", "soc_t_ha"]
INDICES_SPECTRAL = ["ndvi_peak", "ndvi_mean", "ndvi_window", "n_valid_days"]
RAW_SPECTRAL = [f"{b}_{s}" for b in BAND_NAMES for s in BAND_STATS] + ["n_valid_days"]


def feature_cols(mode: str) -> list[str]:
    spec = INDICES_SPECTRAL if mode == "indices" else RAW_SPECTRAL
    return list(spec) + CLIMATE_COLS + SOIL_COLS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npy-root", type=Path, default=DEFAULT_NPY)
    p.add_argument("--yield-csv", type=Path, default=DEFAULT_YIELD)
    p.add_argument(
        "--feature-mode",
        choices=("indices", "raw"),
        default="raw",
        help="indices = NDVI summaries; raw = band mean/peak/percentiles",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default depends on --feature-mode",
    )
    p.add_argument(
        "--models",
        default="ridge,rf,xgboost,mlp",
        help="Comma-separated: ridge,rf,xgboost,mlp",
    )
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rebuild-features", action="store_true")
    return p.parse_args()


def _raw_reflectance(spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    invalid = (spec == NO_DATA) | (spec == 0) | ~np.isfinite(spec)
    day_bad = invalid.any(axis=1)
    refl = np.where(day_bad[:, None], np.nan, spec.astype(np.float64) * 1e-4)
    valid = ~day_bad & ((refl[:, 2] + refl[:, 6]) > 0.01)
    refl = np.where(valid[:, None], refl, np.nan)
    return refl, valid


def _climate_soil(
    clim_path: Path, soil_path: Path
) -> dict[str, float]:
    feats: dict[str, float] = {}
    if clim_path.is_file():
        clim = np.load(clim_path).astype(np.float32)
        if clim.ndim == 3:
            clim = clim[0]
        ok = np.isfinite(clim).all(axis=1) & (clim != NO_DATA).all(axis=1)
        c = np.where(ok[:, None], clim, np.nan)
        pr, eto, rs, tmax, tmin = c.T[:5]
        dec_feb = slice(61, 151)
        with np.errstate(all="ignore"):
            feats["rain_sum_mm"] = float(np.nansum(pr))
            feats["rain_dec_feb_mm"] = float(np.nansum(pr[dec_feb]))
            feats["eto_sum_mm"] = float(np.nansum(eto))
            feats["tmax_mean"] = float(np.nanmean(tmax))
            feats["tmin_mean"] = float(np.nanmean(tmin))
            feats["rs_mean"] = float(np.nanmean(rs))
            dry = (pr < 1.0) & np.isfinite(pr)
            run = best = 0
            for d in dry:
                if d:
                    run += 1
                    best = max(best, run)
                else:
                    run = 0
            feats["dry_spell_max"] = float(best)
    else:
        for k in CLIMATE_COLS:
            feats[k] = np.nan

    if soil_path.is_file():
        soil = np.load(soil_path).astype(np.float32).reshape(-1)
        feats["clay_pct"] = float(soil[0])
        feats["silt_pct"] = float(soil[1])
        feats["sand_pct"] = float(soil[2])
        feats["soc_t_ha"] = float(soil[3]) if soil.size > 3 else np.nan
    else:
        for k in SOIL_COLS:
            feats[k] = np.nan
    return feats


def extract_features(
    npy_root: Path, code: str, year: int, mode: str
) -> dict[str, float] | None:
    season = SEASON[year]
    base = npy_root / season / code
    cube_path = base / f"{code}.npy"
    clim_path = base / f"{code}_climate_daily.npy"
    soil_path = base / f"{code}_soil.npy"
    if not cube_path.is_file():
        return None

    cube = np.load(cube_path)
    if cube.ndim != 3 or cube.shape[0] < 1 or cube.shape[2] < 11:
        return None
    arr = cube[0]
    doy = arr[:, 10]
    refl, valid = _raw_reflectance(arr[:, :10].astype(np.float32))
    feats: dict[str, float] = {"n_valid_days": float(valid.sum())}

    with np.errstate(all="ignore"):
        if mode == "indices":
            red, nir = refl[:, 2], refl[:, 6]
            ndvi = (nir - red) / (nir + red + 1e-8)
            ndvi = np.where(valid, ndvi, np.nan)
            if np.any(valid):
                feats["ndvi_peak"] = float(np.nanmax(ndvi))
                feats["ndvi_mean"] = float(np.nanmean(ndvi))
                window = valid & (doy >= 60) & (doy <= 135)
                feats["ndvi_window"] = float(
                    np.nanmean(np.where(window, ndvi, np.nan))
                )
            else:
                feats["ndvi_peak"] = feats["ndvi_mean"] = feats["ndvi_window"] = np.nan
        else:
            for i, b in enumerate(BAND_NAMES):
                col = refl[:, i]
                if not np.any(valid):
                    for stat in BAND_STATS:
                        feats[f"{b}_{stat}"] = np.nan
                    continue
                feats[f"{b}_mean"] = float(np.nanmean(col))
                feats[f"{b}_peak"] = float(np.nanmax(col))
                pcts = np.nanpercentile(col, BAND_PERCENTILES)
                for p, v in zip(BAND_PERCENTILES, pcts):
                    feats[f"{b}_p{p}"] = float(v)

    feats.update(_climate_soil(clim_path, soil_path))
    return feats


def build_feature_table(
    npy_root: Path, yield_csv: Path, mode: str
) -> pd.DataFrame:
    ydf = pd.read_csv(yield_csv)
    ydf["municipality_code"] = (
        ydf["municipality_code"].astype(str).str.replace(r"\.0$", "", regex=True)
    )
    ydf["year"] = ydf["year"].astype(int)
    ydf = ydf[ydf["year"].isin(YEARS)].copy()

    rows: list[dict] = []
    missing = 0
    for i, r in enumerate(ydf.itertuples(index=False)):
        code = str(r.municipality_code)
        year = int(r.year)
        feats = extract_features(npy_root, code, year, mode)
        if feats is None:
            missing += 1
            continue
        rows.append(
            {
                "municipality_code": code,
                "year": year,
                "yield_t_ha": float(r.yield_t_ha),
                **feats,
            }
        )
        if (i + 1) % 200 == 0:
            print(f"  features {i + 1}/{len(ydf)} (missing npy so far: {missing})")
    print(f"Built {len(rows)} feature rows; skipped {missing} without npy")
    return pd.DataFrame(rows)


def make_models(seed: int, n_jobs: int) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {
        "ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge()),
            ]
        ),
        "rf": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        min_samples_leaf=2,
                        random_state=seed,
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        ),
        "mlp": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        solver="adam",
                        alpha=1e-2,
                        learning_rate_init=5e-4,
                        max_iter=1000,
                        early_stopping=True,
                        validation_fraction=0.2,
                        n_iter_no_change=40,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }
    try:
        from xgboost import XGBRegressor

        models["xgboost"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=400,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="reg:squarederror",
                        random_state=seed,
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        )
    except ImportError:
        print("xgboost not installed; skipping xgboost model")
    return models


def tune_ridge(pipe: Pipeline, X: np.ndarray, y: np.ndarray) -> Pipeline:
    if len(y) < 40:
        pipe.set_params(model__alpha=1.0)
        pipe.fit(X, y)
        return pipe
    grid = GridSearchCV(
        pipe,
        {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
        cv=4,
        scoring="neg_mean_squared_error",
        n_jobs=1,
    )
    grid.fit(X, y)
    return grid.best_estimator_


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y_true)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_loo_collect(
    df: pd.DataFrame,
    cols: list[str],
    model_names: list[str],
    seed: int,
    n_jobs: int,
    out_dir: Path,
) -> pd.DataFrame:
    records: list[dict] = []
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for holdout in YEARS:
        train = df[df["year"] != holdout]
        test = df[df["year"] == holdout]
        X_tr = train[cols].to_numpy(dtype=np.float64)
        y_tr = train["yield_t_ha"].to_numpy(dtype=np.float64)
        X_te = test[cols].to_numpy(dtype=np.float64)
        y_te = test["yield_t_ha"].to_numpy(dtype=np.float64)
        print(f"\nLOO {holdout}: train={len(train)} test={len(test)}")

        for name in model_names:
            pipe = make_models(seed, n_jobs)[name]
            if name == "ridge":
                pipe = tune_ridge(pipe, X_tr, y_tr)
            else:
                pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_te)
            m = metrics(y_te, pred)
            print(
                f"  {name:8s}  n={m['n']}  RMSE={m['rmse']:.3f}  "
                f"MAE={m['mae']:.3f}  R2={m['r2']:.3f}"
            )
            records.append({"model": name, "loo_year": holdout, **m})
            pred_df = test[["municipality_code", "year", "yield_t_ha"]].copy()
            pred_df["pred_t_ha"] = pred
            pred_df.to_csv(pred_dir / f"{name}_loo{holdout}.csv", index=False)

    return pd.DataFrame(records)


def _fold_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["loo_year"] = out["loo_year"].astype(str)
    out = out[~out["loo_year"].isin(["mean", "mean_excl_2022"])].copy()
    out["loo_year_int"] = out["loo_year"].astype(int)
    return out


def summarize(metrics_df: pd.DataFrame) -> pd.DataFrame:
    fold = _fold_rows(metrics_df)
    rows = []
    for model, g in fold.groupby("model"):
        rows.append(
            {
                "model": model,
                "loo_year": "mean",
                "n": int(g["n"].mean()),
                "rmse": float(g["rmse"].mean()),
                "mae": float(g["mae"].mean()),
                "r2": float(g["r2"].mean()),
            }
        )
        g_ex = g[g["loo_year_int"] != 2022]
        rows.append(
            {
                "model": model,
                "loo_year": "mean_excl_2022",
                "n": int(g_ex["n"].mean()),
                "rmse": float(g_ex["rmse"].mean()),
                "mae": float(g_ex["mae"].mean()),
                "r2": float(g_ex["r2"].mean()),
            }
        )
    fold_out = fold.drop(columns=["loo_year_int"])
    fold_out["loo_year"] = fold_out["loo_year"].astype(int)
    return pd.concat([fold_out, pd.DataFrame(rows)], ignore_index=True)


def merge_metrics(out_dir: Path, new_df: pd.DataFrame) -> pd.DataFrame:
    """Keep previous models' fold rows; replace overlapping model names."""
    path = out_dir / "year_loo_metrics.csv"
    new_models = set(new_df["model"].unique())
    fold_new = _fold_rows(new_df).drop(columns=["loo_year_int"])
    if path.is_file():
        old = _fold_rows(pd.read_csv(path))
        old = old[~old["model"].isin(new_models)].drop(columns=["loo_year_int"])
        combined = pd.concat([old, fold_new], ignore_index=True)
    else:
        combined = fold_new
    return summarize(combined)


def main() -> int:
    args = parse_args()
    mode = args.feature_mode
    cols = feature_cols(mode)
    out_dir = args.output_dir or OUT_BY_MODE[mode]
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_path = out_dir / "muni_year_tabular_features.csv"

    if feat_path.is_file() and not args.rebuild_features:
        print(f"Loading cached features: {feat_path}")
        df = pd.read_csv(feat_path)
        df["municipality_code"] = df["municipality_code"].astype(str)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"Cached features missing columns {missing[:5]}...; rebuilding")
            args.rebuild_features = True
    if args.rebuild_features or not feat_path.is_file():
        if not args.npy_root.is_dir():
            print(f"ERROR: npy root not found: {args.npy_root}", file=sys.stderr)
            return 1
        print(f"Building {mode} features from {args.npy_root}")
        df = build_feature_table(args.npy_root, args.yield_csv, mode)
        df.to_csv(feat_path, index=False)
        print(f"Wrote {feat_path}")

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    available = set(make_models(args.seed, args.n_jobs))
    model_names = [m for m in model_names if m in available]
    if not model_names:
        print("No models to run", file=sys.stderr)
        return 1

    print(f"Mode={mode}  models={model_names}  n_features={len(cols)}")
    metrics_df = run_loo_collect(
        df, cols, model_names, args.seed, args.n_jobs, out_dir
    )
    full = merge_metrics(out_dir, metrics_df)
    out_csv = out_dir / "year_loo_metrics.csv"
    full.to_csv(out_csv, index=False)
    print(f"\n=== Summary ({mode}) → {out_csv} ===")
    show = full[full["loo_year"].astype(str).isin(["mean", "mean_excl_2022"])]
    print(show.to_string(index=False))

    ref = {
        "reference": "MSTNet municipal-mean year-LOO (paper Table)",
        "mstnet": {
            "mean": {"rmse": 0.617, "mae": 0.483, "r2": 0.160},
            "mean_excl_2022": {"rmse": 0.435, "mae": 0.336, "r2": 0.275},
        },
        "feature_mode": mode,
        "features": cols,
        "n_rows": int(len(df)),
        "yield_csv": str(args.yield_csv),
        "npy_root": str(args.npy_root),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(ref, indent=2))
    print("\nMSTNet paper reference (muni-mean LOO):")
    print("  mean          RMSE=0.617  MAE=0.483  R2=0.160")
    print("  mean_excl_2022 RMSE=0.435  MAE=0.336  R2=0.275")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
