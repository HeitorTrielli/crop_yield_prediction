#!/usr/bin/env python3
"""Tabular year-LOO on the DualSTNet *pixel path*.

Train on municipal-mean tabular features (mean-then-encode), then for each
holdout municipality score every soy pixel and average predictions
(encode-then-mean) — same asymmetry as MSTNet/DualSTNet pixel-path LOO.

Feature modes: indices | raw
Models: ridge, rf, xgboost  (mlp optional)

Examples:
  python tools/debug/run_tabular_year_loo_pixel_path.py --feature-mode indices
  python tools/debug/run_tabular_year_loo_pixel_path.py --feature-mode raw --models ridge,rf,xgboost
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Reuse muni-mean feature defs + model builders
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tabular_year_loo_baseline import (  # noqa: E402
    BAND_NAMES,
    BAND_PERCENTILES,
    BAND_STATS,
    CLIMATE_COLS,
    NO_DATA,
    SEASON,
    SOIL_COLS,
    YEARS,
    _climate_soil,
    feature_cols,
    make_models,
    metrics,
    summarize,
    tune_ridge,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIXEL_NPY = Path.home() / "sits_moco_data" / "npy"
DEFAULT_YIELD = ROOT / "files" / "pam_soy_pr_mapbiomas_coverage_0.5_1.5.csv"
TRAIN_FEAT = {
    "indices": ROOT
    / "loo_pixel_results"
    / "pr_tabular_year_loo_baseline"
    / "muni_year_tabular_features.csv",
    "raw": ROOT
    / "loo_pixel_results"
    / "pr_tabular_raw_bands_year_loo_baseline"
    / "muni_year_tabular_features.csv",
}
OUT_BY_MODE = {
    "indices": ROOT
    / "loo_pixel_results"
    / "pr_tabular_indices_pixel_path_year_loo",
    "raw": ROOT / "loo_pixel_results" / "pr_tabular_raw_bands_pixel_path_year_loo",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pixel-npy-root", type=Path, default=DEFAULT_PIXEL_NPY)
    p.add_argument("--yield-csv", type=Path, default=DEFAULT_YIELD)
    p.add_argument("--train-features-csv", type=Path, default=None)
    p.add_argument(
        "--feature-mode",
        choices=("indices", "raw"),
        default="indices",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--models",
        default="ridge,rf,xgboost",
        help="Comma-separated: ridge,rf,xgboost,mlp",
    )
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--predict-batch",
        type=int,
        default=65536,
        help="Pixel batch size for model.predict",
    )
    p.add_argument(
        "--max-pixels-per-muni",
        type=int,
        default=16384,
        help="Random subsample cap per municipality (0 = all pixels; slow)",
    )
    p.add_argument(
        "--max-munis",
        type=int,
        default=0,
        help="Debug: limit # municipalities per holdout (0 = all)",
    )
    return p.parse_args()


def _norm_code(code) -> str:
    s = str(code).replace(".0", "")
    if s.isdigit():
        return s.zfill(7)
    return s


def _subsample_cube_soil(
    cube: np.ndarray, soil: np.ndarray, max_pixels: int, seed: int
) -> tuple[np.ndarray, np.ndarray, int]:
    n = int(cube.shape[0])
    if max_pixels <= 0 or n <= max_pixels:
        return cube, soil, n
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_pixels, replace=False)
    idx.sort()
    if soil.ndim == 1:
        soil = soil.reshape(1, -1)
    if soil.shape[0] == n:
        soil = soil[idx]
    elif soil.shape[0] == 1:
        pass
    else:
        soil = np.full((max_pixels, soil.shape[-1] if soil.ndim > 1 else 4), np.nan)
    return cube[idx], soil, n


def _nan_percentiles_axis1(col: np.ndarray, percentiles: tuple[int, ...]) -> np.ndarray:
    """Faster stand-in for np.nanpercentile(..., axis=1) on (N, T) with small T."""
    # Sort along time; NaNs sort to the end.
    ordered = np.sort(col, axis=1)
    finite = np.isfinite(col)
    counts = finite.sum(axis=1).astype(np.float64)
    out = np.empty((len(percentiles), col.shape[0]), dtype=np.float64)
    for j, p in enumerate(percentiles):
        # linear index into the finite prefix: (q)*(count-1)
        q = p / 100.0
        pos = q * np.maximum(counts - 1.0, 0.0)
        lo = np.floor(pos).astype(np.int64)
        hi = np.ceil(pos).astype(np.int64)
        # clamp
        lo = np.clip(lo, 0, col.shape[1] - 1)
        hi = np.clip(hi, 0, col.shape[1] - 1)
        w = pos - lo.astype(np.float64)
        row = np.arange(col.shape[0])
        v = (1.0 - w) * ordered[row, lo] + w * ordered[row, hi]
        v[counts == 0] = np.nan
        out[j] = v
    return out


def extract_pixel_feature_matrix(
    cube: np.ndarray,
    clim_path: Path,
    soil: np.ndarray,
    mode: str,
    cols: list[str],
) -> np.ndarray:
    """Return (N_pix, F) float64 feature matrix aligned with ``cols``."""
    if cube.ndim != 3 or cube.shape[0] < 1 or cube.shape[2] < 11:
        raise ValueError(f"bad cube shape {cube.shape}")
    n = int(cube.shape[0])
    spec_raw = cube[:, :, :10].astype(np.float32, copy=False)
    doy = cube[:, :, 10].astype(np.float32, copy=False)

    invalid = (spec_raw == NO_DATA) | (spec_raw == 0) | ~np.isfinite(spec_raw)
    day_bad = invalid.any(axis=2)  # (N, T)
    refl = np.where(day_bad[..., None], np.nan, spec_raw.astype(np.float32) * 1e-4)
    valid = ~day_bad & ((refl[:, :, 2] + refl[:, :, 6]) > 0.01)
    refl = np.where(valid[..., None], refl, np.nan)

    feats: dict[str, np.ndarray] = {
        "n_valid_days": valid.sum(axis=1).astype(np.float64)
    }

    with np.errstate(all="ignore"):
        if mode == "indices":
            red, nir = refl[:, :, 2], refl[:, :, 6]
            ndvi = (nir - red) / (nir + red + 1e-8)
            ndvi = np.where(valid, ndvi, np.nan)
            feats["ndvi_peak"] = np.nanmax(ndvi, axis=1)
            feats["ndvi_mean"] = np.nanmean(ndvi, axis=1)
            window = valid & (doy >= 60) & (doy <= 135)
            w_ndvi = np.where(window, ndvi, np.nan)
            feats["ndvi_window"] = np.nanmean(w_ndvi, axis=1)
        else:
            for i, b in enumerate(BAND_NAMES):
                col = refl[:, :, i]
                feats[f"{b}_mean"] = np.nanmean(col, axis=1)
                feats[f"{b}_peak"] = np.nanmax(col, axis=1)
                pcts = _nan_percentiles_axis1(col, BAND_PERCENTILES)
                for j, p in enumerate(BAND_PERCENTILES):
                    feats[f"{b}_p{p}"] = pcts[j]

    # climate: shared across pixels in the municipality
    clim_feats = _climate_soil(clim_path, Path("__no_soil__"))
    for k in CLIMATE_COLS:
        feats[k] = np.full(n, float(clim_feats.get(k, np.nan)), dtype=np.float64)

    # soil: per-pixel
    if soil.ndim == 1:
        soil = soil.reshape(1, -1)
    if soil.shape[0] == 1 and n > 1:
        soil = np.repeat(soil, n, axis=0)
    if soil.shape[0] != n:
        for k in SOIL_COLS:
            feats[k] = np.full(n, np.nan, dtype=np.float64)
    else:
        soil_f = soil.astype(np.float64, copy=False)
        feats["clay_pct"] = soil_f[:, 0]
        feats["silt_pct"] = soil_f[:, 1]
        feats["sand_pct"] = soil_f[:, 2]
        feats["soc_t_ha"] = (
            soil_f[:, 3] if soil_f.shape[1] > 3 else np.full(n, np.nan)
        )

    mat = np.column_stack([feats[c] for c in cols]).astype(np.float64, copy=False)
    mat[~np.isfinite(mat)] = np.nan
    return mat


def predict_muni_from_pixels(
    pipes: dict,
    pixel_root: Path,
    code: str,
    year: int,
    mode: str,
    cols: list[str],
    batch: int,
    max_pixels: int,
    seed: int,
) -> dict[str, float] | None:
    season = SEASON[year]
    base = pixel_root / season / code
    cube_path = base / f"{code}.npy"
    clim_path = base / f"{code}_climate_daily.npy"
    soil_path = base / f"{code}_soil.npy"
    if not cube_path.is_file():
        return None

    cube = np.load(cube_path)
    if soil_path.is_file():
        soil = np.load(soil_path)
    else:
        soil = np.full((cube.shape[0], 4), np.nan, dtype=np.float32)

    # Deterministic subsample keyed by code+year so folds are stable.
    sub_seed = (seed * 1_000_003 + int(code) * 17 + year) % (2**31 - 1)
    cube, soil, n_total = _subsample_cube_soil(cube, soil, max_pixels, sub_seed)

    X = extract_pixel_feature_matrix(cube, clim_path, soil, mode, cols)
    ok = np.isfinite(X[:, cols.index("n_valid_days")]) & (
        X[:, cols.index("n_valid_days")] > 0
    )
    if not np.any(ok):
        return None
    X = X[ok]

    out: dict[str, float] = {
        "n_pixels": float(X.shape[0]),
        "n_pixels_total": float(n_total),
    }
    for name, pipe in pipes.items():
        preds = np.empty(X.shape[0], dtype=np.float64)
        for start in range(0, X.shape[0], batch):
            sl = slice(start, start + batch)
            preds[sl] = pipe.predict(X[sl])
        out[name] = float(np.nanmean(preds))
    return out


def run_pixel_loo(
    train_df: pd.DataFrame,
    cols: list[str],
    model_names: list[str],
    pixel_root: Path,
    seed: int,
    n_jobs: int,
    out_dir: Path,
    mode: str,
    batch: int,
    max_munis: int,
    max_pixels: int,
) -> pd.DataFrame:
    records: list[dict] = []
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for holdout in YEARS:
        train = train_df[train_df["year"] != holdout]
        test = train_df[train_df["year"] == holdout].copy()
        if max_munis > 0:
            test = test.head(max_munis)
        X_tr = train[cols].to_numpy(dtype=np.float64)
        y_tr = train["yield_t_ha"].to_numpy(dtype=np.float64)
        print(
            f"\nLOO {holdout}: train={len(train)} test_munis={len(test)} "
            f"max_pixels/muni={max_pixels or 'all'}"
        )

        pipes = {}
        for name in model_names:
            pipe = make_models(seed, n_jobs)[name]
            if name == "ridge":
                pipe = tune_ridge(pipe, X_tr, y_tr)
            else:
                pipe.fit(X_tr, y_tr)
            pipes[name] = pipe

        rows = []
        t0 = time.time()
        missing = 0
        for i, r in enumerate(test.itertuples(index=False)):
            code = _norm_code(r.municipality_code)
            got = predict_muni_from_pixels(
                pipes,
                pixel_root,
                code,
                holdout,
                mode,
                cols,
                batch,
                max_pixels,
                seed,
            )
            if got is None:
                missing += 1
                continue
            row = {
                "municipality_code": code,
                "year": holdout,
                "yield_t_ha": float(r.yield_t_ha),
                "n_pixels": int(got["n_pixels"]),
                "n_pixels_total": int(got["n_pixels_total"]),
            }
            for name in model_names:
                row[f"pred_{name}"] = got[name]
            rows.append(row)
            if (i + 1) % 25 == 0 or (i + 1) == len(test):
                elapsed = time.time() - t0
                print(
                    f"  scored {i + 1}/{len(test)} munis "
                    f"({missing} missing npy)  [{elapsed:.0f}s]"
                )

        pred_df = pd.DataFrame(rows)
        if pred_df.empty:
            print(f"  WARNING: no predictions for {holdout}")
            continue
        pred_df.to_csv(pred_dir / f"all_models_loo{holdout}.csv", index=False)

        y_te = pred_df["yield_t_ha"].to_numpy(dtype=np.float64)
        for name in model_names:
            pred = pred_df[f"pred_{name}"].to_numpy(dtype=np.float64)
            m = metrics(y_te, pred)
            print(
                f"  {name:8s}  n={m['n']}  RMSE={m['rmse']:.3f}  "
                f"MAE={m['mae']:.3f}  R2={m['r2']:.3f}"
            )
            records.append({"model": name, "loo_year": holdout, **m})
            one = pred_df[
                ["municipality_code", "year", "yield_t_ha", "n_pixels", "n_pixels_total"]
            ].copy()
            one["pred_t_ha"] = pred
            one.to_csv(pred_dir / f"{name}_loo{holdout}.csv", index=False)

    return pd.DataFrame(records)


def main() -> int:
    args = parse_args()
    mode = args.feature_mode
    cols = feature_cols(mode)
    out_dir = args.output_dir or OUT_BY_MODE[mode]
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_csv = args.train_features_csv or TRAIN_FEAT[mode]
    if not feat_csv.is_file():
        print(
            f"ERROR: train features CSV missing: {feat_csv}\n"
            f"Run run_tabular_year_loo_baseline.py --feature-mode {mode} first.",
            file=sys.stderr,
        )
        return 1
    if not args.pixel_npy_root.is_dir():
        print(f"ERROR: pixel npy root not found: {args.pixel_npy_root}", file=sys.stderr)
        return 1

    train_df = pd.read_csv(feat_csv)
    train_df["municipality_code"] = train_df["municipality_code"].map(_norm_code)
    train_df["year"] = train_df["year"].astype(int)
    missing_cols = [c for c in cols if c not in train_df.columns]
    if missing_cols:
        print(f"ERROR: train CSV missing columns: {missing_cols[:8]}", file=sys.stderr)
        return 1

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    available = set(make_models(args.seed, args.n_jobs))
    model_names = [m for m in model_names if m in available]
    if not model_names:
        print("No models to run", file=sys.stderr)
        return 1

    print(
        f"Pixel-path tabular LOO  mode={mode}  models={model_names}\n"
        f"  train features: {feat_csv}  (n={len(train_df)})\n"
        f"  pixel npy:      {args.pixel_npy_root}\n"
        f"  max_pixels/muni:{args.max_pixels_per_muni or 'all'}\n"
        f"  out:            {out_dir}"
    )

    metrics_df = run_pixel_loo(
        train_df,
        cols,
        model_names,
        args.pixel_npy_root,
        args.seed,
        args.n_jobs,
        out_dir,
        mode,
        args.predict_batch,
        args.max_munis,
        args.max_pixels_per_muni,
    )
    if metrics_df.empty:
        print("No metrics produced", file=sys.stderr)
        return 1

    full = summarize(metrics_df)
    out_csv = out_dir / "year_loo_metrics.csv"
    full.to_csv(out_csv, index=False)
    print(f"\n=== Summary ({mode}, pixel path) → {out_csv} ===")
    show = full[full["loo_year"].astype(str).isin(["mean", "mean_excl_2022"])]
    print(show.to_string(index=False))

    meta = {
        "eval_path": "pixel_by_pixel",
        "train_path": "muni_mean_tabular_features",
        "feature_mode": mode,
        "features": cols,
        "models": model_names,
        "train_features_csv": str(feat_csv),
        "pixel_npy_root": str(args.pixel_npy_root),
        "yield_csv": str(args.yield_csv),
        "n_train_rows": int(len(train_df)),
        "max_pixels_per_muni": int(args.max_pixels_per_muni),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
