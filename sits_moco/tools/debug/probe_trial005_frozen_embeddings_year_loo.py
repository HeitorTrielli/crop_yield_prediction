#!/usr/bin/env python3
"""Leakage-safe year-LOO probes on frozen trial_005 fused embeddings.

For each holdout year, the script uses that year's LOO encoder to embed both
the four training years and the held-out year. Probe preprocessing, model
selection, and fitting use only the four training years.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.debug.analyze_trial005_encoder_vs_decoder import (  # noqa: E402
    EXP,
    FEAT_CSV,
    TRIAL,
    build_dataset_for_fold,
)
from training.muni_agg_batch import (  # noqa: E402
    load_stacked_muni_agg_batch,
    muni_agg_on_device,
)

YEARS = tuple(range(2020, 2025))
DEFAULT_OUT = EXP / "frozen_embedding_probes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--models", default="ridge,xgboost")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def encode_stacked(model, stacked, device, t_mean: float, t_std: float):
    x = muni_agg_on_device(stacked, device, argparse.Namespace(h2d_pin_host=False))
    autocast_enabled = device.type == "cuda"
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled
    ):
        x_s2, mask_s2, doy_s2, w_s2, x_c, mask_c, doy_c, w_c = x
        x_spec, soil = model._split_soil(x_s2, mask_s2)
        parts = [
            model.spectral(x_spec, mask_s2, doy_s2, w_s2),
            model.climate(x_c, mask_c, doy_c, w_c),
        ]
        if soil is not None:
            parts.append(soil)
        fused = torch.cat(parts, dim=-1)
        pred = model.decoder(fused).reshape(-1)
    return (
        fused.float().cpu().numpy(),
        (pred.float().cpu().numpy() * float(t_std)) + float(t_mean),
    )


def embedding_columns(width: int) -> list[str]:
    return [f"embedding_{i:03d}" for i in range(width)]


def build_fold_cache(
    fold: int,
    labels: pd.DataFrame,
    output_dir: Path,
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
    rebuild: bool,
) -> pd.DataFrame:
    cache_path = output_dir / f"trial005_loo{fold}_fused_embeddings.parquet"
    if cache_path.is_file() and not rebuild:
        print(f"LOO {fold}: loading {cache_path}")
        return pd.read_parquet(cache_path)

    ckpt = TRIAL / f"loo{fold}" / "training" / "model_best.pth"
    print(f"LOO {fold}: encoding {len(labels)} municipality-years with {ckpt}")
    model, dataset, t_mean, t_std, _ = build_dataset_for_fold(ckpt, device)
    model.eval()
    rows: list[pd.DataFrame] = []
    for start in range(0, len(labels), batch_size):
        source = labels.iloc[start : start + batch_size].reset_index(drop=True)
        stacked, kept = load_stacked_muni_agg_batch(
            dataset,
            source["municipality_code"].tolist(),
            source["harvest_year"].tolist(),
            workers=workers,
        )
        if stacked is None or not kept:
            continue
        fused, decoder_pred = encode_stacked(
            model, stacked, device, t_mean=t_mean, t_std=t_std
        )
        metadata = source.iloc[kept][
            ["municipality_code", "municipality_name", "harvest_year", "actual"]
        ].reset_index(drop=True)
        encoded = pd.DataFrame(fused, columns=embedding_columns(fused.shape[1]))
        metadata["decoder_prediction"] = decoder_pred
        metadata["encoder_fold"] = int(fold)
        rows.append(pd.concat([metadata, encoded], axis=1))
        print(f"  {min(start + batch_size, len(labels))}/{len(labels)}", flush=True)

    result = pd.concat(rows, ignore_index=True)
    result.to_parquet(cache_path, index=False)
    del model, dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def probe_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    rho = float(spearmanr(actual, predicted).statistic)
    actual_std = float(np.std(actual))
    pred_std = float(np.std(predicted))
    slope = float(np.polyfit(actual, predicted, 1)[0])
    return {
        "r2": float(r2_score(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "spearman": rho,
        "spearman_r2": rho * rho,
        "prediction_std": pred_std,
        "actual_std": actual_std,
        "std_ratio": pred_std / actual_std if actual_std > 0 else float("nan"),
        "calibration_slope": slope,
    }


def make_probe(name: str, n_jobs: int):
    if name == "ridge":
        estimator = Pipeline(
            [("scale", StandardScaler()), ("model", Ridge(max_iter=10000))]
        )
        grid = {"model__alpha": np.logspace(-4, 4, 17)}
        return estimator, grid
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError(
                "XGBoost is not installed. Install the probe extra or run --models ridge."
            ) from exc
        estimator = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=101010,
            n_jobs=max(1, n_jobs),
        )
        grid = {
            "n_estimators": [150, 400],
            "max_depth": [2, 3],
            "learning_rate": [0.03, 0.08],
            "reg_lambda": [1.0, 10.0],
        }
        return estimator, grid
    raise ValueError(f"Unknown probe model {name!r}")


def fit_probe(name: str, train: pd.DataFrame, feature_cols: list[str], n_jobs: int):
    estimator, grid = make_probe(name, n_jobs)
    inner_cv = GroupKFold(n_splits=train["harvest_year"].nunique())
    search = GridSearchCV(
        estimator,
        grid,
        scoring="r2",
        cv=inner_cv,
        n_jobs=max(1, n_jobs),
        refit=True,
        error_score="raise",
    )
    search.fit(
        train[feature_cols].to_numpy(dtype=np.float32),
        train["actual"].to_numpy(dtype=np.float32),
        groups=train["harvest_year"].to_numpy(),
    )
    return clone(search.best_estimator_).fit(
        train[feature_cols].to_numpy(dtype=np.float32),
        train["actual"].to_numpy(dtype=np.float32),
    ), search.best_params_, float(search.best_score_)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    labels = pd.read_csv(FEAT_CSV)
    labels = labels.rename(columns={"actual_productivity_t_ha": "actual"})
    labels["municipality_code"] = labels["municipality_code"].astype(str)
    labels["harvest_year"] = labels["harvest_year"].astype(int)
    labels = labels[labels["harvest_year"].isin(YEARS)].reset_index(drop=True)

    caches = {
        fold: build_fold_cache(
            fold,
            labels,
            output_dir,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
            rebuild=args.rebuild_cache,
        )
        for fold in YEARS
    }

    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    summaries: list[dict] = []
    predictions: list[pd.DataFrame] = []
    for fold, encoded in caches.items():
        feature_cols = [col for col in encoded if col.startswith("embedding_")]
        train = encoded[encoded["harvest_year"] != fold].copy()
        test = encoded[encoded["harvest_year"] == fold].copy()

        baseline = probe_metrics(test["actual"], test["decoder_prediction"])
        summaries.append({"fold": fold, "model": "neural_decoder", **baseline})
        baseline_rows = test[
            ["municipality_code", "municipality_name", "harvest_year", "actual"]
        ].copy()
        baseline_rows["model"] = "neural_decoder"
        baseline_rows["prediction"] = test["decoder_prediction"].to_numpy()
        predictions.append(baseline_rows)

        for name in model_names:
            model, best_params, inner_r2 = fit_probe(
                name, train, feature_cols, args.n_jobs
            )
            predicted = model.predict(test[feature_cols].to_numpy(dtype=np.float32))
            metrics = probe_metrics(test["actual"], predicted)
            summaries.append(
                {
                    "fold": fold,
                    "model": name,
                    "inner_cv_r2": inner_r2,
                    "best_params": json.dumps(best_params, sort_keys=True),
                    **metrics,
                }
            )
            pred_rows = test[
                ["municipality_code", "municipality_name", "harvest_year", "actual"]
            ].copy()
            pred_rows["model"] = name
            pred_rows["prediction"] = predicted
            predictions.append(pred_rows)
            print(
                f"LOO {fold} {name}: R2={metrics['r2']:.4f}, "
                f"rho2={metrics['spearman_r2']:.4f}, "
                f"std_ratio={metrics['std_ratio']:.3f}"
            )

    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "probe_summary_by_year.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        output_dir / "probe_predictions.csv", index=False
    )
    aggregate = (
        summary.groupby("model", as_index=False)
        .agg(
            mean_r2=("r2", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_spearman_r2=("spearman_r2", "mean"),
            mean_std_ratio=("std_ratio", "mean"),
            mean_calibration_slope=("calibration_slope", "mean"),
        )
        .sort_values("mean_r2", ascending=False)
    )
    aggregate.to_csv(output_dir / "probe_summary.csv", index=False)
    print("\n" + aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
