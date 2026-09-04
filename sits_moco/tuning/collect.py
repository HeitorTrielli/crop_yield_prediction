"""Extract metrics from completed training runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from run_paths import trainlog_path


def _read_log_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    if "epoch" in df.columns:
        df = df.set_index("epoch")
    return df


def _trainlog_epoch_stats(run_dir: Path) -> dict[str, Any]:
    """Last trained epoch and early-stop checkpoint epoch from trainlog.csv."""
    df = _read_log_csv(trainlog_path(run_dir))
    if df is None or df.empty:
        return {}
    epochs = pd.Series(pd.to_numeric(df.index, errors="coerce"), index=df.index)
    mask = epochs.notna()
    if not mask.any():
        return {}
    out: dict[str, Any] = {"n_epochs": int(epochs[mask].max())}
    col = "valloss" if "valloss" in df.columns else ("r2" if "r2" in df.columns else None)
    if col is None:
        return out
    values = pd.to_numeric(df[col], errors="coerce")
    ok = mask & values.notna()
    if not ok.any():
        return out
    subset = values[ok]
    raw_idx = subset.idxmin() if col == "valloss" else subset.idxmax()
    try:
        out["best_epoch"] = str(int(pd.to_numeric(raw_idx)))
    except (TypeError, ValueError):
        out["best_epoch"] = str(raw_idx)
    return out


def format_run_epochs(outcome: dict, *, mean: bool = False) -> str:
    """Suffix like ' (11 epochs)' or ' (211 epochs, best 201)' — never 'epoch test'."""
    n = outcome.get("n_epochs")
    best = outcome.get("best_epoch")
    n_f: float | None = None
    if n is not None:
        try:
            n_f = float(n)
        except (TypeError, ValueError):
            n_f = None
    b_i: int | None = None
    if best is not None:
        try:
            b_i = int(best)
        except (TypeError, ValueError):
            b_i = None
    if n_f is None:
        return ""
    n_label = f"{n_f:.1f}" if abs(n_f - round(n_f)) > 1e-9 else str(int(round(n_f)))
    prefix = "mean " if mean else ""
    if not mean and b_i is not None and abs(b_i - n_f) > 1e-9:
        return f" ({prefix}{n_label} epochs, best {b_i})"
    return f" ({prefix}{n_label} epochs)"


def best_epoch_metrics(
    run_dir: Path | str,
    *,
    objective: str = "val_r2",
    objective_spec: dict | None = None,
) -> dict[str, Any]:
    """
    Read trainlog.csv (and optional testlog.csv) and return best-epoch summary.

    Returns dict with keys: status, best_epoch, objective_value, metrics, run_dir, ...
    """
    run_dir = Path(run_dir)
    spec = objective_spec or {}
    column = spec.get("column", "r2")
    mode = spec.get("mode", "max")
    log_name = spec.get("log", "trainlog")

    if log_name == "testlog":
        log_file = run_dir / "training" / "testlog.csv"
    else:
        log_file = trainlog_path(run_dir)

    df = _read_log_csv(log_file)
    if df is None or column not in df.columns:
        return {
            "status": "missing_log",
            "run_dir": str(run_dir.resolve()),
            "log_file": str(log_file),
            "objective": objective,
            "objective_column": column,
        }

    numeric = pd.to_numeric(df[column], errors="coerce")
    valid = numeric.notna()
    if not valid.any():
        return {
            "status": "no_valid_metric",
            "run_dir": str(run_dir.resolve()),
            "log_file": str(log_file),
            "objective": objective,
        }

    series = numeric[valid]
    idx = series.idxmin() if mode == "min" else series.idxmax()
    row = df.loc[idx]
    metrics = {str(k): float(v) if pd.notna(v) else None for k, v in row.items()}

    result: dict[str, Any] = {
        "status": "ok",
        "run_dir": str(run_dir.resolve()),
        "log_file": str(log_file.resolve()),
        "objective": objective,
        "objective_column": column,
        "objective_mode": mode,
        "best_epoch": str(idx),
        "objective_value": float(series.loc[idx]),
        "metrics": metrics,
    }
    train_stats = _trainlog_epoch_stats(run_dir)
    if train_stats.get("n_epochs") is not None:
        result["n_epochs"] = train_stats["n_epochs"]
    if train_stats.get("best_epoch") is not None:
        result["best_epoch"] = train_stats["best_epoch"]
    elif str(idx).lower() == "test":
        result.pop("best_epoch", None)

    testlog = run_dir / "training" / "testlog.csv"
    test_df = _read_log_csv(testlog)
    if test_df is not None:
        test_row = test_df.iloc[0]
        result["test_metrics"] = {
            str(k): float(v) if pd.notna(v) else None for k, v in test_row.items()
        }

    config_path = run_dir / "training" / "config.json"
    if config_path.is_file():
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        sessions = cfg.get("sessions") or []
        if sessions:
            computed = (sessions[-1].get("computed") or {})
            result["training_config"] = {
                "model_kwargs": computed.get("model_kwargs"),
                "cli": sessions[-1].get("cli"),
                "chunk_pipeline": computed.get("chunk_pipeline"),
            }
            chunk_pipeline = computed.get("chunk_pipeline") or {}
            if chunk_pipeline:
                result["chunk_pipeline"] = chunk_pipeline

    return result


_VAL_METRIC_KEYS = ("rmse", "mae", "r2", "mape", "trainloss", "valloss")
_TEST_METRIC_KEYS = ("rmse", "mae", "r2", "mape", "testloss")


def _mean_numeric(values: list[Any]) -> float | None:
    nums: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if pd.isna(number):
            continue
        nums.append(number)
    if not nums:
        return None
    return float(sum(nums) / len(nums))


def aggregate_year_loo_folds(
    folds: list[dict[str, Any]],
    *,
    objective: str,
    trial_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Mean val/test metrics across successful year-LOO folds of one trial."""
    ok = [f for f in folds if f.get("status") == "ok"]
    skipped = [f for f in folds if f.get("status") == "skipped"]
    failed = [f for f in folds if f.get("status") not in ("ok", "skipped")]

    metrics: dict[str, Any] = {}
    test_metrics: dict[str, Any] = {}
    if ok:
        for key in _VAL_METRIC_KEYS:
            metrics[key] = _mean_numeric(
                [(f.get("metrics") or {}).get(key) for f in ok]
            )
        for key in _TEST_METRIC_KEYS:
            test_metrics[key] = _mean_numeric(
                [(f.get("test_metrics") or {}).get(key) for f in ok]
            )

    year_metrics: dict[str, Any] = {}
    for fold in folds:
        year = fold.get("holdout_year")
        if year is None:
            continue
        year_metrics[f"val_r2_{int(year)}"] = (fold.get("metrics") or {}).get("r2")
        year_metrics[f"test_r2_{int(year)}"] = (fold.get("test_metrics") or {}).get(
            "r2"
        )
        year_metrics[f"n_epochs_{int(year)}"] = fold.get("n_epochs")

    if failed:
        status = "train_failed"
        returncode = int(failed[0].get("returncode") or 1)
        error = failed[0].get("error") or f"{len(failed)} fold(s) failed"
    elif not ok:
        status = "no_folds"
        returncode = 1
        error = "no successful year-LOO folds"
    else:
        status = "ok"
        returncode = 0
        error = None

    result: dict[str, Any] = {
        "status": status,
        "returncode": returncode,
        "objective": objective,
        "objective_value": _mean_numeric([f.get("objective_value") for f in ok]),
        "n_epochs": _mean_numeric([f.get("n_epochs") for f in ok]),
        "metrics": metrics,
        "test_metrics": test_metrics,
        "n_folds": len(ok),
        "n_folds_skipped": len(skipped),
        "n_folds_failed": len(failed),
        "folds": folds,
        "year_metrics": year_metrics,
    }
    if trial_dir is not None:
        result["run_dir"] = str(Path(trial_dir).resolve())
    if error:
        result["error"] = error
    return result


def trial_row(
    trial_id: str,
    params: dict,
    outcome: dict,
    *,
    started_at_utc: str | None = None,
    finished_at_utc: str | None = None,
) -> dict[str, Any]:
    """Flatten trial params + outcome into one CSV-friendly row."""
    row: dict[str, Any] = {"trial_id": trial_id, "status": outcome.get("status")}
    if started_at_utc is not None:
        row["started_at_utc"] = started_at_utc
    if finished_at_utc is not None:
        row["finished_at_utc"] = finished_at_utc
    if started_at_utc and finished_at_utc:
        try:
            start_dt = pd.Timestamp(started_at_utc)
            finish_dt = pd.Timestamp(finished_at_utc)
            row["duration_seconds"] = max(0.0, (finish_dt - start_dt).total_seconds())
        except (TypeError, ValueError):
            pass
    for k, v in sorted(params.items()):
        if k == "year_loo_folds":
            years = []
            for fold in v or []:
                if isinstance(fold, dict) and fold.get("holdout_year") is not None:
                    years.append(str(int(fold["holdout_year"])))
            if years:
                row["param_year_loo"] = ",".join(years)
            continue
        if isinstance(v, dict):
            continue
        if isinstance(v, (list, tuple)):
            if v and isinstance(v[0], dict):
                continue
            row[f"param_{k}"] = ",".join(str(x) for x in v)
        else:
            row[f"param_{k}"] = v

    row["objective"] = outcome.get("objective")
    row["objective_value"] = outcome.get("objective_value")
    row["n_epochs"] = outcome.get("n_epochs")
    row["best_epoch"] = outcome.get("best_epoch")
    row["run_dir"] = outcome.get("run_dir")
    if outcome.get("n_folds") is not None:
        row["n_folds"] = outcome.get("n_folds")
        row["n_folds_skipped"] = outcome.get("n_folds_skipped")
        row["n_folds_failed"] = outcome.get("n_folds_failed")
    for key, value in (outcome.get("year_metrics") or {}).items():
        row[key] = value

    metrics = outcome.get("metrics") or {}
    for mk in ("rmse", "mae", "r2", "mape", "trainloss", "valloss"):
        if mk in metrics:
            row[f"val_{mk}"] = metrics[mk]

    test_metrics = outcome.get("test_metrics") or {}
    for mk in ("rmse", "mae", "r2", "mape", "testloss"):
        if mk in test_metrics:
            row[f"test_{mk}"] = test_metrics[mk]

    if outcome.get("error"):
        row["error"] = outcome["error"]
    if outcome.get("returncode") is not None:
        row["returncode"] = outcome["returncode"]

    chunk_pipeline = outcome.get("chunk_pipeline") or {}
    if chunk_pipeline:
        row["chunks_per_grad_requested"] = chunk_pipeline.get(
            "chunks_per_grad_requested"
        )
        row["resolved_chunks_per_grad"] = chunk_pipeline.get(
            "chunks_per_grad_effective"
        )
        row["pixel_chunk_size_requested"] = chunk_pipeline.get(
            "pixel_chunk_size_requested"
        )
        row["resolved_pixel_chunk_size"] = chunk_pipeline.get(
            "pixel_chunk_size_effective"
        )
        row["max_pixels_per_backward_requested"] = chunk_pipeline.get(
            "max_pixels_per_backward_requested"
        )
        row["resolved_max_pixels_per_backward"] = chunk_pipeline.get(
            "max_pixels_per_backward_effective"
        )
        row["auto_chunks_per_grad_search_trials"] = len(
            chunk_pipeline.get("search_trials") or []
        )

    return row
