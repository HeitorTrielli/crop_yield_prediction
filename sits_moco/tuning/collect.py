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
            result["training_config"] = {
                "model_kwargs": (sessions[-1].get("computed") or {}).get("model_kwargs"),
                "cli": sessions[-1].get("cli"),
            }

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
        if isinstance(v, (list, tuple)):
            row[f"param_{k}"] = ",".join(str(x) for x in v)
        else:
            row[f"param_{k}"] = v

    row["objective"] = outcome.get("objective")
    row["objective_value"] = outcome.get("objective_value")
    row["best_epoch"] = outcome.get("best_epoch")
    row["run_dir"] = outcome.get("run_dir")

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

    return row
