"""Early stopping state derived from trainlog.csv (source of truth on resume)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def early_stop_state_from_trainlog(trainlog_path: Path | str) -> dict | None:
    """
    Compute early-stop counters from completed validation epochs in trainlog.csv.

    Returns dict with keys: val_loss_min, best_epoch, last_epoch, not_improved_count.
    ``not_improved_count`` = last_epoch - best_epoch (epochs since the best val loss).
    """
    path = Path(trainlog_path)
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or "valloss" not in df.columns or "epoch" not in df.columns:
        return None

    df = df.copy()
    df = df[df["epoch"].astype(str) != "test"]
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["valloss"] = pd.to_numeric(df["valloss"], errors="coerce")
    df = df.dropna(subset=["epoch", "valloss"])
    if df.empty:
        return None

    df["epoch"] = df["epoch"].astype(int)
    best_idx = df["valloss"].idxmin()
    best_epoch = int(df.loc[best_idx, "epoch"])
    val_loss_min = float(df.loc[best_idx, "valloss"])
    last_epoch = int(df["epoch"].max())
    not_improved_count = last_epoch - best_epoch
    return {
        "val_loss_min": val_loss_min,
        "best_epoch": best_epoch,
        "last_epoch": last_epoch,
        "not_improved_count": not_improved_count,
    }


def initial_session_optimizer_hparams(training_dir: Path | str) -> dict | None:
    """Base lr/weight_decay from the first training session in config.json."""
    from run_paths import load_training_sessions

    training_dir = Path(training_dir)
    config_path = training_dir / "config.json"
    sessions = load_training_sessions(config_path)
    if not sessions:
        return None
    cli = sessions[0].get("cli") or {}
    if "learning_rate" not in cli:
        return None
    return {
        "learning_rate": float(cli["learning_rate"]),
        "weight_decay": float(cli.get("weight_decay", 0.0)),
    }
