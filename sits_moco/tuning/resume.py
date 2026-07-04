"""Detect partial training runs and pick a checkpoint to resume from."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from run_paths import training_dir
from training.early_stop import early_stop_state_from_trainlog

_CHECKPOINT_EPOCH_RE = re.compile(r"^checkpoint_epoch_(\d+)\.pth$")


def find_resume_checkpoint(run_dir: Path | str) -> Path | None:
    """
    Return the latest resumable checkpoint under ``{run_dir}/training/``.

    Prefers ``checkpoint_epoch_{N}.pth`` (highest N), then ``model_best.pth``.
    """
    td = training_dir(run_dir)
    if not td.is_dir():
        return None

    epoch_ckpts: list[tuple[int, Path]] = []
    for path in td.glob("checkpoint_epoch_*.pth"):
        match = _CHECKPOINT_EPOCH_RE.match(path.name)
        if match:
            epoch_ckpts.append((int(match.group(1)), path))

    if epoch_ckpts:
        return max(epoch_ckpts, key=lambda item: item[0])[1]

    best = td / "model_best.pth"
    if best.is_file():
        return best
    return None


def _checkpoint_epoch(checkpoint: Path) -> int | None:
    try:
        import torch

        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "epoch" in data:
            return int(data["epoch"])
    except Exception:
        pass

    match = _CHECKPOINT_EPOCH_RE.match(checkpoint.name)
    if match:
        return int(match.group(1))
    if checkpoint.name == "model_best.pth":
        return _trainlog_last_epoch(training_dir(checkpoint.parent))
    return None


def _trainlog_last_epoch(td: Path) -> int | None:
    trainlog = td / "trainlog.csv"
    if not trainlog.is_file():
        return None
    try:
        df = pd.read_csv(trainlog)
        if df.empty or "epoch" not in df.columns:
            return None
        epochs = pd.to_numeric(df["epoch"], errors="coerce").dropna()
        if epochs.empty:
            return None
        return int(epochs.max())
    except Exception:
        return None


def plan_trial_resume(
    run_dir: Path | str,
    *,
    target_epochs: int,
    early_stop_patience: int | None = None,
) -> dict[str, Any] | None:
    """
    If training can continue from a checkpoint, return resume metadata.

    Returns None when there is no checkpoint, the saved epoch already reached
    ``target_epochs``, or early-stop patience is already exhausted in trainlog.
    """
    run_dir = Path(run_dir)
    td = training_dir(run_dir)

    if early_stop_patience is not None:
        es = early_stop_state_from_trainlog(td / "trainlog.csv")
        if es is not None and es["not_improved_count"] >= int(early_stop_patience):
            return None

    checkpoint = find_resume_checkpoint(run_dir)
    if checkpoint is None:
        return None

    epoch = _checkpoint_epoch(checkpoint)
    if epoch is not None and epoch >= int(target_epochs):
        return None

    trainlog_epoch = _trainlog_last_epoch(td)
    es = early_stop_state_from_trainlog(td / "trainlog.csv")
    return {
        "checkpoint": checkpoint.resolve(),
        "epoch": epoch,
        "trainlog_last_epoch": trainlog_epoch,
        "target_epochs": int(target_epochs),
        "early_stop": es,
    }
