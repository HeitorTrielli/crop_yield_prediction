"""
Standard directory layout for model training runs.

Layout::

    {logdir}/{experiment_name}/{feature_layout}/
        training/      model_best.pth, checkpoint_epoch_*.pth, trainlog.csv, config.json
        predictions/   CSV outputs from predict_yield.py and related scripts
        figures/
            intramunicipal_heatmap/   pixel-level yield / NDVI heatmaps
            municipal_heatmaps/       state/municipality choropleth maps

Legacy runs (before feature-layout subfolders) may still live at::

    {logdir}/{experiment_name}/training/ ...

``run_dir_from_path`` resolves both layouts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets.feature_layout import normalize_feature_layout

TRAINING_SUBDIR = "training"
FIGURES_SUBDIR = "figures"
PREDICTIONS_SUBDIR = "predictions"
INTRAMUNICIPAL_HEATMAP_SUBDIR = "intramunicipal_heatmap"
MUNICIPAL_HEATMAPS_SUBDIR = "municipal_heatmaps"
_RUN_SUBDIRS = frozenset({TRAINING_SUBDIR, FIGURES_SUBDIR, PREDICTIONS_SUBDIR})


def experiment_dir(logdir: Path | str, experiment_name: str) -> Path:
    """Top-level run folder shared across feature layouts (e.g. Productivity_..._Seed27)."""
    return Path(logdir) / experiment_name


def run_dir_for_experiment(
    logdir: Path | str,
    experiment_name: str,
    feature_layout: str,
) -> Path:
    """Layout root: {logdir}/{experiment_name}/{feature_layout}/."""
    layout = normalize_feature_layout(feature_layout)
    return experiment_dir(logdir, experiment_name) / layout


def run_dir_from_path(path: Path | str) -> Path:
    """Resolve the layout root from a checkpoint path or any file inside a run."""
    p = Path(path).resolve()
    if p.is_file():
        p = p.parent
    elif p.suffix == ".pth":
        # Checkpoint path may not exist yet during dry-runs / path planning.
        p = p.parent
    if p.name == TRAINING_SUBDIR:
        return p.parent
    if p.name in (INTRAMUNICIPAL_HEATMAP_SUBDIR, MUNICIPAL_HEATMAPS_SUBDIR):
        return p.parent.parent
    if p.name in (FIGURES_SUBDIR, PREDICTIONS_SUBDIR):
        return p.parent
    if p.parent.name in (INTRAMUNICIPAL_HEATMAP_SUBDIR, MUNICIPAL_HEATMAPS_SUBDIR):
        return p.parent.parent.parent
    # Legacy: checkpoint passed as experiment_dir/training/checkpoint.pth
    if p.name in _RUN_SUBDIRS:
        return p.parent
    return p


def training_dir(run_dir: Path | str, *, create: bool = False) -> Path:
    run_dir = Path(run_dir)
    td = run_dir / TRAINING_SUBDIR
    if create:
        td.mkdir(parents=True, exist_ok=True)
    return td


def figures_dir(run_dir: Path | str, *, create: bool = False) -> Path:
    run_dir = Path(run_dir)
    fd = run_dir / FIGURES_SUBDIR
    if create:
        fd.mkdir(parents=True, exist_ok=True)
    return fd


def intramunicipal_heatmap_dir(run_dir: Path | str, *, create: bool = False) -> Path:
    run_dir = Path(run_dir)
    d = run_dir / FIGURES_SUBDIR / INTRAMUNICIPAL_HEATMAP_SUBDIR
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def municipal_heatmaps_dir(run_dir: Path | str, *, create: bool = False) -> Path:
    run_dir = Path(run_dir)
    d = run_dir / FIGURES_SUBDIR / MUNICIPAL_HEATMAPS_SUBDIR
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def predictions_dir(run_dir: Path | str, *, create: bool = False) -> Path:
    run_dir = Path(run_dir)
    pd = run_dir / PREDICTIONS_SUBDIR
    if create:
        pd.mkdir(parents=True, exist_ok=True)
    return pd


def ensure_run_layout(run_dir: Path | str) -> tuple[Path, Path, Path]:
    """Create training/, figures/, and predictions/ under the run root."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return (
        training_dir(run_dir, create=True),
        figures_dir(run_dir, create=True),
        predictions_dir(run_dir, create=True),
    )


def model_best_in_run(run_dir: Path | str) -> Path:
    """Path to model_best.pth under training/."""
    run_dir = Path(run_dir)
    return run_dir / TRAINING_SUBDIR / "model_best.pth"


def resolve_checkpoint_path(path: Path | str) -> Path:
    """Resolve model_best.pth from a .pth file or a run directory."""
    p = Path(path)
    if p.is_file():
        return p.resolve()

    if p.is_dir():
        if p.name == TRAINING_SUBDIR:
            candidate = p / "model_best.pth"
        else:
            candidate = p / TRAINING_SUBDIR / "model_best.pth"
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Checkpoint not found at {path!r}. "
        f"Pass a .pth file or a run directory with training/model_best.pth."
    )


def trainlog_path(run_dir: Path | str) -> Path:
    """Path to trainlog.csv under training/."""
    return Path(run_dir) / TRAINING_SUBDIR / "trainlog.csv"


def run_dir_from_forecasts_csv(csv_path: Path | str) -> Path | None:
    """If CSV lives under {run}/predictions/, return the run root."""
    csv_path = Path(csv_path).resolve()
    if csv_path.parent.name == PREDICTIONS_SUBDIR:
        return csv_path.parent.parent
    return None


def training_config_path(run_dir: Path | str) -> Path:
    """Path to training/config.json under a run root."""
    return training_dir(run_dir) / "config.json"


def load_training_sessions(config_path: Path) -> list[dict[str, Any]]:
    """Load session records from training/config.json (supports legacy single-object format)."""
    if not config_path.is_file():
        return []
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "sessions" in data:
        return list(data["sessions"])
    if isinstance(data, dict) and "cli" in data:
        return [
            {
                "session_index": 0,
                "kind": "initial",
                "cli": data.get("cli"),
                "computed": data.get("computed"),
                "environment": data.get("environment"),
                "resume": None,
            }
        ]
    return []


def load_latest_run_config(checkpoint_or_run_dir: Path | str) -> dict[str, Any]:
    """
    Return the latest training session from training/config.json for a run.

    Keys: run_dir, config_path, session_index, cli, computed.
    """
    run_dir = run_dir_from_path(checkpoint_or_run_dir)
    config_path = training_config_path(run_dir)
    sessions = load_training_sessions(config_path)
    if not sessions:
        raise FileNotFoundError(
            f"No training config at {config_path}. "
            "Pass a checkpoint under a run with training/config.json."
        )
    latest = sessions[-1]
    return {
        "run_dir": run_dir,
        "config_path": config_path,
        "session_index": latest.get("session_index", len(sessions) - 1),
        "cli": dict(latest.get("cli") or {}),
        "computed": dict(latest.get("computed") or {}),
    }


def _config_value(cli: dict, computed: dict, cli_key: str, computed_key: str | None = None):
    if computed_key and computed.get(computed_key) is not None:
        return computed[computed_key]
    return cli.get(cli_key)


def apply_run_config_to_args(
    args,
    run_config: dict[str, Any],
    *,
    path_fields: frozenset[str] = frozenset({"datapath", "yield_csv"}),
) -> None:
    """
    Fill argparse Namespace fields from the latest training session when unset (None).

    Boolean flags rc/interp are taken from config when the key is present.
    """
    cli = run_config["cli"]
    computed = run_config["computed"]
    config_path = run_config["config_path"]

    scalar_fields = {
        "datapath": ("datapath", None),
        "yield_csv": ("yield_csv", None),
        "sequencelength": ("sequencelength", None),
        "feature_layout": ("feature_layout", "feature_layout"),
        "seed": ("seed", None),
    }
    for arg_name, (cli_key, computed_key) in scalar_fields.items():
        if not hasattr(args, arg_name):
            continue
        if getattr(args, arg_name) is not None:
            continue
        val = _config_value(cli, computed, cli_key, computed_key)
        if val is None:
            raise ValueError(
                f"Missing {arg_name!r} in {config_path} and not passed on the command line."
            )
        setattr(args, arg_name, Path(val) if arg_name in path_fields else val)

    for arg_name, cli_key, default in (
        ("rc", "rc", False),
        ("interp", "interp", False),
    ):
        setattr(args, arg_name, bool(cli[cli_key]) if cli_key in cli else default)
