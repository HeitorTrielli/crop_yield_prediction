"""
Standard directory layout for model training runs.

Layout::

    {logdir}/{run_name}/
        training/      model_best.pth, checkpoint_epoch_*.pth, trainlog.csv, config.json
        predictions/   CSV outputs from predict_yield.py and related scripts
        figures/
            intramunicipal_heatmap/   pixel-level yield / NDVI heatmaps
            municipal_heatmaps/       state/municipality choropleth maps

Legacy runs (artifacts directly under the run folder) are still supported for loading.
"""

from __future__ import annotations

from pathlib import Path

TRAINING_SUBDIR = "training"
FIGURES_SUBDIR = "figures"
PREDICTIONS_SUBDIR = "predictions"
INTRAMUNICIPAL_HEATMAP_SUBDIR = "intramunicipal_heatmap"
MUNICIPAL_HEATMAPS_SUBDIR = "municipal_heatmaps"


def run_dir_from_path(path: Path | str) -> Path:
    """Resolve the run root from a checkpoint path or any file inside a run."""
    p = Path(path).resolve()
    if p.is_file():
        p = p.parent
    if p.name == TRAINING_SUBDIR:
        return p.parent
    if p.name in (INTRAMUNICIPAL_HEATMAP_SUBDIR, MUNICIPAL_HEATMAPS_SUBDIR):
        return p.parent.parent
    if p.name in (FIGURES_SUBDIR, PREDICTIONS_SUBDIR):
        return p.parent
    if p.parent.name in (INTRAMUNICIPAL_HEATMAP_SUBDIR, MUNICIPAL_HEATMAPS_SUBDIR):
        return p.parent.parent.parent
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
    """Path to model_best.pth (new layout first, then legacy at run root)."""
    run_dir = Path(run_dir)
    new = run_dir / TRAINING_SUBDIR / "model_best.pth"
    if new.is_file():
        return new
    legacy = run_dir / "model_best.pth"
    if legacy.is_file():
        return legacy
    return new


def resolve_checkpoint_path(path: Path | str) -> Path:
    """Resolve model_best.pth from a file path, run directory, or legacy folder name."""
    p = Path(path)
    if p.is_file():
        return p.resolve()

    if p.is_dir():
        if p.name == TRAINING_SUBDIR:
            candidate = p / "model_best.pth"
            if candidate.is_file():
                return candidate.resolve()
            run_dir = p.parent
        elif p.name == "model_best.pth":
            run_dir = p.parent
        else:
            run_dir = p

        for candidate in (
            run_dir / TRAINING_SUBDIR / "model_best.pth",
            run_dir / "model_best.pth",
        ):
            if candidate.is_file():
                return candidate.resolve()

        nested = p / "model_best.pth"
        if nested.is_file():
            return nested.resolve()

    raise FileNotFoundError(
        f"Checkpoint not found at {path!r}. "
        f"Pass a .pth file or a run directory (tried "
        f"{{run}}/training/model_best.pth)."
    )


def trainlog_path(run_dir: Path | str) -> Path:
    """Path to trainlog.csv (prefers training/ subfolder)."""
    run_dir = Path(run_dir)
    p = run_dir / TRAINING_SUBDIR / "trainlog.csv"
    if p.is_file():
        return p
    return run_dir / "trainlog.csv"


def run_dir_from_forecasts_csv(csv_path: Path | str) -> Path | None:
    """If CSV lives under {run}/predictions/, return the run root."""
    csv_path = Path(csv_path).resolve()
    if csv_path.parent.name == PREDICTIONS_SUBDIR:
        return csv_path.parent.parent
    return None
