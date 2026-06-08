"""
Standard directory layout for model training runs.

Layout::

    {logdir}/{run_name}/
        training/      model_best.pth, checkpoint_epoch_*.pth, trainlog.csv, config.json
        predictions/   CSV outputs from predict_yield.py and related scripts
        figures/
            intramunicipal_heatmap/   pixel-level yield / NDVI heatmaps
            municipal_heatmaps/       state/municipality choropleth maps

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
