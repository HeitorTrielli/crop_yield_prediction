"""Ensure a MoCo checkpoint matching a yield trial's trunk exists (train if needed)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from datasets.feature_layout import normalize_feature_layout
from env_config import resolve_datapath
from run_paths import ensure_dir

REPO_ROOT = Path(__file__).resolve().parent.parent


def _harvest_years_list(value: Any) -> list[int]:
    if value is None:
        return [2020, 2021, 2022, 2023, 2024]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def _year_tag(harvest_years: list[int]) -> str:
    years = sorted({int(y) for y in harvest_years})
    if len(years) == 1:
        return str(years[0])
    return f"{years[0]}-{years[-1]}"


def moco_arch_tag(
    *,
    feature_layout: str,
    d_model: int,
    n_head: int,
    d_inner: int,
    n_layers: int,
) -> str:
    layout = normalize_feature_layout(feature_layout)
    return f"{layout}_d{int(d_model)}_h{int(n_head)}_i{int(d_inner)}_L{int(n_layers)}"


def predict_moco_modelname(
    *,
    harvest_years: Any,
    feature_layout: str,
    d_model: int = 128,
    n_head: int = 16,
    d_inner: int = 128,
    n_layers: int = 1,
    rc: bool = True,
    use_doy: bool = True,
    mlp: bool = True,
    model: str = "transformer",
) -> str:
    """
    Mirror main_moco.py naming for Paraná --useall runs:

      P_MoCoV2TransformerModel_RC_{years}_doy_{layout}_d{D}_h{H}_i{I}_L{L}
    """
    model = str(model).lower()
    if model != "transformer":
        raise ValueError(
            f"auto-MoCo currently supports model='transformer', got {model!r}"
        )
    base = "MoCo" + ("V2" if mlp else "") + "TransformerModel"
    rc_str = "RC" if rc else "Pad"
    year = _year_tag(_harvest_years_list(harvest_years))
    name = f"P_{base}_{rc_str}_{year}"
    arch = moco_arch_tag(
        feature_layout=feature_layout,
        d_model=d_model,
        n_head=n_head,
        d_inner=d_inner,
        n_layers=n_layers,
    )
    suffix = f"doy_{arch}" if use_doy else arch
    return f"{name}_{suffix}"


def arch_from_trial_params(trial_params: dict) -> dict[str, Any]:
    return {
        "feature_layout": normalize_feature_layout(
            trial_params.get("feature_layout", "spectral")
        ),
        "d_model": int(trial_params.get("model_d_model", 128)),
        "n_head": int(trial_params.get("model_n_head", 16)),
        "d_inner": int(trial_params.get("model_d_inner", 128)),
        "n_layers": int(trial_params.get("model_n_layers", 1)),
    }


def predict_moco_run_dir(
    trial_params: dict,
    moco_cfg: dict,
    *,
    repo_root: Path | None = None,
) -> Path:
    repo_root = repo_root or REPO_ROOT
    arch = arch_from_trial_params(trial_params)
    harvest_years = moco_cfg.get("harvest_years") or trial_params.get(
        "harvest_years", "2020,2021,2022,2023,2024"
    )
    modelname = predict_moco_modelname(
        harvest_years=harvest_years,
        feature_layout=arch["feature_layout"],
        d_model=arch["d_model"],
        n_head=arch["n_head"],
        d_inner=arch["d_inner"],
        n_layers=arch["n_layers"],
        rc=bool(moco_cfg.get("rc", True)),
        use_doy=bool(moco_cfg.get("use_doy", True)),
        mlp=bool(moco_cfg.get("mlp", True)),
        model=str(moco_cfg.get("model", "transformer")),
    )
    logdir = Path(moco_cfg.get("logdir") or trial_params.get("logdir") or "./results")
    if not logdir.is_absolute():
        logdir = (repo_root / logdir).resolve()
    return logdir / modelname


def predict_moco_checkpoint_path(
    trial_params: dict,
    moco_cfg: dict,
    *,
    repo_root: Path | None = None,
) -> Path:
    return (
        predict_moco_run_dir(trial_params, moco_cfg, repo_root=repo_root)
        / "training"
        / "model_best.pth"
    )


def build_moco_argv(
    trial_params: dict,
    moco_cfg: dict,
    *,
    repo_root: Path | None = None,
    python_exe: str | None = None,
) -> list[str]:
    """Build subprocess argv for main_moco.py matching the yield trial trunk."""
    repo_root = repo_root or REPO_ROOT
    arch = arch_from_trial_params(trial_params)
    harvest_years = moco_cfg.get("harvest_years") or trial_params.get(
        "harvest_years", "2020,2021,2022,2023,2024"
    )
    if isinstance(harvest_years, (list, tuple)):
        harvest_years = ",".join(str(int(y)) for y in harvest_years)

    datapath = moco_cfg.get("datapath") or trial_params.get("datapath")
    datapath = str(resolve_datapath(datapath))

    script = repo_root / str(moco_cfg.get("script", "main_moco.py"))
    argv = [
        python_exe or sys.executable,
        str(script),
        str(moco_cfg.get("model", "transformer")),
        "--feature-layout",
        arch["feature_layout"],
        "--model-d-model",
        str(arch["d_model"]),
        "--model-n-head",
        str(arch["n_head"]),
        "--model-d-inner",
        str(arch["d_inner"]),
        "--model-n-layers",
        str(arch["n_layers"]),
        "--model-dropout",
        str(float(moco_cfg.get("model_dropout", 0.2))),
        "--datapath",
        datapath,
        "--harvest-years",
        str(harvest_years),
        "--max-samples",
        str(int(moco_cfg.get("max_samples", 500_000))),
        "-e",
        str(int(moco_cfg.get("epochs", 100))),
        "-b",
        str(int(moco_cfg.get("batchsize", 512))),
        "-j",
        str(int(moco_cfg.get("workers", 0))),
        "--sequencelength",
        str(int(moco_cfg.get("sequencelength", 70))),
        "-lr",
        str(float(moco_cfg.get("learning_rate", 1e-3))),
        "--weight-decay",
        str(float(moco_cfg.get("weight_decay", 1e-4))),
        "--seed",
        str(int(moco_cfg.get("seed", 111))),
        "-l",
        str(moco_cfg.get("logdir") or trial_params.get("logdir") or "./results"),
    ]

    if bool(moco_cfg.get("rc", True)):
        argv.append("--rc")
    if bool(moco_cfg.get("use_doy", True)):
        argv.append("--use-doy")
    if bool(moco_cfg.get("useall", True)):
        argv.append("--useall")
    if bool(moco_cfg.get("mlp", True)):
        argv.append("--mlp")
    if bool(moco_cfg.get("rebuild_cache", False)):
        argv.append("--rebuild-cache")
    if moco_cfg.get("warmup_epochs") is not None:
        argv.extend(["--warmup-epochs", str(int(moco_cfg["warmup_epochs"]))])

    return argv


def ensure_moco_checkpoint(
    trial_params: dict,
    moco_cfg: dict,
    *,
    repo_root: Path | None = None,
    dry_run: bool = False,
    capture_log: Path | None = None,
) -> Path:
    """
    Return path to model_best.pth for this trial's MoCo fingerprint.

    Trains MoCo via subprocess when the checkpoint is missing.
    """
    repo_root = repo_root or REPO_ROOT
    ckpt = predict_moco_checkpoint_path(trial_params, moco_cfg, repo_root=repo_root)
    if ckpt.is_file():
        return ckpt.resolve()

    argv = build_moco_argv(trial_params, moco_cfg, repo_root=repo_root)
    if dry_run:
        return ckpt

    print(f"MoCo checkpoint missing: {ckpt}")
    print(f"Training MoCo: {' '.join(argv)}")
    if capture_log is not None:
        ensure_dir(capture_log.parent, quiet=True)
        with open(capture_log, "w", encoding="utf-8") as log_f:
            proc = subprocess.run(
                argv,
                cwd=str(repo_root),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                check=False,
            )
    else:
        proc = subprocess.run(argv, cwd=str(repo_root), check=False)

    if proc.returncode != 0:
        raise RuntimeError(
            f"MoCo pretrain failed (returncode={proc.returncode}) for {ckpt}"
        )
    if not ckpt.is_file():
        raise RuntimeError(
            f"MoCo pretrain finished but checkpoint not found: {ckpt}"
        )
    return ckpt.resolve()
