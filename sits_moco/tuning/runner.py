"""Execute a single tuning trial via the training script."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tuning.config import BOOL_PARAMS, coerce_param_types

from run_paths import ensure_dir, run_dir_for_experiment

REPO_ROOT = Path(__file__).resolve().parent.parent

TARGET_PREFIX = {
    "productivity": "Productivity",
    "productivity_dev": "ProductivityDev",
    "total_adj": "TotalAdj",
    "total": "Total",
}


def _format_years(harvest_years_set: set[int]) -> str:
    return "_".join(f"{y % 100:02d}" for y in sorted(harvest_years_set))


def predict_run_name(params: dict) -> str:
    target = params["target"]
    parts = [TARGET_PREFIX.get(target, "Total")]
    harvest_years_set = {
        int(x.strip()) for x in str(params["harvest_years"]).split(",") if x.strip()
    }
    parts.append(_format_years(harvest_years_set))
    parts.append(f"Seed{int(params.get('seed', 42))}")
    suffix = params.get("suffix")
    if suffix:
        parts.append(str(suffix))
    return "_".join(parts)

# CLI flags that differ from snake_case param names.
FLAG_ALIASES: dict[str, tuple[str, ...]] = {
    "learning_rate": ("-lr", "--learning-rate"),
    "batchsize": ("-b", "--batchsize"),
    "sequencelength": ("-seq", "--sequencelength"),
    "workers": ("-j", "--workers"),
    "epochs": ("-e", "--epochs"),
    "logdir": ("-l", "--logdir"),
    "suffix": ("-s", "--suffix"),
    "target": ("--target",),
    "harvest_years": ("--harvest-years",),
    "feature_layout": ("--feature-layout",),
    "weight_decay": ("--weight-decay",),
    "sample_ratio": ("--sample-ratio",),
    "warmup_epochs": ("--warmup-epochs",),
    "early_stop_patience": ("--early-stop-patience",),
    "model_d_model": ("--model-d-model",),
    "model_n_head": ("--model-n-head",),
    "model_n_layers": ("--model-n-layers",),
    "model_d_inner": ("--model-d-inner",),
    "model_dropout": ("--model-dropout",),
    "pretrained": ("--pretrained",),
    "datapath": ("--datapath",),
    "yield_csv": ("--yield-csv",),
    "seed": ("--seed",),
    "min_coverage_ratio": ("--min-coverage-ratio",),
    "max_coverage_ratio": ("--max-coverage-ratio",),
    "train_mid_yield_keep_fraction": ("--train-mid-yield-keep-fraction",),
    "train_mid_yield_lo": ("--train-mid-yield-lo",),
    "train_mid_yield_hi": ("--train-mid-yield-hi",),
    "train_mid_yield_bin_width": ("--train-mid-yield-bin-width",),
    "min_images": ("--min-images",),
    "min_months": ("--min-months",),
    "pretrained": ("--pretrained",),
    "schedule": ("--schedule",),
    "pixel_chunk_size": ("--pixel-chunk-size",),
    "prefetch_chunks": ("--prefetch-chunks",),
    "chunks_per_grad": ("--chunks-per-grad",),
    "auto_chunks_per_grad_min": ("--auto-chunks-per-grad-min",),
    "auto_pixel_chunk_size_min": ("--auto-pixel-chunk-size-min",),
    "auto_pixel_chunk_size_step": ("--auto-pixel-chunk-size-step",),
    "auto_chunks_per_grad_vram_frac": ("--auto-chunks-per-grad-vram-frac",),
    "auto_chunks_per_grad_vram_target_frac": (
        "--auto-chunks-per-grad-vram-target-frac",
    ),
    "auto_chunks_per_grad_narrow_threshold": (
        "--auto-chunks-per-grad-narrow-threshold",
    ),
    "auto_chunks_per_grad_refine_radius": (
        "--auto-chunks-per-grad-refine-radius",
    ),
}


def _param_to_flags(key: str, value: Any) -> list[str]:
    if value is None:
        return []
    if key in BOOL_PARAMS:
        return [f"--{key.replace('_', '-')}"] if value else []

    flags = FLAG_ALIASES.get(key)
    if flags is None:
        flags = (f"--{key.replace('_', '-')}",)

    if key == "schedule":
        if isinstance(value, str):
            items = [x.strip() for x in value.split(",") if x.strip()]
        else:
            items = [str(int(x)) for x in value]
        return [flags[0], *items]

    primary = flags[0]
    if isinstance(value, bool):
        return [primary] if value else []
    if isinstance(value, float):
        if value == 0.0:
            return [primary, "0"]
        text = f"{value:.12g}"
        return [primary, text]
    return [primary, str(value)]


def build_training_argv(
    params: dict,
    *,
    training_script: Path,
    repo_root: Path | None = None,
) -> list[str]:
    """Build subprocess argv for main_yield_regression_polars.py."""
    repo_root = repo_root or REPO_ROOT
    script = training_script if training_script.is_absolute() else repo_root / training_script
    argv = [sys.executable, str(script)]

    # Required ordering: --target and --harvest-years first (argparse friendly).
    priority = ("target", "harvest_years")
    ordered_keys = [k for k in priority if k in params]
    ordered_keys += sorted(k for k in params if k not in priority)

    for key in ordered_keys:
        argv.extend(_param_to_flags(key, params[key]))
    return argv


def predict_run_dir(params: dict, *, logdir: str | Path = "./results") -> Path:
    """Predict results folder for a trial from its merged params."""
    return run_dir_for_experiment(
        logdir,
        predict_run_name(params),
        params.get("feature_layout", "spectral"),
    )


def apply_resume_params(
    params: dict,
    resume_checkpoint: Path | str,
) -> dict:
    """Return a copy of trial params configured to continue an interrupted run."""
    out = dict(params)
    out["pretrained"] = str(Path(resume_checkpoint).resolve())
    out["overwrite_run"] = False
    return out


def run_trial_subprocess(
    params: dict,
    *,
    training_script: Path | str,
    repo_root: Path | None = None,
    dry_run: bool = False,
    capture_log: Path | None = None,
    append_log: bool = False,
) -> dict[str, Any]:
    """
    Run one training trial. Returns outcome dict (includes subprocess returncode).
    """
    repo_root = repo_root or REPO_ROOT
    params = coerce_param_types(params)
    argv = build_training_argv(
        params,
        training_script=Path(training_script),
        repo_root=repo_root,
    )
    run_dir = predict_run_dir(params, logdir=params.get("logdir", "./results"))

    if dry_run:
        return {
            "status": "dry_run",
            "argv": argv,
            "run_dir": str(run_dir.resolve()),
            "returncode": 0,
        }

    stdout_path = capture_log
    if stdout_path is not None:
        ensure_dir(stdout_path.parent, quiet=True)
        log_mode = "a" if append_log and stdout_path.is_file() else "w"
        with open(stdout_path, log_mode, encoding="utf-8") as log_f:
            if log_mode == "a":
                log_f.write(
                    f"\n\n=== resumed {datetime.now(timezone.utc).isoformat()} ===\n\n"
                )
                log_f.flush()
            proc = subprocess.run(
                argv,
                cwd=str(repo_root),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                check=False,
            )
    else:
        proc = subprocess.run(argv, cwd=str(repo_root), check=False)

    return {
        "argv": argv,
        "run_dir": str(run_dir.resolve()),
        "returncode": proc.returncode,
        "log_file": str(capture_log) if capture_log else None,
    }
