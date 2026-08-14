"""Load and validate tuning study YAML configs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from env_config import resolve_datapath

OBJECTIVES = {
    "val_loss": {"mode": "min", "column": "valloss"},
    "val_r2": {"mode": "max", "column": "r2"},
    "val_rmse": {"mode": "min", "column": "rmse"},
    "test_r2": {"mode": "max", "column": "r2", "log": "testlog"},
    "test_loss": {"mode": "min", "column": "testloss", "log": "testlog"},
}

SEARCH_STRATEGIES = ("grid", "random", "list")

BOOL_PARAMS = frozenset(
    {
        "rc",
        "interp",
        "no_compile",
        "overwrite_run",
        "quiet_training",
        "disable_pipeline_h2d",
        "zero_grad_set_to_none",
        "skip_batch_empty_cache",
        "batch_empty_cache",
        "legacy_zero_grad",
        "dataloader_persistent_workers",
        "h2d_pin_host",
        "no_dataloader_pin_memory",
        "auto_chunks_per_grad",
        "no_coverage_filter",
    }
)

INT_PARAMS = frozenset(
    {
        "epochs",
        "batchsize",
        "workers",
        "sequencelength",
        "warmup_epochs",
        "seed",
        "early_stop_patience",
        "model_d_model",
        "model_n_head",
        "model_n_layers",
        "model_d_inner",
        "pixel_chunk_size",
        "prefetch_chunks",
        "dataloader_prefetch_factor",
        "npy_cache_size",
        "chunks_per_grad",
        "auto_chunks_per_grad_min",
        "auto_pixel_chunk_size_min",
        "auto_pixel_chunk_size_step",
        "auto_chunks_per_grad_narrow_threshold",
        "auto_chunks_per_grad_refine_radius",
        "min_images",
        "min_months",
    }
)

FLOAT_PARAMS = frozenset(
    {
        "learning_rate",
        "weight_decay",
        "sample_ratio",
        "model_dropout",
        "aux_loss_weight",
        "aux_min_std",
        "auto_chunks_per_grad_vram_frac",
        "auto_chunks_per_grad_vram_target_frac",
        "min_coverage_ratio",
        "max_coverage_ratio",
        "train_mid_yield_keep_fraction",
        "train_mid_yield_lo",
        "train_mid_yield_hi",
        "train_mid_yield_bin_width",
    }
)


def _require_mapping(obj: Any, name: str) -> dict:
    if not isinstance(obj, dict):
        raise ValueError(f"{name} must be a mapping, got {type(obj).__name__}")
    return obj


def _apply_env_datapath(base: dict) -> None:
    """Set base['datapath'] from .env when omitted in study YAML."""
    override = base.get("datapath")
    base["datapath"] = str(resolve_datapath(override))


def load_study_config(path: Path | str) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Study config not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = _require_mapping(raw, "study config root")

    name = cfg.get("name")
    if not name or not isinstance(name, str):
        raise ValueError("Study config must include a non-empty string 'name'")

    objective = cfg.get("objective", "val_r2")
    if objective not in OBJECTIVES:
        choices = ", ".join(sorted(OBJECTIVES))
        raise ValueError(f"Unknown objective {objective!r}; expected one of: {choices}")

    base = _require_mapping(cfg.get("base"), "base")
    _apply_env_datapath(base)
    search = _require_mapping(cfg.get("search", {}), "search")
    strategy = search.get("strategy", "grid")
    if strategy not in SEARCH_STRATEGIES:
        raise ValueError(f"search.strategy must be one of {SEARCH_STRATEGIES}, got {strategy!r}")

    parameters = search.get("parameters", {})
    if parameters is not None and not isinstance(parameters, dict):
        raise ValueError("search.parameters must be a mapping")

    n_trials = search.get("n_trials")
    explicit_trials = search.get("trials")
    if strategy == "random":
        if n_trials is None or int(n_trials) < 1:
            raise ValueError("random search requires search.n_trials >= 1")
    elif strategy == "list":
        if not isinstance(explicit_trials, list) or not explicit_trials:
            raise ValueError("list search requires a non-empty search.trials list")
        for i, trial in enumerate(explicit_trials):
            if not isinstance(trial, dict):
                raise ValueError(f"search.trials[{i}] must be a mapping")
    elif parameters:
        for pname, spec in parameters.items():
            if not isinstance(spec, dict):
                raise ValueError(f"Parameter {pname!r} spec must be a mapping")
            ptype = spec.get("type", "choice")
            if ptype == "choice" and "values" not in spec:
                raise ValueError(f"Parameter {pname!r}: choice requires 'values'")

    output_dir = cfg.get("output_dir", "./results/tuning")
    training_script = cfg.get("training_script", "main_yield_regression_polars.py")
    seed_offset = int(cfg.get("seed_offset", 0))

    moco_raw = cfg.get("moco")
    moco_cfg = None
    if moco_raw is not None:
        moco_cfg = deepcopy(_require_mapping(moco_raw, "moco"))

    return {
        "name": name,
        "objective": objective,
        "objective_spec": OBJECTIVES[objective],
        "base": deepcopy(base),
        "search": {
            "strategy": strategy,
            "n_trials": int(n_trials) if n_trials is not None else None,
            "parameters": deepcopy(parameters or {}),
            "trials": deepcopy(explicit_trials) if explicit_trials is not None else None,
            "seed": int(search.get("seed", 42)),
        },
        "output_dir": str(output_dir),
        "training_script": str(training_script),
        "seed_offset": seed_offset,
        "config_path": str(path.resolve()),
        "moco": moco_cfg,
    }


def merge_trial_params(base: dict, sampled: dict) -> dict:
    """Deep-merge sampled search params onto base config."""
    merged = deepcopy(base)
    for key, value in sampled.items():
        merged[key] = value
    return merged


def coerce_param_types(params: dict) -> dict:
    """Cast trial params to expected Python types for CLI building."""
    out = deepcopy(params)
    for key, value in out.items():
        if key in BOOL_PARAMS:
            out[key] = bool(value)
        elif key in INT_PARAMS and value is not None:
            out[key] = int(value)
        elif key in FLOAT_PARAMS and value is not None:
            out[key] = float(value)
        elif key == "schedule" and value is not None:
            if isinstance(value, str):
                out[key] = [int(x.strip()) for x in value.split(",") if x.strip()]
            else:
                out[key] = [int(x) for x in value]
    return out
