"""Shared STNetRegression checkpoint load + pixel-forward helpers for eval/viz."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from datasets.daily_climate import CLIMATE_MAX_SEQ_LEN
from datasets.feature_layout import (
    feature_layout_climate_input_dim,
    feature_layout_input_dim,
    feature_layout_spectral_dim,
    normalize_feature_layout,
)
from models import DualSTNetRegression, STNetRegression
from yield_utils import (
    is_dual_stnet_state_dict,
    resolve_head_output,
    resolve_inference_target,
    resolve_model_kwargs,
    stnet_regression_input_dim_from_state_dict,
)


def load_checkpoint(path: str | Path, device: str | torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def build_stnet_from_checkpoint(
    checkpoint: dict[str, Any],
    *,
    device: torch.device,
    sequencelength: int,
    run_config: dict | None = None,
    feature_layout: str | None = None,
) -> tuple[STNetRegression, dict[str, Any]]:
    """
    Construct ``STNetRegression``, load weights, return ``(model, meta)``.

    ``meta`` includes input_dim, feature_layout, head_output, target_*, model_kw.
    """
    state_dict = checkpoint["model_state"]
    input_dim = stnet_regression_input_dim_from_state_dict(state_dict)
    ck_fl = checkpoint.get("feature_layout")
    if feature_layout is not None:
        expected_dim = feature_layout_input_dim(feature_layout)
        if input_dim != expected_dim:
            raise ValueError(
                f"Checkpoint has input_dim={input_dim} but feature_layout "
                f"{feature_layout!r} implies {expected_dim}."
            )
        if ck_fl is not None and normalize_feature_layout(ck_fl) != normalize_feature_layout(
            feature_layout
        ):
            raise ValueError(
                f"Checkpoint feature_layout={ck_fl!r} does not match {feature_layout!r}."
            )
    target, target_column, aggregation, target_unit = resolve_inference_target(
        run_config or {}, checkpoint
    )
    head_output = resolve_head_output(checkpoint, run_config)
    model_kw = resolve_model_kwargs(run_config or {}, checkpoint)
    if is_dual_stnet_state_dict(state_dict):
        spec_w = state_dict.get("spectral.mlp1.0.lin.weight")
        clim_w = state_dict.get("climate.mlp1.0.lin.weight")
        spectral_dim = int(spec_w.shape[1]) if spec_w is not None else 10
        climate_dim = int(clim_w.shape[1]) if clim_w is not None else 5
        if feature_layout is not None:
            spectral_dim = feature_layout_spectral_dim(feature_layout)
            climate_dim = feature_layout_climate_input_dim(feature_layout) or climate_dim
        dual_kw = {
            k: v
            for k, v in model_kw.items()
            if k
            in {
                "d_model",
                "n_head",
                "n_layers",
                "d_inner",
                "dropout",
                "temporal_pooling",
                "attn_pool_queries",
                "soil_fusion",
                "climate_pooling",
                "climate_max_seq_len",
            }
        }
        cli = (run_config or {}).get("cli") or {}
        climate_seq = int(
            dual_kw.pop("climate_max_seq_len", None)
            or cli.get("climate_sequencelength")
            or CLIMATE_MAX_SEQ_LEN
        )
        model = DualSTNetRegression(
            spectral_dim=spectral_dim,
            climate_dim=climate_dim,
            input_dim=input_dim,
            num_outputs=1,
            max_seq_len=sequencelength,
            climate_max_seq_len=climate_seq,
            **dual_kw,
        ).to(device)
    else:
        single_kw = {
            k: v
            for k, v in model_kw.items()
            if k
            not in {"climate_pooling", "climate_max_seq_len"}
        }
        model = STNetRegression(
            input_dim=input_dim,
            num_outputs=1,
            max_seq_len=sequencelength,
            **single_kw,
        ).to(device)
    if hasattr(model, "_orig_mod"):
        model._orig_mod.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    meta = {
        "input_dim": input_dim,
        "feature_layout": ck_fl,
        "target": target,
        "target_column": target_column,
        "aggregation": aggregation,
        "target_unit": target_unit,
        "target_mean": checkpoint.get("target_mean", 0.0),
        "target_std": checkpoint.get("target_std", 1.0),
        "head_output": head_output,
        "model_kw": model_kw,
    }
    return model, meta
