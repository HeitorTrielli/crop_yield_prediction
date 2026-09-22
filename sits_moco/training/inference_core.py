"""Shared STNetRegression checkpoint load + pixel-forward helpers for eval/viz."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from datasets.feature_layout import feature_layout_input_dim, normalize_feature_layout
from models import STNetRegression
from yield_utils import (
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
    model = STNetRegression(
        input_dim=input_dim,
        num_outputs=1,
        max_seq_len=sequencelength,
        **model_kw,
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
