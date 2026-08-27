"""Map a MoCo encoder_q checkpoint onto STNetRegression.

When the yield MLP is wider than the MoCo stem (rain 12 → climate 16), copy the
overlapping in_features and leave extra input columns at their current init.
"""

from __future__ import annotations

from typing import Any

import torch


def pad_linear_in_features(
    src: torch.Tensor, dst: torch.Tensor
) -> torch.Tensor | None:
    """Copy a narrower Linear weight into ``dst``, keeping extra columns as-is."""
    if src.ndim != 2 or dst.ndim != 2:
        return None
    if src.shape[0] != dst.shape[0]:
        return None
    if src.shape[1] >= dst.shape[1]:
        return None
    out = dst.clone()
    out[:, : src.shape[1]] = src.to(device=out.device, dtype=out.dtype)
    return out


def remap_moco_encoder_state(
    pretrain_state: dict[str, Any],
    model_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[tuple], list[tuple]]:
    """
    Return (loadable_state, skipped_mismatches, padded_keys).

    ``padded_keys`` entries are ``(name, src_shape, dst_shape)`` for Linear
    weights that were copied along a prefix of in_features.
    """
    state_dict: dict[str, torch.Tensor] = {}
    skipped_shape: list[tuple] = []
    padded_keys: list[tuple] = []

    for k, v in pretrain_state.items():
        if not k.startswith("encoder_q."):
            continue
        if (
            k.startswith("encoder_q.decoder")
            or k.startswith("encoder_q.classification")
            or k.startswith("encoder_q.position_enc.pe")
        ):
            continue
        new_k = k[len("encoder_q.") :]
        if new_k not in model_state:
            continue
        dst = model_state[new_k]
        if tuple(dst.shape) == tuple(v.shape):
            state_dict[new_k] = v
            continue
        padded = pad_linear_in_features(v, dst)
        if padded is not None:
            state_dict[new_k] = padded
            padded_keys.append((new_k, tuple(v.shape), tuple(dst.shape)))
        else:
            skipped_shape.append((new_k, tuple(v.shape), tuple(dst.shape)))

    return state_dict, skipped_shape, padded_keys
