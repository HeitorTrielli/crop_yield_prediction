"""
Post-hoc municipal bias correction for downscaled (pixel) yield maps.

Idea
----
The model predicts a yield at every soy pixel. Those pixels are averaged to a
municipality forecast. Even when that municipal forecast is biased (too high /
too low vs IBGE PAM), the *relative* pattern inside the municipality can still
be useful for heatmaps / talhões.

Bias correction keeps that spatial pattern but forces the municipal mean to
match the PAM label for that (municipality, year):

  shift:  y'_i = y_i + (y_PAM - mean(y))     # preferred for t/ha
  scale:  y'_i = y_i * (y_PAM / mean(y))     # keeps positivity if y>0

Important
---------
Applied to a *single municipal scalar* this is trivial (forecast becomes PAM).
It is meant for **pixel maps** (and plot aggregates built from corrected pixels),
not as a way to inflate municipal R².
"""

from __future__ import annotations

import numpy as np


def correct_values_to_target_mean(
    values: np.ndarray,
    target_mean: float,
    *,
    mode: str = "shift",
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Rescale/shift ``values`` so their mean equals ``target_mean``.

    ``mask`` (bool, same shape) selects which entries participate in the mean
    and are updated; others are left unchanged. NaNs are ignored in the mean.
    """
    if mode not in ("shift", "scale"):
        raise ValueError(f"mode must be 'shift' or 'scale', got {mode!r}")

    out = np.array(values, dtype=np.float64, copy=True)
    if mask is None:
        valid = np.isfinite(out)
    else:
        valid = np.asarray(mask, dtype=bool) & np.isfinite(out)

    if not np.any(valid):
        return out

    current = float(np.mean(out[valid]))
    if mode == "shift":
        out[valid] = out[valid] + (float(target_mean) - current)
        return out

    if abs(current) < 1e-12:
        # Degenerate: fall back to constant map at PAM
        out[valid] = float(target_mean)
        return out
    out[valid] = out[valid] * (float(target_mean) / current)
    return out


def correct_prediction_map_to_pam(
    prediction_map: dict,
    pam_value: float,
    *,
    mode: str = "shift",
) -> dict:
    """Apply bias correction to a heatmap prediction_map {(…): float}."""
    if not prediction_map:
        return prediction_map
    keys = list(prediction_map.keys())
    vals = np.array([prediction_map[k] for k in keys], dtype=np.float64)
    corrected = correct_values_to_target_mean(vals, pam_value, mode=mode)
    return {k: float(v) for k, v in zip(keys, corrected)}
