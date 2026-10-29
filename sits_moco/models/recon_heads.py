"""Parallel reconstruction heads that keep encoder features invertible."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error over valid time steps (and channels)."""
    if pred.shape != target.shape:
        raise ValueError(f"recon shape mismatch {tuple(pred.shape)} vs {tuple(target.shape)}")
    if valid.dim() == 2:
        # [B, T] -> [B, T, 1]
        valid = valid.unsqueeze(-1)
    valid = valid.to(dtype=pred.dtype)
    denom = torch.clamp(valid.sum(), min=1.0)
    return ((pred - target).pow(2) * valid).sum() / denom


class PooledSummaryRecon(nn.Module):
    """MLP from fused embedding → modality summary vector (means + soil)."""

    def __init__(self, fused_dim: int, summary_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, summary_dim),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.net(fused)


class SequenceReconHead(nn.Module):
    """Per-timestep linear map ``[B,T,d] → [B,T,C]``."""

    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(tokens)


def pooled_summary_target(
    x_spec: torch.Tensor | None,
    mask_s2: torch.Tensor | None,
    x_clim: torch.Tensor | None,
    mask_c: torch.Tensor | None,
    soil: torch.Tensor | None,
) -> torch.Tensor:
    """Concatenate masked temporal means (+ soil) as recon target."""
    parts: list[torch.Tensor] = []
    if x_spec is not None:
        valid = (~mask_s2).to(dtype=x_spec.dtype).unsqueeze(-1) if mask_s2 is not None else None
        if valid is None:
            parts.append(x_spec.mean(dim=1))
        else:
            denom = torch.clamp(valid.sum(dim=1), min=1e-8)
            parts.append((x_spec * valid).sum(dim=1) / denom)
    if x_clim is not None:
        valid = (~mask_c).to(dtype=x_clim.dtype).unsqueeze(-1) if mask_c is not None else None
        if valid is None:
            parts.append(x_clim.mean(dim=1))
        else:
            denom = torch.clamp(valid.sum(dim=1), min=1e-8)
            parts.append((x_clim * valid).sum(dim=1) / denom)
    if soil is not None:
        parts.append(soil)
    if not parts:
        raise ValueError("pooled recon target is empty")
    return torch.cat(parts, dim=-1)


def pooled_summary_dim(
    spectral_dim: int,
    climate_dim: int,
    soil_dim: int,
    *,
    use_spectral: bool,
    use_climate: bool,
    use_soil: bool,
) -> int:
    dim = 0
    if use_spectral:
        dim += int(spectral_dim)
    if use_climate:
        dim += int(climate_dim)
    if use_soil:
        dim += int(soil_dim)
    return dim
