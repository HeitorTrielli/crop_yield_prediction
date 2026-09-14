"""
Optional within-municipality auxiliary losses (variance / ranking).

Used on top of municipal aggregated MSE to discourage mean-collapse: every pixel
predicting the same constant.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pixel_ndvi_proxy(x: torch.Tensor) -> torch.Tensor:
    """
    Per-pixel phenology proxy from spectral cube ``x`` [N, T, C].

    Uses red (band 2) and NIR (band 6) — same indexing as ``getWeight_batch``.
    Returns mean NDVI over time for each pixel, shape [N].
    """
    if x.dim() != 3 or x.size(-1) < 7:
        raise ValueError(f"expected x [N,T,C>=7], got {tuple(x.shape)}")
    red = x[:, :, 2]
    nir = x[:, :, 6]
    ndvi = (nir - red) / (nir + red + 1e-8)
    return ndvi.mean(dim=1)


def variance_hinge_loss(
    pixel_std: torch.Tensor,
    *,
    min_std: float = 0.05,
) -> torch.Tensor:
    """Penalize when within-muni pixel std falls below ``min_std``.

    ``min_std`` is in decoder units (z-score for new heads; t/ha for legacy raw).
    """
    if pixel_std.dim() == 0:
        pixel_std = pixel_std.unsqueeze(0)
    # Use primary channel if multi-output
    std0 = pixel_std.reshape(-1)[0]
    return F.relu(float(min_std) - std0).pow(2)


def ranking_correlation_loss(
    predictions: torch.Tensor,
    proxy: torch.Tensor,
) -> torch.Tensor:
    """
    Soft ranking loss: 1 - Pearson corr(pred, proxy) on the current pixel chunk.

    Encourages higher predictions where the NDVI proxy is higher. Returns 0 when
    either vector has near-zero variance (no ranking signal).
    """
    pred = predictions
    if pred.dim() > 1:
        pred = pred[:, 0]
    pred = pred.reshape(-1).to(torch.float32)
    proxy = proxy.reshape(-1).to(torch.float32)
    if pred.numel() < 2:
        return pred.new_zeros(())

    pred = pred - pred.mean()
    proxy = proxy - proxy.mean()
    pred_std = pred.std(unbiased=False)
    proxy_std = proxy.std(unbiased=False)
    if float(pred_std) < 1e-8 or float(proxy_std) < 1e-8:
        return pred.new_zeros(())

    corr = (pred * proxy).mean() / (pred_std * proxy_std + 1e-8)
    # Bound numerically; loss in [0, 2]
    corr = torch.clamp(corr, -1.0, 1.0)
    return 1.0 - corr
