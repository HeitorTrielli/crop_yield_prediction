"""
Optional within- and across-municipality auxiliary losses.

Used on top of municipal aggregated MSE to discourage mean-collapse: every pixel
predicting the same constant.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ibge_state_from_code(code) -> str | None:
    """Two-digit IBGE UF prefix from a municipality code (e.g. 4106902 → '41')."""
    if code is None:
        return None
    text = str(code).strip()
    if len(text) < 2 or not text[:2].isdigit():
        return None
    return text[:2]


def _group_indices(
    years,
    n: int,
    device: torch.device,
    *,
    municipality_codes=None,
    by_state: bool = False,
) -> list[torch.Tensor]:
    """Indices for each (year) or (year, state) group present in the batch."""
    if years is None or len(years) != n:
        return []
    groups: dict[tuple, list[int]] = {}
    for i, year in enumerate(years):
        if year is None:
            continue
        key: tuple
        if by_state:
            if municipality_codes is None or len(municipality_codes) != n:
                return []
            state = ibge_state_from_code(municipality_codes[i])
            if state is None:
                continue
            key = (int(year), state)
        else:
            key = (int(year),)
        groups.setdefault(key, []).append(i)
    return [
        torch.as_tensor(indices, dtype=torch.long, device=device)
        for _, indices in sorted(groups.items())
        if indices
    ]


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
    if float(pred_std.detach()) < 1e-8 or float(proxy_std.detach()) < 1e-8:
        return pred.new_zeros(())

    corr = (pred * proxy).mean() / (pred_std * proxy_std + 1e-8)
    # Bound numerically; loss in [0, 2]
    corr = torch.clamp(corr, -1.0, 1.0)
    return 1.0 - corr


def _parse_muni_mode(mode: str) -> tuple[str, bool]:
    raw = str(mode).lower().removeprefix("muni_")
    by_state = raw.endswith("_state")
    kind = raw[: -len("_state")] if by_state else raw
    if kind not in {"corr", "rank", "rank_corr"}:
        raise ValueError(f"Unknown within-year municipality loss mode {mode!r}")
    return kind, by_state


def within_year_municipality_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    years,
    *,
    mode: str = "corr",
    min_count: int = 3,
    rank_temperature: float = 0.5,
    min_target_gap: float = 0.1,
    municipality_codes=None,
) -> torch.Tensor:
    """Separate municipalities within each year (or year×state) using labels.

    ``corr`` maximizes Pearson correlation. ``rank`` applies a smooth pairwise
    ordering loss, ignoring nearly tied targets. ``rank_corr`` averages both.
    Modes ending in ``_state`` (e.g. ``muni_rank_corr_state``) group by IBGE UF
    as well as year, so ranking is intra-estadual.
    Predictions and targets must use the same units (normally target z-scores).
    """
    pred = predictions.reshape(predictions.shape[0], -1)[:, 0].to(torch.float32)
    target = targets.reshape(targets.shape[0], -1)[:, 0].to(torch.float32)
    if years is None or len(years) != pred.numel():
        return pred.new_zeros(())

    kind, by_state = _parse_muni_mode(mode)
    losses: list[torch.Tensor] = []
    for idx in _group_indices(
        years,
        pred.numel(),
        pred.device,
        municipality_codes=municipality_codes,
        by_state=by_state,
    ):
        if idx.numel() < int(min_count):
            continue
        pred_y = pred.index_select(0, idx)
        target_y = target.index_select(0, idx)

        components: list[torch.Tensor] = []
        if kind in {"corr", "rank_corr"}:
            components.append(ranking_correlation_loss(pred_y, target_y))
        if kind in {"rank", "rank_corr"}:
            rows, cols = torch.triu_indices(
                pred_y.numel(), pred_y.numel(), offset=1, device=pred.device
            )
            target_delta = target_y[rows] - target_y[cols]
            keep = target_delta.abs() >= float(min_target_gap)
            if bool(keep.any()):
                signed_pred_delta = (
                    target_delta[keep].sign() * (pred_y[rows[keep]] - pred_y[cols[keep]])
                )
                temperature = max(float(rank_temperature), 1e-6)
                # Divide by log(2): an uninformative tie has loss 1, like corr.
                pair_loss = F.softplus(-signed_pred_delta / temperature).mean()
                components.append(pair_loss / 0.6931471805599453)
        if components:
            losses.append(torch.stack(components).mean())

    if not losses:
        return pred.new_zeros(())
    return torch.stack(losses).mean()


def within_group_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    years,
    *,
    municipality_codes=None,
    by_state: bool = True,
    min_count: int = 2,
) -> torch.Tensor:
    """MSE after demeaning pred/target within each year (or year×state) group.

    Forces the primary regression signal onto intra-group residuals instead of
    between-state mean differences. Returns mean over municipalities that fall
    in groups with at least ``min_count`` members (others contribute 0).
    """
    pred = predictions.reshape(predictions.shape[0], -1)[:, 0].to(torch.float32)
    target = targets.reshape(targets.shape[0], -1)[:, 0].to(torch.float32)
    if years is None or len(years) != pred.numel():
        return pred.new_zeros(())

    residual = pred.new_zeros(pred.shape)
    counted = 0
    for idx in _group_indices(
        years,
        pred.numel(),
        pred.device,
        municipality_codes=municipality_codes,
        by_state=by_state,
    ):
        if idx.numel() < int(min_count):
            continue
        pred_g = pred.index_select(0, idx)
        target_g = target.index_select(0, idx)
        residual.index_copy_(0, idx, (pred_g - pred_g.mean()) - (target_g - target_g.mean()))
        counted += int(idx.numel())
    if counted == 0:
        return pred.new_zeros(())
    return (residual ** 2).sum() / float(counted)


def within_year_supcon_loss(
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    years,
    *,
    temperature: float = 0.2,
    n_bins: int = 3,
    min_count: int = 6,
    municipality_codes=None,
    by_state: bool = False,
) -> torch.Tensor:
    """Supervised contrastive loss on fused embeddings, within year or year×state.

    Targets are binned into ``n_bins`` quantiles per group; same-bin pairs are
    positives. Encourages alto/médio/baixo separation in embedding space.
    """
    z = embeddings.to(torch.float32)
    if z.dim() != 2:
        z = z.reshape(z.shape[0], -1)
    y = targets.reshape(-1).to(torch.float32)
    if z.size(0) != y.numel():
        return z.new_zeros(())

    z = F.normalize(z, dim=-1, eps=1e-8)
    temp = max(float(temperature), 1e-4)
    losses: list[torch.Tensor] = []
    for idx in _group_indices(
        years,
        z.size(0),
        z.device,
        municipality_codes=municipality_codes,
        by_state=by_state,
    ):
        if idx.numel() < int(min_count):
            continue
        zy = z.index_select(0, idx)
        yy = y.index_select(0, idx)
        # Quantile bins via ranks (stable, differentiable-free labels).
        order = torch.argsort(yy)
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(yy.numel(), device=yy.device, dtype=torch.float32)
        bins = torch.clamp(
            (ranks * int(n_bins) / max(yy.numel(), 1)).long(),
            max=int(n_bins) - 1,
        )
        # Skip groups where a single bin dominates everything.
        if int(bins.unique().numel()) < 2:
            continue
        sim = zy @ zy.T / temp
        # Mask self-comparisons.
        eye = torch.eye(zy.size(0), device=zy.device, dtype=torch.bool)
        sim = sim.masked_fill(eye, -1e4)
        pos = bins.unsqueeze(0) == bins.unsqueeze(1)
        pos = pos & ~eye
        if not bool(pos.any()):
            continue
        # Log-sum-exp InfoNCE with multiple positives (SupCon).
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        pos_counts = pos.sum(dim=1).clamp(min=1).to(log_prob.dtype)
        mean_log_prob = (log_prob * pos.to(log_prob.dtype)).sum(dim=1) / pos_counts
        # Only anchors that have ≥1 positive contribute.
        has_pos = pos.any(dim=1)
        if bool(has_pos.any()):
            losses.append(-mean_log_prob[has_pos].mean())

    if not losses:
        return z.new_zeros(())
    return torch.stack(losses).mean()


def within_year_ordinal_embed_loss(
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    years,
    *,
    min_count: int = 4,
    min_target_gap: float = 0.1,
    municipality_codes=None,
    by_state: bool = False,
) -> torch.Tensor:
    """Push embedding cosine similarity to track within-group yield closeness.

    Loss = 1 - Spearman-like soft correlation between pairwise cosine and
    negative absolute yield gap (same year, or year×state when ``by_state``).
    """
    z = embeddings.to(torch.float32)
    if z.dim() != 2:
        z = z.reshape(z.shape[0], -1)
    y = targets.reshape(-1).to(torch.float32)
    if z.size(0) != y.numel():
        return z.new_zeros(())

    z = F.normalize(z, dim=-1, eps=1e-8)
    losses: list[torch.Tensor] = []
    for idx in _group_indices(
        years,
        z.size(0),
        z.device,
        municipality_codes=municipality_codes,
        by_state=by_state,
    ):
        if idx.numel() < int(min_count):
            continue
        zy = z.index_select(0, idx)
        yy = y.index_select(0, idx)
        rows, cols = torch.triu_indices(zy.size(0), zy.size(0), offset=1, device=z.device)
        gap = (yy[rows] - yy[cols]).abs()
        keep = gap >= float(min_target_gap)
        if not bool(keep.any()):
            continue
        cos = (zy[rows[keep]] * zy[cols[keep]]).sum(dim=-1)
        # Want high cosine when gap is small → correlate cos with -gap.
        target = -gap[keep]
        cos_c = cos - cos.mean()
        tgt_c = target - target.mean()
        cos_std = cos_c.std(unbiased=False)
        tgt_std = tgt_c.std(unbiased=False)
        if float(cos_std.detach()) < 1e-8 or float(tgt_std.detach()) < 1e-8:
            continue
        corr = (cos_c * tgt_c).mean() / (cos_std * tgt_std + 1e-8)
        corr = torch.clamp(corr, -1.0, 1.0)
        losses.append(1.0 - corr)

    if not losses:
        return z.new_zeros(())
    return torch.stack(losses).mean()
