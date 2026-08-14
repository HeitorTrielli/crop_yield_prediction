"""Municipality-level pixel prediction accumulator (sum / mean / mixed)."""

from __future__ import annotations

from typing import Sequence

import torch


def _normalize_aggregations(
    aggregation: str | Sequence[str],
) -> tuple[str, ...]:
    if isinstance(aggregation, str):
        aggs = (aggregation,)
    else:
        aggs = tuple(aggregation)
    for a in aggs:
        if a not in ("sum", "mean"):
            raise ValueError(f"aggregation must be 'sum' or 'mean', got {a!r}")
    return aggs


class MunicipalityPixelAccumulator:
    """Accumulate pixel predictions into a municipality-level sum and/or mean.

    Always stores a running sum (+ count). ``value()`` applies per-output
    aggregation when ``aggregation`` is a list.
    """

    def __init__(self, aggregation: str | Sequence[str] = "sum"):
        self.aggregations = _normalize_aggregations(aggregation)
        # Backward-compat scalar attribute used by some call sites / logs.
        self.aggregation = (
            self.aggregations[0] if len(self.aggregations) == 1 else list(self.aggregations)
        )
        self._sum: torch.Tensor | None = None
        self.pixel_count = 0

    def add(self, chunk_predictions: torch.Tensor) -> None:
        chunk_sum = chunk_predictions.sum(dim=0).to(torch.float32)
        if self._sum is None:
            self._sum = chunk_sum
            n_out = int(chunk_sum.numel())
            if len(self.aggregations) == 1 and n_out > 1:
                # Broadcast a single aggregation mode across all output dims.
                self.aggregations = tuple(self.aggregations[0] for _ in range(n_out))
                self.aggregation = (
                    self.aggregations[0]
                    if n_out == 1
                    else list(self.aggregations)
                )
            elif len(self.aggregations) not in (1, n_out):
                raise ValueError(
                    f"Got {n_out} prediction dims but {len(self.aggregations)} "
                    f"aggregations {self.aggregations}"
                )
        else:
            self._sum = self._sum + chunk_sum
        self.pixel_count += int(chunk_predictions.shape[0])

    @property
    def valid(self) -> bool:
        return self._sum is not None and self.pixel_count > 0

    def value(self) -> torch.Tensor | None:
        if not self.valid:
            return None
        out = self._sum.clone()
        count = float(self.pixel_count)
        if len(self.aggregations) == 1:
            if self.aggregations[0] == "mean":
                return out / count
            return out
        for i, agg in enumerate(self.aggregations):
            if agg == "mean":
                out[i] = out[i] / count
        return out

    def detach(self) -> None:
        if self._sum is not None:
            self._sum = self._sum.detach()

    def start_new_grad_segment(self) -> None:
        """Break the autograd chain after backward while keeping running totals."""
        self.detach()

    def is_extreme(self) -> bool:
        val = self.value()
        if val is None or torch.isinf(val).any():
            return True
        if val.dim() == 0 or val.numel() == 1:
            limit = 1e7 if self.aggregations[0] == "sum" else 1e3
            return torch.abs(val).max().item() > limit
        for i, agg in enumerate(self.aggregations):
            limit = 1e7 if agg == "sum" else 1e3
            if torch.abs(val[i]).item() > limit:
                return True
        return False
