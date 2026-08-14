"""Municipality-level pixel prediction accumulator (sum / mean)."""

from __future__ import annotations

import torch


class MunicipalityPixelAccumulator:
    """
    Accumulate pixel predictions into a municipality-level sum or mean.

    Also tracks sum-of-squares so within-muni variance is available for
    optional auxiliary losses.
    """

    def __init__(self, aggregation: str = "sum"):
        if aggregation not in ("sum", "mean"):
            raise ValueError(
                f"aggregation must be 'sum' or 'mean', got {aggregation!r}"
            )
        self.aggregation = aggregation
        self._sum: torch.Tensor | None = None
        self._sum_sq: torch.Tensor | None = None
        self.pixel_count = 0

    def add(self, chunk_predictions: torch.Tensor) -> None:
        preds = chunk_predictions.to(torch.float32)
        chunk_sum = preds.sum(dim=0)
        chunk_sum_sq = (preds * preds).sum(dim=0)
        if self._sum is None:
            self._sum = chunk_sum
            self._sum_sq = chunk_sum_sq
        else:
            self._sum = self._sum + chunk_sum
            self._sum_sq = self._sum_sq + chunk_sum_sq
        self.pixel_count += int(preds.shape[0])

    @property
    def valid(self) -> bool:
        return self._sum is not None and self.pixel_count > 0

    def value(self) -> torch.Tensor | None:
        if not self.valid:
            return None
        if self.aggregation == "sum":
            return self._sum
        return self._sum / float(self.pixel_count)

    def variance(self) -> torch.Tensor | None:
        """Population variance of pixel predictions."""
        if not self.valid or self._sum_sq is None or self.pixel_count < 1:
            return None
        n = float(self.pixel_count)
        mean = self._sum / n
        var = self._sum_sq / n - mean * mean
        return torch.clamp(var, min=0.0)

    def std(self) -> torch.Tensor | None:
        var = self.variance()
        if var is None:
            return None
        return torch.sqrt(var + 1e-12)

    def detach(self) -> None:
        if self._sum is not None:
            self._sum = self._sum.detach()
        if self._sum_sq is not None:
            self._sum_sq = self._sum_sq.detach()

    def start_new_grad_segment(self) -> None:
        """Break the autograd chain after backward while keeping running totals."""
        self.detach()

    def is_extreme(self) -> bool:
        val = self.value()
        if val is None or torch.isinf(val).any():
            return True
        limit = 1e7 if self.aggregation == "sum" else 1e3
        return torch.abs(val).max().item() > limit
