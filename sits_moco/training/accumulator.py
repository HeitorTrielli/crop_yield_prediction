"""Municipality-level pixel prediction accumulator (sum / mean)."""

from __future__ import annotations

import torch


class MunicipalityPixelAccumulator:
    """Accumulate pixel predictions into a municipality-level sum or mean."""

    def __init__(self, aggregation: str = "sum"):
        if aggregation not in ("sum", "mean"):
            raise ValueError(
                f"aggregation must be 'sum' or 'mean', got {aggregation!r}"
            )
        self.aggregation = aggregation
        self._sum: torch.Tensor | None = None
        self.pixel_count = 0

    def add(self, chunk_predictions: torch.Tensor) -> None:
        chunk_sum = chunk_predictions.sum(dim=0).to(torch.float32)
        if self._sum is None:
            self._sum = chunk_sum
        else:
            self._sum = self._sum + chunk_sum
        self.pixel_count += int(chunk_predictions.shape[0])

    @property
    def valid(self) -> bool:
        return self._sum is not None and self.pixel_count > 0

    def value(self) -> torch.Tensor | None:
        if not self.valid:
            return None
        if self.aggregation == "sum":
            return self._sum
        return self._sum / float(self.pixel_count)

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
        limit = 1e7 if self.aggregation == "sum" else 1e3
        return torch.abs(val).max().item() > limit
