"""Batched municipality inference (reuses training chunk/H2D pipeline)."""

from __future__ import annotations

from datetime import date

import torch

from datasets.pixel_chunk import prepare_chunk_on_device
from training.accumulator import MunicipalityPixelAccumulator
from training.pipeline import (
    iter_inference_period_batch_chunks,
    iter_training_batch_chunks,
    pipeline_h2d,
)


def _entry_pixel_count(dataset, muni_code: str, year: int) -> int:
    if getattr(dataset, "use_multi_year", False):
        key: str | tuple[str, int] = (str(muni_code), int(year))
    else:
        key = str(muni_code)
    return int(dataset.municipality_pixel_counts[key])


def _finalize_accumulator(acc: MunicipalityPixelAccumulator) -> float | None:
    result = acc.value()
    if result is None:
        return None
    result = result.squeeze()
    if result.dim() > 0:
        result = result[0] if len(result) > 0 else torch.tensor(0.0)
    return float(result.item())


def run_period_inference_batch(
    *,
    model,
    dataset,
    entries: list[tuple[str, int]],
    device: torch.device,
    args,
    chunk_size: int,
    num_periods: int,
    reference_date: date,
    aggregation: str,
) -> dict[tuple[str, int], float]:
    """
    Forward a batch of municipality-years with incomplete-series filtering.

    Uses the same cross-muni prefetch/H2D pipeline as validation training.
    """
    from torch.amp import autocast

    municipalities = [code for code, _year in entries]
    years = [year for _code, year in entries]
    num_pixels_list = [
        _entry_pixel_count(dataset, code, year) for code, year in entries
    ]

    accumulators = {
        muni_idx: MunicipalityPixelAccumulator(aggregation)
        for muni_idx in range(len(entries))
    }
    use_pipeline_h2d = pipeline_h2d(args, device)

    chunk_stream = iter_inference_period_batch_chunks(
        dataset,
        municipalities,
        years,
        num_pixels_list,
        chunk_size,
        args,
        device,
        num_periods=num_periods,
        reference_date=reference_date,
    )

    model.eval()
    with torch.no_grad():
        for muni_idx, chunk_item in chunk_stream:
            if use_pipeline_h2d:
                municipality_X_chunk = chunk_item
            else:
                municipality_X_chunk = prepare_chunk_on_device(chunk_item, device)

            with (
                autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else torch.no_grad()
            ):
                chunk_predictions = model(municipality_X_chunk)
            accumulators[muni_idx].add(chunk_predictions)

    out: dict[tuple[str, int], float] = {}
    for muni_idx, entry in enumerate(entries):
        value = _finalize_accumulator(accumulators[muni_idx])
        if value is not None:
            out[entry] = value
    return out


def run_standard_inference_batch(
    *,
    model,
    dataset,
    entries: list[tuple[str, int]],
    device: torch.device,
    args,
    chunk_size: int,
    aggregation: str,
) -> dict[tuple[str, int], float]:
    """Forward a batch using the standard training pixel loader (full series)."""
    from torch.amp import autocast

    municipalities = [code for code, _year in entries]
    years = [year for _code, year in entries]
    num_pixels_list = [
        _entry_pixel_count(dataset, code, year) for code, year in entries
    ]

    accumulators = {
        muni_idx: MunicipalityPixelAccumulator(aggregation)
        for muni_idx in range(len(entries))
    }
    use_pipeline_h2d = pipeline_h2d(args, device)

    chunk_stream = iter_training_batch_chunks(
        dataset,
        municipalities,
        years,
        num_pixels_list,
        chunk_size,
        args,
        device,
        skip_muni_indices=set(),
    )

    model.eval()
    with torch.no_grad():
        for muni_idx, chunk_item in chunk_stream:
            if use_pipeline_h2d:
                municipality_X_chunk = chunk_item
            else:
                municipality_X_chunk = prepare_chunk_on_device(chunk_item, device)

            with (
                autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else torch.no_grad()
            ):
                chunk_predictions = model(municipality_X_chunk)
            accumulators[muni_idx].add(chunk_predictions)

    out: dict[tuple[str, int], float] = {}
    for muni_idx, entry in enumerate(entries):
        value = _finalize_accumulator(accumulators[muni_idx])
        if value is not None:
            out[entry] = value
    return out
