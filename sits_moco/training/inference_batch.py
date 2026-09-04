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
from utils_aggregated import (
    HEAD_OUTPUT_RAW,
    denormalize_head_output,
    pixel_pool_for_head,
)


def _entry_pixel_count(dataset, muni_code: str, year: int) -> int:
    if getattr(dataset, "use_multi_year", False):
        key: str | tuple[str, int] = (str(muni_code), int(year))
    else:
        key = str(muni_code)
    return int(dataset.municipality_pixel_counts[key])


def _finalize_pred(
    pred: torch.Tensor,
    *,
    head_output: str = HEAD_OUTPUT_RAW,
    target_mean=None,
    target_std=None,
) -> float | None:
    result = pred.squeeze()
    if result.dim() > 0:
        result = result[0] if len(result) > 0 else torch.tensor(0.0)
    return float(
        denormalize_head_output(
            result.item(), target_mean, target_std, head_output
        )
    )


def _finalize_accumulator(
    acc: MunicipalityPixelAccumulator,
    *,
    head_output: str = HEAD_OUTPUT_RAW,
    target_mean=None,
    target_std=None,
) -> float | None:
    result = acc.value()
    if result is None:
        return None
    return _finalize_pred(
        result, head_output=head_output, target_mean=target_mean, target_std=target_std
    )


def _try_megapixel_inference(
    *,
    model,
    dataset,
    entries: list[tuple[str, int]],
    municipalities: list[str],
    years,
    num_pixels_list: list[int],
    device: torch.device,
    args,
    head_output: str,
    target_mean,
    target_std,
    num_periods: int | None = None,
    reference_date: date | None = None,
) -> dict[tuple[str, int], float] | None:
    from training.megapixel_batch import (
        _log_fastpath_once,
        dataset_supports_megapixel_stack,
        forward_megapixel_batch,
        is_megapixel_batch,
        load_stacked_megapixel_batch,
    )

    if not (
        is_megapixel_batch(num_pixels_list) and dataset_supports_megapixel_stack(dataset)
    ):
        return None
    stacked, kept = load_stacked_megapixel_batch(
        dataset,
        municipalities,
        years,
        workers=max(1, int(getattr(args, "workers", 8) or 8)),
        num_periods=num_periods,
        reference_date=reference_date,
    )
    if stacked is None or not kept:
        return None
    _log_fastpath_once(len(kept))
    model.eval()
    with torch.no_grad():
        preds = forward_megapixel_batch(model, stacked, device, args)
    out: dict[tuple[str, int], float] = {}
    for i, muni_idx in enumerate(kept):
        value = _finalize_pred(
            preds[i],
            head_output=head_output,
            target_mean=target_mean,
            target_std=target_std,
        )
        if value is not None:
            out[entries[muni_idx]] = value
    return out


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
    head_output: str = HEAD_OUTPUT_RAW,
    target_mean=None,
    target_std=None,
) -> dict[tuple[str, int], float]:
    """
    Forward a batch of municipality-years with incomplete-series filtering.

    Uses the same cross-muni prefetch/H2D pipeline as validation training.
    """
    municipalities = [code for code, _year in entries]
    years = [year for _code, year in entries]
    num_pixels_list = [
        _entry_pixel_count(dataset, code, year) for code, year in entries
    ]
    mega = _try_megapixel_inference(
        model=model,
        dataset=dataset,
        entries=entries,
        municipalities=municipalities,
        years=years,
        num_pixels_list=num_pixels_list,
        device=device,
        args=args,
        head_output=head_output,
        target_mean=target_mean,
        target_std=target_std,
        num_periods=num_periods,
        reference_date=reference_date,
    )
    if mega is not None:
        return mega

    from torch.amp import autocast

    pool = pixel_pool_for_head(aggregation, head_output)
    accumulators = {
        muni_idx: MunicipalityPixelAccumulator(pool)
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
        value = _finalize_accumulator(
            accumulators[muni_idx],
            head_output=head_output,
            target_mean=target_mean,
            target_std=target_std,
        )
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
    head_output: str = HEAD_OUTPUT_RAW,
    target_mean=None,
    target_std=None,
) -> dict[tuple[str, int], float]:
    """Forward a batch using the standard training pixel loader (full series)."""
    municipalities = [code for code, _year in entries]
    years = [year for _code, year in entries]
    num_pixels_list = [
        _entry_pixel_count(dataset, code, year) for code, year in entries
    ]
    mega = _try_megapixel_inference(
        model=model,
        dataset=dataset,
        entries=entries,
        municipalities=municipalities,
        years=years,
        num_pixels_list=num_pixels_list,
        device=device,
        args=args,
        head_output=head_output,
        target_mean=target_mean,
        target_std=target_std,
    )
    if mega is not None:
        return mega

    from torch.amp import autocast

    accumulators = {
        muni_idx: MunicipalityPixelAccumulator(pixel_pool_for_head(aggregation, head_output))
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
        value = _finalize_accumulator(
            accumulators[muni_idx],
            head_output=head_output,
            target_mean=target_mean,
            target_std=target_std,
        )
        if value is not None:
            out[entry] = value
    return out
