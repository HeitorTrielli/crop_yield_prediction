"""Validation batch: cross-muni pipeline, GPU-side prediction concat."""

from __future__ import annotations

import torch

from datasets.pixel_chunk import prepare_chunk_on_device
from training.pipeline import (
    iter_training_batch_chunks,
    pipeline_h2d,
)


def run_eval_batch(
    *,
    model,
    dataset,
    municipalities,
    targets,
    num_pixels_list,
    years,
    device: torch.device,
    args,
    chunk_size: int,
) -> list[torch.Tensor]:
    """
    Forward all municipalities in a batch; return per-muni pixel predictions on GPU.

    Uses the same cross-muni CPU/H2D pipeline as training (no per-chunk CPU sync).
    """
    from torch.amp import autocast

    use_pipeline_h2d = pipeline_h2d(args, device)

    predictions_by_muni: dict[int, list[torch.Tensor]] = {}
    current_muni = -1
    muni_chunks: list[torch.Tensor] = []

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

    for stream_muni_idx, chunk_item in chunk_stream:
        if stream_muni_idx != current_muni:
            if current_muni >= 0 and muni_chunks:
                predictions_by_muni[current_muni] = muni_chunks
            current_muni = stream_muni_idx
            muni_chunks = []

        if use_pipeline_h2d:
            municipality_X_chunk = chunk_item
        else:
            municipality_X_chunk = prepare_chunk_on_device(chunk_item, device)

        with autocast("cuda", dtype=torch.bfloat16):
            chunk_predictions = model(municipality_X_chunk)
        muni_chunks.append(chunk_predictions)

    if current_muni >= 0 and muni_chunks:
        predictions_by_muni[current_muni] = muni_chunks

    predictions_list: list[torch.Tensor] = []
    for muni_idx in range(len(municipalities)):
        chunks = predictions_by_muni.get(muni_idx)
        if not chunks:
            continue
        predictions_list.append(torch.cat(chunks, dim=0))

    return predictions_list
