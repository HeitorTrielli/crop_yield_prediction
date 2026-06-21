"""Batched pixel-chunk helpers: unpack, prefetch, and GPU transfer."""

from __future__ import annotations

import queue
import threading
from typing import Iterator

import torch

BatchChunk = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def is_batched_pixel_chunk(pixel_chunk) -> bool:
    return (
        isinstance(pixel_chunk, tuple)
        and len(pixel_chunk) == 4
        and torch.is_tensor(pixel_chunk[0])
        and pixel_chunk[0].dim() == 3
    )


def pixel_chunk_batch_size(pixel_chunk) -> int:
    if is_batched_pixel_chunk(pixel_chunk):
        return int(pixel_chunk[0].shape[0])
    return len(pixel_chunk)


def unpack_pixel_chunk(pixel_chunk):
    """
    Return (chunk_x, chunk_mask, chunk_doy, chunk_weight) or None if too small.
    Accepts batched 4-tuple or legacy list of 4-tuples.
    """
    if is_batched_pixel_chunk(pixel_chunk):
        chunk_x, chunk_mask, chunk_doy, chunk_weight = pixel_chunk
        if chunk_x.shape[0] < 2:
            return None
        return chunk_x, chunk_mask, chunk_doy, chunk_weight

    if not pixel_chunk or len(pixel_chunk) < 2:
        return None
    chunk_x = torch.stack([p[0] for p in pixel_chunk])
    chunk_mask = torch.stack([p[1] for p in pixel_chunk])
    chunk_doy = torch.stack([p[2] for p in pixel_chunk])
    chunk_weight = torch.stack([p[3] for p in pixel_chunk])
    return chunk_x, chunk_mask, chunk_doy, chunk_weight


class PrefetchIterator:
    """Load the next pixel chunk on a background thread while GPU runs."""

    _sentinel = object()

    def __init__(self, iterable, *, max_pending: int = 2):
        self._iter = iter(iterable)
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(max_pending)))
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        try:
            for item in self._iter:
                self._queue.put(item)
        finally:
            self._queue.put(self._sentinel)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if item is self._sentinel:
            raise StopIteration
        return item


def iter_municipality_pixel_chunks(
    dataset,
    municipality_code,
    year,
    chunk_size: int,
    *,
    prefetch: bool = True,
) -> Iterator:
    source = dataset.load_pixels_from_municipality(
        municipality_code, year=year, chunk_size=chunk_size
    )
    if prefetch:
        source = PrefetchIterator(source, max_pending=2)
    yield from source


def move_batched_chunk_to_device(
    chunk_x: torch.Tensor,
    chunk_mask: torch.Tensor,
    chunk_doy: torch.Tensor,
    chunk_weight: torch.Tensor,
    device: torch.device,
    *,
    non_blocking: bool = True,
) -> BatchChunk:
    use_non_blocking = non_blocking and device.type == "cuda"
    if use_non_blocking:
        chunk_x = chunk_x.pin_memory()
        chunk_mask = chunk_mask.pin_memory()
        chunk_doy = chunk_doy.pin_memory()
        chunk_weight = chunk_weight.pin_memory()
        return (
            chunk_x.to(device, non_blocking=True),
            chunk_mask.to(device, non_blocking=True),
            chunk_doy.to(device, non_blocking=True),
            chunk_weight.to(device, non_blocking=True),
        )
    return (
        chunk_x.to(device),
        chunk_mask.to(device),
        chunk_doy.to(device),
        chunk_weight.to(device),
    )


def prepare_chunk_on_device(unpacked, device: torch.device) -> BatchChunk:
    chunk_x, chunk_mask, chunk_doy, chunk_weight = unpacked
    if chunk_x.dim() == 2:
        chunk_x = chunk_x.unsqueeze(0)
        chunk_mask = chunk_mask.unsqueeze(0)
        chunk_doy = chunk_doy.unsqueeze(0)
        chunk_weight = chunk_weight.unsqueeze(0)
    return move_batched_chunk_to_device(
        chunk_x,
        chunk_mask,
        chunk_doy,
        chunk_weight,
        device,
        non_blocking=device.type == "cuda",
    )
