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
    """
    Pull items from ``iterable`` on a background thread while the consumer runs.

    Used to overlap CPU .npy read + transform + torch.stack with GPU forward/backward.
    """

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
        except Exception as exc:
            self._queue.put(exc)
        finally:
            self._queue.put(self._sentinel)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if item is self._sentinel:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        return item


class GpuH2DPipelineIterator:
    """
    Double-buffer H2D copies on a side CUDA stream.

    While the default stream runs forward/backward on chunk *i*, chunk *i+1* is
    copied host→device on ``h2d_stream``. Same chunk order and tensors as a
    synchronous ``prepare_chunk_on_device`` loop.
    """

    def __init__(self, cpu_chunks: Iterator[BatchChunk], device: torch.device):
        self._cpu_iter = iter(cpu_chunks)
        self._device = device
        self._h2d_stream = torch.cuda.Stream(device=device)
        self._ready: BatchChunk | None = None
        self._load_next()

    def _transfer(self, unpacked: BatchChunk) -> BatchChunk:
        with torch.cuda.stream(self._h2d_stream):
            return prepare_chunk_on_device(unpacked, self._device)

    def _load_next(self) -> None:
        try:
            unpacked = next(self._cpu_iter)
        except StopIteration:
            self._ready = None
            return
        self._ready = self._transfer(unpacked)

    def __iter__(self):
        return self

    def __next__(self) -> BatchChunk:
        if self._ready is None:
            raise StopIteration
        torch.cuda.current_stream(self._device).wait_stream(self._h2d_stream)
        batch = self._ready
        self._load_next()
        return batch


def _iter_prepared_from_raw(raw_source) -> Iterator[BatchChunk]:
    for pixel_chunk in raw_source:
        unpacked = unpack_pixel_chunk(pixel_chunk)
        if unpacked is not None:
            yield unpacked


def iter_municipality_pixel_chunks(
    dataset,
    municipality_code,
    year,
    chunk_size: int,
    *,
    prefetch_depth: int = 2,
    device: torch.device | None = None,
    pipeline_h2d: bool = False,
) -> Iterator[BatchChunk]:
    """
    Yield pixel chunks for one municipality.

    Default: CPU tensors (chunk_x, mask, doy, weight). With ``pipeline_h2d=True``
    and a CUDA ``device``, yields GPU-ready batches using a side stream to overlap
    H2D with forward/backward on the default stream.

    When ``prefetch_depth > 0``, the next chunk is loaded/transformed/stacked on a
    background thread while the GPU processes the current chunk.
    """
    raw = dataset.load_pixels_from_municipality(
        municipality_code, year=year, chunk_size=chunk_size
    )
    prepared = _iter_prepared_from_raw(raw)
    depth = int(prefetch_depth)
    if depth > 0:
        prepared = PrefetchIterator(prepared, max_pending=depth)
    if (
        pipeline_h2d
        and device is not None
        and device.type == "cuda"
        and torch.cuda.is_available()
    ):
        yield from GpuH2DPipelineIterator(prepared, device)
        return
    yield from prepared


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
