"""Batched pixel-chunk helpers: unpack, prefetch, and GPU transfer."""

from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Iterator

import numpy as np
import torch

BatchChunk = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
TaggedChunk = tuple[int, BatchChunk]
TaggedGpuChunk = tuple[int, BatchChunk]


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
        if chunk_x.shape[0] < 1:
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

    def __init__(self, cpu_chunks: Iterator[BatchChunk], device: torch.device, *, pin_host: bool = False):
        self._cpu_iter = iter(cpu_chunks)
        self._device = device
        self._pin_host = pin_host
        self._h2d_stream = torch.cuda.Stream(device=device)
        self._ready: BatchChunk | None = None
        self._load_next()

    def _transfer(self, unpacked: BatchChunk) -> BatchChunk:
        with torch.cuda.stream(self._h2d_stream):
            # Pinned host pages on 30k+ px chunks can exhaust locked RAM and freeze WSL;
            # side-stream copies still overlap with default-stream compute.
            return prepare_chunk_on_device(unpacked, self._device, pin_host=self._pin_host)

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
        default_stream = torch.cuda.current_stream(self._device)
        default_stream.wait_stream(self._h2d_stream)
        batch = self._ready
        # Next H2D copy may start before the default stream finishes forward on
        # ``batch``; record_stream prevents the caching allocator from reusing
        # this storage while it is still read on the default stream.
        for t in batch:
            t.record_stream(default_stream)
        self._load_next()
        return batch


class TaggedGpuH2DPipelineIterator:
    """Like GpuH2DPipelineIterator but preserves ``muni_idx`` tags across H2D."""

    def __init__(
        self,
        tagged_cpu_chunks: Iterator[TaggedChunk],
        device: torch.device,
        *,
        pin_host: bool = False,
    ):
        self._cpu_iter = iter(tagged_cpu_chunks)
        self._device = device
        self._pin_host = pin_host
        self._h2d_stream = torch.cuda.Stream(device=device)
        self._ready: TaggedGpuChunk | None = None
        self._load_next()

    def _transfer(self, muni_idx: int, unpacked: BatchChunk) -> TaggedGpuChunk:
        with torch.cuda.stream(self._h2d_stream):
            gpu = prepare_chunk_on_device(unpacked, self._device, pin_host=self._pin_host)
        return muni_idx, gpu

    def _load_next(self) -> None:
        try:
            muni_idx, unpacked = next(self._cpu_iter)
        except StopIteration:
            self._ready = None
            return
        self._ready = self._transfer(muni_idx, unpacked)

    def __iter__(self):
        return self

    def __next__(self) -> TaggedGpuChunk:
        if self._ready is None:
            raise StopIteration
        default_stream = torch.cuda.current_stream(self._device)
        default_stream.wait_stream(self._h2d_stream)
        muni_idx, batch = self._ready
        for t in batch:
            t.record_stream(default_stream)
        self._load_next()
        return muni_idx, batch


def _iter_tagged_chunks_with_mmap_lookahead(
    dataset,
    entries: list[tuple[int, object, object]],
    *,
    chunk_size: int,
) -> Iterator[TaggedChunk]:
    """
    Transform chunks sequentially while the next municipality .npy is mmap'd on a side thread.
    """
    if not entries:
        return

    def _mmap(entry: tuple[int, object, object]) -> tuple[int, object | None]:
        muni_idx, municipality_code, year = entry
        return muni_idx, dataset.mmap_municipality(municipality_code, year=year)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="npy-mmap") as executor:
        pending = executor.submit(_mmap, entries[0])
        for i, _entry in enumerate(entries):
            muni_idx, municipality_data = pending.result()
            if i + 1 < len(entries):
                pending = executor.submit(_mmap, entries[i + 1])

            if municipality_data is None:
                continue
            muni_code = entries[i][1]
            year = entries[i][2]
            year_resolved = dataset._resolve_load_year(muni_code, year)
            cache_key = dataset._resolve_npy_path(muni_code, year_resolved)
            municipality_data = dataset.filter_municipality_data(
                municipality_data, cache_key=cache_key
            )
            if municipality_data is None or len(municipality_data) == 0:
                continue
            num_pixels = len(municipality_data)
            for start in range(0, num_pixels, chunk_size):
                end = min(start + chunk_size, num_pixels)
                unpacked = unpack_pixel_chunk(
                    dataset._transform_chunk(municipality_data[start:end])
                )
                if unpacked is not None:
                    yield muni_idx, unpacked


def _iter_tagged_period_chunks_with_mmap_lookahead(
    dataset,
    entries: list[tuple[int, object, object]],
    *,
    chunk_size: int,
    num_periods: int,
    reference_date: date,
) -> Iterator[TaggedChunk]:
    """Period-filtered chunks with the same mmap-ahead pattern as training."""
    if not entries:
        return

    def _mmap(entry: tuple[int, object, object]) -> tuple[int, object | None]:
        muni_idx, municipality_code, year = entry
        return muni_idx, dataset.mmap_municipality(municipality_code, year=year)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="npy-mmap") as executor:
        pending = executor.submit(_mmap, entries[0])
        for i, _entry in enumerate(entries):
            muni_idx, municipality_data = pending.result()
            if i + 1 < len(entries):
                pending = executor.submit(_mmap, entries[i + 1])

            if municipality_data is None:
                continue
            muni_code = entries[i][1]
            year = entries[i][2]
            year_resolved = dataset._resolve_load_year(muni_code, year)
            cache_key = dataset._resolve_npy_path(muni_code, year_resolved)
            for pixel_chunk in dataset.iter_period_pixel_chunks_from_data(
                municipality_data,
                num_periods=num_periods,
                chunk_size=chunk_size,
                reference_date=reference_date,
                cache_key=cache_key,
            ):
                unpacked = unpack_pixel_chunk(pixel_chunk)
                if unpacked is not None:
                    yield muni_idx, unpacked


def _iter_npy_arrays_with_mmap_lookahead(
    dataset,
    entries: list[tuple[int, object, object]],
    *,
    chunk_size: int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield contiguous float32 pixel chunks; mmap the next municipality on a side thread."""
    if not entries:
        return

    def _mmap(entry: tuple[int, object, object]) -> tuple[int, object | None]:
        muni_idx, municipality_code, year = entry
        return muni_idx, dataset.mmap_municipality(municipality_code, year=year)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="npy-mmap") as executor:
        pending = executor.submit(_mmap, entries[0])
        for i, _entry in enumerate(entries):
            muni_idx, municipality_data = pending.result()
            if i + 1 < len(entries):
                pending = executor.submit(_mmap, entries[i + 1])
            if municipality_data is None:
                continue
            num_pixels = len(municipality_data)
            for start in range(0, num_pixels, chunk_size):
                end = min(start + chunk_size, num_pixels)
                yield muni_idx, np.ascontiguousarray(
                    municipality_data[start:end], dtype=np.float32
                )


def iter_npy_array_batch_chunks(
    dataset,
    municipalities,
    years,
    num_pixels_list,
    *,
    chunk_size: int,
    prefetch_depth: int = 2,
    skip_muni_indices: set[int] | None = None,
    mmap_lookahead: bool = True,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(muni_idx, float32[N,T,C])`` for one inference batch (one disk read)."""
    skip = skip_muni_indices or set()
    entries: list[tuple[int, object, object]] = []
    for muni_idx, municipality_code in enumerate(municipalities):
        if muni_idx in skip:
            continue
        if num_pixels_list[muni_idx] <= 0:
            continue
        year = years[muni_idx] if years[muni_idx] is not None else None
        entries.append((muni_idx, municipality_code, year))

    if mmap_lookahead and len(entries) > 1:
        stream: Iterator[tuple[int, np.ndarray]] = _iter_npy_arrays_with_mmap_lookahead(
            dataset, entries, chunk_size=chunk_size
        )
    else:
        def _serial() -> Iterator[tuple[int, np.ndarray]]:
            for muni_idx, municipality_code, year in entries:
                municipality_data = dataset.mmap_municipality(
                    municipality_code, year=year
                )
                if municipality_data is None:
                    continue
                num_pixels = len(municipality_data)
                for start in range(0, num_pixels, chunk_size):
                    end = min(start + chunk_size, num_pixels)
                    yield muni_idx, np.ascontiguousarray(
                        municipality_data[start:end], dtype=np.float32
                    )

        stream = _serial()

    depth = min(max(int(prefetch_depth), 0), 4)
    if depth > 0:
        yield from PrefetchIterator(stream, max_pending=depth)
        return
    yield from stream


def _iter_raw_batch_period_cpu_chunks(
    dataset,
    municipalities,
    years,
    num_pixels_list,
    *,
    chunk_size: int,
    num_periods: int,
    reference_date: date,
    skip_muni_indices: set[int] | None = None,
    mmap_lookahead: bool = False,
) -> Iterator[TaggedChunk]:
    """Chain incomplete-series pixel chunks for a batch of municipalities."""
    skip = skip_muni_indices or set()
    entries: list[tuple[int, object, object]] = []
    for muni_idx, municipality_code in enumerate(municipalities):
        if muni_idx in skip:
            continue
        num_pixels = num_pixels_list[muni_idx]
        if num_pixels <= 0:
            continue
        year = years[muni_idx] if years[muni_idx] is not None else None
        entries.append((muni_idx, municipality_code, year))

    if mmap_lookahead and len(entries) > 1:
        yield from _iter_tagged_period_chunks_with_mmap_lookahead(
            dataset,
            entries,
            chunk_size=chunk_size,
            num_periods=num_periods,
            reference_date=reference_date,
        )
        return

    for muni_idx, municipality_code, year in entries:
        raw = dataset.load_pixels_from_municipality_with_periods(
            municipality_code,
            year,
            num_periods,
            chunk_size,
            reference_date,
        )
        for pixel_chunk in raw:
            unpacked = unpack_pixel_chunk(pixel_chunk)
            if unpacked is not None:
                yield muni_idx, unpacked


def iter_batch_municipality_period_chunks(
    dataset,
    municipalities,
    years,
    num_pixels_list,
    *,
    chunk_size: int,
    num_periods: int,
    reference_date: date,
    prefetch_depth: int = 2,
    device: torch.device | None = None,
    pipeline_h2d: bool = False,
    pin_host: bool = False,
    skip_muni_indices: set[int] | None = None,
    mmap_lookahead: bool = False,
) -> Iterator[TaggedChunk | TaggedGpuChunk]:
    """Yield ``(muni_idx, chunk)`` for incomplete-series inference (training pipeline)."""
    stream: Iterator[TaggedChunk] = _iter_raw_batch_period_cpu_chunks(
        dataset,
        municipalities,
        years,
        num_pixels_list,
        chunk_size=chunk_size,
        num_periods=num_periods,
        reference_date=reference_date,
        skip_muni_indices=skip_muni_indices,
        mmap_lookahead=mmap_lookahead,
    )
    depth = int(prefetch_depth)
    if depth > 0:
        stream = PrefetchIterator(stream, max_pending=depth)  # type: ignore[assignment]
    if (
        pipeline_h2d
        and device is not None
        and device.type == "cuda"
        and torch.cuda.is_available()
    ):
        yield from TaggedGpuH2DPipelineIterator(stream, device, pin_host=pin_host)
        return
    yield from stream


def _iter_raw_batch_cpu_chunks(
    dataset,
    municipalities,
    years,
    num_pixels_list,
    *,
    chunk_size: int,
    skip_muni_indices: set[int] | None = None,
    mmap_lookahead: bool = False,
) -> Iterator[TaggedChunk]:
    """Chain all municipalities in one batch (CPU tensors, tagged by muni index)."""
    skip = skip_muni_indices or set()
    entries: list[tuple[int, object, object]] = []
    for muni_idx, municipality_code in enumerate(municipalities):
        if muni_idx in skip:
            continue
        num_pixels = num_pixels_list[muni_idx]
        if num_pixels <= 0:
            continue
        year = years[muni_idx] if years[muni_idx] is not None else None
        entries.append((muni_idx, municipality_code, year))

    if mmap_lookahead and len(entries) > 1:
        yield from _iter_tagged_chunks_with_mmap_lookahead(
            dataset, entries, chunk_size=chunk_size
        )
        return

    for muni_idx, municipality_code, year in entries:
        raw = dataset.load_pixels_from_municipality(
            municipality_code, year=year, chunk_size=chunk_size
        )
        for pixel_chunk in raw:
            unpacked = unpack_pixel_chunk(pixel_chunk)
            if unpacked is not None:
                yield muni_idx, unpacked


def iter_batch_municipality_chunks(
    dataset,
    municipalities,
    years,
    num_pixels_list,
    *,
    chunk_size: int,
    prefetch_depth: int = 2,
    device: torch.device | None = None,
    pipeline_h2d: bool = False,
    pin_host: bool = False,
    skip_muni_indices: set[int] | None = None,
    mmap_lookahead: bool = False,
) -> Iterator[TaggedChunk | TaggedGpuChunk]:
    """
    Yield ``(muni_idx, chunk)`` for a whole dataloader batch.

    Keeps CPU prefetch + H2D pipeline running across municipality boundaries
    (no idle gap while opening the next .npy file).
    """
    stream: Iterator[TaggedChunk] = _iter_raw_batch_cpu_chunks(
        dataset,
        municipalities,
        years,
        num_pixels_list,
        chunk_size=chunk_size,
        skip_muni_indices=skip_muni_indices,
        mmap_lookahead=mmap_lookahead,
    )
    depth = int(prefetch_depth)
    if depth > 0:
        stream = PrefetchIterator(stream, max_pending=depth)  # type: ignore[assignment]
    if (
        pipeline_h2d
        and device is not None
        and device.type == "cuda"
        and torch.cuda.is_available()
    ):
        yield from TaggedGpuH2DPipelineIterator(stream, device, pin_host=pin_host)
        return
    yield from stream


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
    pin_host: bool = False,
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
        yield from GpuH2DPipelineIterator(prepared, device, pin_host=pin_host)
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


def prepare_chunk_on_device(
    unpacked, device: torch.device, *, pin_host: bool = True
) -> BatchChunk:
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
        non_blocking=pin_host and device.type == "cuda",
    )
