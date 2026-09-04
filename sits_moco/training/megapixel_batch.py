"""Fast path when each municipality is a single mega-pixel.

The pixel-chunk pipeline stays the default. This stacks a whole dataloader
batch into one STNet forward (and one backward in training) when every
municipality has at most one pixel.

When the dataset is all mega-pixels, ``ensure_megapixel_ram_cache`` loads every
series into RAM once (tens of MB) so training/eval do no .npy I/O after that.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
import torch

from datasets.pixel_chunk import (
    BatchChunk,
    prepare_chunk_on_device,
    unpack_pixel_chunk,
)

_FASTPATH_LOGGED = False
_RAM_CACHE_ATTR = "_megapixel_ram_cache"
_RAM_CACHE_LOGGED = False


def is_megapixel_batch(num_pixels_list) -> bool:
    if not num_pixels_list:
        return False
    return max(int(n) for n in num_pixels_list) <= 1


def dataset_is_megapixel(dataset) -> bool:
    counts = getattr(dataset, "municipality_pixel_counts", None)
    if not counts:
        return False
    return max(int(n) for n in counts.values()) <= 1


def dataset_supports_megapixel_stack(dataset) -> bool:
    return hasattr(dataset, "_resolve_npy_path") and hasattr(
        dataset, "_transform_chunk"
    )


def _log_fastpath_once(n: int) -> None:
    global _FASTPATH_LOGGED
    if _FASTPATH_LOGGED:
        return
    _FASTPATH_LOGGED = True
    print(
        f"Megapixel batch: stacking {n} municipalities into one forward "
        "(pixel-chunk pipeline unused for this run)",
        flush=True,
    )


def _cache_key(code, year) -> tuple:
    return (str(code), None if year is None else int(year))


def _load_raw_npy(path: str) -> np.ndarray | None:
    try:
        arr = np.load(path)
    except Exception:
        return None
    if arr is None or len(arr) == 0:
        return None
    return arr


def _stack_chunks(chunks: list[BatchChunk]) -> BatchChunk:
    xs, masks, doys, weights = zip(*chunks)
    return (
        torch.cat(xs, dim=0),
        torch.cat(masks, dim=0),
        torch.cat(doys, dim=0),
        torch.cat(weights, dim=0),
    )


def _transform_raw(
    dataset,
    arr: np.ndarray,
    *,
    path: str | None,
    num_periods: int | None,
    reference_date: date | None,
) -> BatchChunk | None:
    if hasattr(dataset, "filter_municipality_data"):
        arr = dataset.filter_municipality_data(arr, cache_key=path)
    if arr is None or len(arr) == 0:
        return None
    period_mode = num_periods is not None and reference_date is not None
    if period_mode:
        for pixel_chunk in dataset.iter_period_pixel_chunks_from_data(
            arr,
            num_periods=int(num_periods),
            chunk_size=max(1, len(arr)),
            reference_date=reference_date,
            cache_key=path,
        ):
            return unpack_pixel_chunk(pixel_chunk)
        return None
    return unpack_pixel_chunk(dataset._transform_chunk(arr))


def ensure_megapixel_ram_cache(dataset, *, workers: int = 8) -> dict | None:
    """
    Preload every mega-pixel series into ``dataset._megapixel_ram_cache``.

    Stores transformed CPU tensors when ``rc``/``interp`` are off (training
    default); otherwise stores raw float32 arrays and transforms per batch.
    Returns the cache dict, or None when the dataset is not all mega-pixels.
    """
    global _RAM_CACHE_LOGGED
    existing = getattr(dataset, _RAM_CACHE_ATTR, None)
    if isinstance(existing, dict):
        return existing
    if not dataset_supports_megapixel_stack(dataset) or not dataset_is_megapixel(dataset):
        return None

    store_transformed = not bool(getattr(dataset, "rc", False)) and not bool(
        getattr(dataset, "interp", False)
    )
    jobs: list[tuple[tuple, str]] = []
    for key in getattr(dataset, "municipality_list", []):
        if isinstance(key, tuple):
            code, year = key[0], key[1]
        else:
            code, year = key, getattr(dataset, "municipality_years", {}).get(key)
        load_year = dataset._resolve_load_year(code, year)
        path = dataset._resolve_npy_path(code, load_year)
        if path is None:
            continue
        jobs.append((_cache_key(code, year), str(path)))

    n_workers = max(1, min(int(workers), len(jobs), 32)) if jobs else 1
    raw_by_path: dict[str, np.ndarray | None] = {}
    unique_paths = list(dict.fromkeys(p for _, p in jobs))
    if n_workers == 1:
        for path in unique_paths:
            raw_by_path[path] = _load_raw_npy(path)
    else:
        with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="mega-preload") as ex:
            futs = {ex.submit(_load_raw_npy, path): path for path in unique_paths}
            for fut, path in futs.items():
                raw_by_path[path] = fut.result()

    cache: dict = {}
    nbytes = 0
    for key, path in jobs:
        arr = raw_by_path.get(path)
        if arr is None or len(arr) == 0:
            continue
        if store_transformed:
            unpacked = _transform_raw(
                dataset, arr, path=path, num_periods=None, reference_date=None
            )
            if unpacked is None:
                continue
            cache[key] = unpacked
            nbytes += sum(int(t.numel() * t.element_size()) for t in unpacked)
        else:
            if hasattr(dataset, "filter_municipality_data"):
                arr = dataset.filter_municipality_data(arr, cache_key=path)
            if arr is None or len(arr) == 0:
                continue
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            cache[key] = arr
            nbytes += int(arr.nbytes)

    setattr(dataset, _RAM_CACHE_ATTR, cache)
    if not _RAM_CACHE_LOGGED:
        _RAM_CACHE_LOGGED = True
        kind = "transformed tensors" if store_transformed else "raw arrays"
        print(
            f"Megapixel RAM cache: {len(cache)} series ({kind}, "
            f"~{nbytes / (1024 * 1024):.1f} MiB) — no .npy I/O after this",
            flush=True,
        )
    return cache


def load_stacked_megapixel_batch(
    dataset,
    municipalities,
    years,
    *,
    skip_muni_indices: set[int] | None = None,
    workers: int = 8,
    num_periods: int | None = None,
    reference_date: date | None = None,
) -> tuple[BatchChunk | None, list[int]]:
    """
    Load every kept municipality and stack to ``[B, T, …]``.

    Prefers the dataset RAM cache when present (and not period-mode).
    Returns ``(stacked_chunk, kept_muni_indices)``. ``stacked_chunk`` is None
    when nothing loaded.
    """
    ensure_megapixel_ram_cache(dataset, workers=workers)
    skip = skip_muni_indices or set()
    period_mode = num_periods is not None and reference_date is not None
    ram = None if period_mode else getattr(dataset, _RAM_CACHE_ATTR, None)

    chunks: list[BatchChunk] = []
    kept: list[int] = []

    if isinstance(ram, dict) and ram:
        for muni_idx, code in enumerate(municipalities):
            if muni_idx in skip:
                continue
            year = years[muni_idx] if years is not None else None
            key = _cache_key(code, year)
            entry = ram.get(key)
            if entry is None and year is not None:
                # Flat-season / year-agnostic keys sometimes stored with None year.
                entry = ram.get(_cache_key(code, None))
            if entry is None:
                continue
            if isinstance(entry, tuple) and len(entry) == 4:
                unpacked = entry
            else:
                unpacked = _transform_raw(
                    dataset,
                    entry,
                    path=None,
                    num_periods=None,
                    reference_date=None,
                )
            if unpacked is None:
                continue
            chunks.append(unpacked)
            kept.append(muni_idx)
        if chunks:
            return _stack_chunks(chunks), kept

    jobs: list[tuple[int, str]] = []
    for muni_idx, code in enumerate(municipalities):
        if muni_idx in skip:
            continue
        year = years[muni_idx] if years is not None else None
        load_year = dataset._resolve_load_year(code, year)
        path = dataset._resolve_npy_path(code, load_year)
        if path is None:
            continue
        jobs.append((muni_idx, str(path)))

    if not jobs:
        return None, []

    n_workers = max(1, min(int(workers), len(jobs), 32))
    raw_by_idx: dict[int, np.ndarray | None] = {}
    if n_workers == 1:
        for muni_idx, path in jobs:
            raw_by_idx[muni_idx] = _load_raw_npy(path)
    else:
        with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="mega-npy") as ex:
            futs = {
                ex.submit(_load_raw_npy, path): muni_idx for muni_idx, path in jobs
            }
            for fut, muni_idx in futs.items():
                raw_by_idx[muni_idx] = fut.result()

    for muni_idx, path in jobs:
        arr = raw_by_idx.get(muni_idx)
        if arr is None or len(arr) == 0:
            continue
        unpacked = _transform_raw(
            dataset,
            arr,
            path=path,
            num_periods=num_periods,
            reference_date=reference_date,
        )
        if unpacked is None:
            continue
        chunks.append(unpacked)
        kept.append(muni_idx)

    if not chunks:
        return None, []
    return _stack_chunks(chunks), kept


def megapixel_on_device(
    stacked: BatchChunk,
    device: torch.device,
    args,
) -> BatchChunk:
    from training_runtime import h2d_pin_host

    pin = bool(h2d_pin_host(args)) and device.type == "cuda"
    return prepare_chunk_on_device(stacked, device, pin_host=pin)


def forward_megapixel_batch(
    model,
    stacked: BatchChunk,
    device: torch.device,
    args,
) -> torch.Tensor:
    from torch.amp import autocast

    from training_runtime import mark_cudagraph_step

    x = megapixel_on_device(stacked, device, args)
    mark_cudagraph_step()
    with autocast("cuda", dtype=torch.bfloat16):
        preds = model(x)
    return preds


def _as_2d(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 0:
        return t.view(1, 1)
    if t.dim() == 1:
        return t.unsqueeze(1)
    return t


def _stats_on(stat, ref: torch.Tensor):
    if stat is None:
        return None
    if not torch.is_tensor(stat):
        return torch.as_tensor(stat, dtype=ref.dtype, device=ref.device)
    return stat.to(device=ref.device, dtype=ref.dtype)


def process_train_megapixel_batch(
    *,
    model,
    optimizer,
    scaler,
    dataset,
    municipalities,
    targets,
    num_pixels_list,
    years,
    device: torch.device,
    args,
    target_mean,
    target_std,
    batch_idx: int,
    run_stats: dict | None = None,
    max_municipalities: int | None = None,
    vram_probe_state: dict | None = None,
    head_output: str,
) -> tuple[float, bool, int]:
    """One forward/backward over the whole municipality batch."""
    from torch.amp import autocast

    from training.pipeline import scan_skip_municipalities
    from training_runtime import mark_cudagraph_step, zero_grad
    from utils_aggregated import HEAD_OUTPUT_ZSCORE

    targets = targets.to(device).float()
    skip = scan_skip_municipalities(municipalities, targets, num_pixels_list)
    load_workers = max(1, int(getattr(args, "workers", 8) or 8))

    stacked, kept = load_stacked_megapixel_batch(
        dataset,
        municipalities,
        years,
        skip_muni_indices=skip,
        workers=load_workers,
    )
    if max_municipalities is not None and kept:
        kept = kept[: int(max_municipalities)]
        if stacked is not None:
            n = len(kept)
            stacked = (stacked[0][:n], stacked[1][:n], stacked[2][:n], stacked[3][:n])

    if stacked is None or not kept:
        zero_grad(optimizer, args)
        if vram_probe_state is None:
            print(
                f"  ⚠️  DEBUG: Skipping optimizer step for batch {batch_idx} - no megapixels loaded"
            )
        return 0.0, False, 0

    _log_fastpath_once(len(kept))
    x = megapixel_on_device(stacked, device, args)
    mark_cudagraph_step()
    with autocast("cuda", dtype=torch.bfloat16):
        preds = model(x)
    if run_stats is not None:
        run_stats["forward_calls"] = run_stats.get("forward_calls", 0) + 1
        run_stats["chunks_processed"] = run_stats.get("chunks_processed", 0) + 1
        run_stats["pixels_processed"] = run_stats.get("pixels_processed", 0) + int(
            preds.shape[0]
        )

    preds = torch.clamp(preds, min=-1e4, max=1e4)
    preds = torch.nan_to_num(preds, nan=0.0, posinf=1e4, neginf=-1e4)
    pred_n = _as_2d(preds).to(torch.float32)
    tgt_n = _as_2d(targets[kept]).to(torch.float32)
    if head_output != HEAD_OUTPUT_ZSCORE and target_mean is not None and target_std is not None:
        mean = _stats_on(target_mean, pred_n)
        std = _stats_on(target_std, pred_n)
        pred_n = (pred_n - mean) / std

    # Match per-municipality MSE + sequential backward: sum of per-row means.
    per_muni = torch.mean((pred_n - tgt_n) ** 2, dim=1)
    loss_sum = per_muni.sum()
    scaler.scale(loss_sum).backward()
    if run_stats is not None:
        run_stats["backward_calls"] = run_stats.get("backward_calls", 0) + 1

    if vram_probe_state is not None and device.type == "cuda":
        from training.vram_adjuster import _pressure_peak, read_vram_pressure

        vram_probe_state["chunks"] = int(vram_probe_state.get("chunks", 0)) + 1
        pressure = read_vram_pressure(device)
        vram_probe_state["last_pressure"] = pressure
        peak = vram_probe_state.get("peak_pressure")
        vram_probe_state["peak_pressure"] = (
            dict(pressure) if peak is None else _pressure_peak(peak, pressure)
        )
        if pressure["driver_used_frac"] > float(vram_probe_state["vram_frac"]):
            vram_probe_state["over_limit"] = True
        min_chunks = int(vram_probe_state.get("min_chunks", 1))
        if vram_probe_state["chunks"] >= min_chunks:
            vram_probe_state["complete"] = True

    batch_has_gradients = True
    if vram_probe_state is None:
        has_nan_grad = any(
            p.grad is not None and torch.isnan(p.grad).any() for p in model.parameters()
        )
        if has_nan_grad:
            zero_grad(optimizer, args)
            batch_has_gradients = False
        else:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            zero_grad(optimizer, args)
            if run_stats is not None:
                run_stats["optimizer_steps"] = run_stats.get("optimizer_steps", 0) + 1
    else:
        zero_grad(optimizer, args)

    return float(loss_sum.detach().item()), batch_has_gradients, len(kept)
