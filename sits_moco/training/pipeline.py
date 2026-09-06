"""Pixel-chunk pipeline configuration and batch-level chunk iterators."""

from __future__ import annotations

import torch

from datasets.pixel_chunk import (
    PrefetchIterator,
    iter_batch_municipality_chunks,
    iter_batch_municipality_period_chunks,
    iter_npy_array_batch_chunks,
    prepare_chunk_on_device,
)


def pixel_chunk_size(args) -> int:
    return int(args.pixel_chunk_size)


def prefetch_depth(args) -> int:
    return max(0, int(getattr(args, "prefetch_chunks", 2)))


def chunk_debug_syncs(args) -> bool:
    return not getattr(args, "quiet_training", False)


def pipeline_h2d(args, device: torch.device) -> bool:
    if device.type != "cuda" or getattr(args, "disable_pipeline_h2d", False):
        return False
    return chunk_debug_syncs(args) is False


def effective_chunks_per_grad(args) -> int:
    from training.vram_adjuster import effective_chunks_per_grad as _effective

    return _effective(args)


def _chunk_pipeline_has_resolved_sizes(pipe: dict) -> bool:
    """True if this session recorded a real auto-resolve (probe or prior restore)."""
    cpg = pipe.get("chunks_per_grad_effective")
    px = pipe.get("pixel_chunk_size_effective")
    if cpg is None or px is None:
        return False
    trials = pipe.get("search_trials") or []
    if trials:
        return True
    if pipe.get("restored_from_prior_session"):
        return True
    req_c = pipe.get("chunks_per_grad_requested")
    req_p = pipe.get("pixel_chunk_size_requested")
    if req_c is None or req_p is None:
        return False
    # Probe kept requested sizes only if trials exist; without trials, equal
    # requested/effective usually means a resume that fell back to CLI defaults.
    return int(cpg) != int(req_c) or int(px) != int(req_p)


def restore_auto_chunk_pipeline_from_prior_session(
    args,
    training_dir_path,
) -> dict | None:
    """
    On resume/pretrained runs, reuse a prior session's effective chunk sizes.

    Auto VRAM probing is skipped when ``--pretrained`` is set, but the CLI often
    still carries the *requested* grid (e.g. 15×8000) rather than the resolved
    pair (e.g. 14×6500). Prefer the latest ``training/config.json`` session that
    actually probed or restored sizes — not a later resume that only echoed CLI.

    Returns the pipeline dict that was applied, or None if nothing usable was found.
    """
    from pathlib import Path

    from run_paths import load_training_sessions

    config_path = Path(training_dir_path) / "config.json"
    sessions = load_training_sessions(config_path)
    for session in reversed(sessions):
        computed = session.get("computed") or {}
        pipe = computed.get("chunk_pipeline")
        if not isinstance(pipe, dict):
            continue
        if not _chunk_pipeline_has_resolved_sizes(pipe):
            continue
        args.chunks_per_grad = int(pipe["chunks_per_grad_effective"])
        args.pixel_chunk_size = int(pipe["pixel_chunk_size_effective"])
        return dict(pipe)
    return None


def describe_chunk_pipeline(args, device: torch.device) -> str:
    """Human-readable summary of resolved pixel-chunk settings used in training."""
    chunk_px = pixel_chunk_size(args)
    chunks_per_grad = effective_chunks_per_grad(args)
    resolver = getattr(args, "_chunks_per_grad_resolver", None)
    auto_suffix = ""
    if resolver is not None and resolver.enabled:
        from training.vram_adjuster import max_pixels_per_backward

        auto_suffix = (
            f", auto_chunk_pipeline(requested={resolver.requested}x"
            f"{resolver.requested_pixel_chunk}, "
            f"resolved={chunks_per_grad}x{chunk_px}, "
            f"max_px/back={max_pixels_per_backward(chunks_per_grad, chunk_px):,}, "
            f"target={resolver.vram_target_frac:.0%})"
        )
    return (
        "Chunk pipeline (active): "
        f"pixel_chunk_size={chunk_px}, "
        f"chunks_per_grad={chunks_per_grad}{auto_suffix}, "
        f"prefetch_chunks={prefetch_depth(args)}, "
        f"pipeline_h2d={pipeline_h2d(args, device)}, "
        f"max_pixels_per_backward={chunk_px * chunks_per_grad}, "
        f"batch_vram_cleanup={'after each dataloader batch' if chunks_per_grad > 1 else 'default'}"
    )


def scan_skip_municipalities(
    municipalities,
    targets,
    num_pixels_list,
) -> set[int]:
    skip: set[int] = set()
    for muni_idx, municipality_code in enumerate(municipalities):
        num_pixels = num_pixels_list[muni_idx]
        t = targets[muni_idx]
        if torch.isnan(t).any() or torch.isinf(t).any():
            print(
                f"  ❌ DEBUG: Municipality {municipality_code} has invalid target: {t}"
            )
            skip.add(muni_idx)
        elif num_pixels <= 0:
            print(
                f"  ⚠️  Warning: Municipality {municipality_code} has 0 pixels, skipping"
            )
            skip.add(muni_idx)
    return skip


def iter_training_batch_chunks(
    dataset,
    municipalities,
    years,
    num_pixels_list,
    chunk_size: int,
    args,
    device: torch.device,
    *,
    skip_muni_indices: set[int] | None = None,
):
    """Yield ``(muni_idx, chunk)`` for one dataloader batch (cross-muni + mmap lookahead)."""
    from training_runtime import h2d_pin_host

    skip = skip_muni_indices or set()
    use_pipeline = pipeline_h2d(args, device)
    yield from iter_batch_municipality_chunks(
        dataset,
        municipalities,
        years,
        num_pixels_list,
        chunk_size=chunk_size,
        prefetch_depth=prefetch_depth(args),
        device=device,
        pipeline_h2d=use_pipeline,
        pin_host=h2d_pin_host(args),
        skip_muni_indices=skip,
        mmap_lookahead=True,
    )


def iter_inference_period_batch_chunks(
    dataset,
    municipalities,
    years,
    num_pixels_list,
    chunk_size: int,
    args,
    device: torch.device,
    *,
    num_periods: int,
    reference_date,
    skip_muni_indices: set[int] | None = None,
):
    """Yield ``(muni_idx, chunk)`` for incomplete-series inference (same pipeline as training)."""
    from training_runtime import h2d_pin_host

    skip = skip_muni_indices or set()
    use_pipeline = pipeline_h2d(args, device)
    mmap_lookahead = bool(getattr(args, "mmap_lookahead", True))
    yield from iter_batch_municipality_period_chunks(
        dataset,
        municipalities,
        years,
        num_pixels_list,
        chunk_size=chunk_size,
        num_periods=int(num_periods),
        reference_date=reference_date,
        prefetch_depth=prefetch_depth(args),
        device=device,
        pipeline_h2d=use_pipeline,
        pin_host=h2d_pin_host(args),
        skip_muni_indices=skip,
        mmap_lookahead=mmap_lookahead,
    )


def iter_sorted_season_inference_chunks(
    dataset,
    municipalities,
    years,
    num_pixels_list,
    chunk_size: int,
    args,
    device: torch.device,
    *,
    reference_date,
    skip_muni_indices: set[int] | None = None,
):
    """
    Yield ``(muni_idx, cpu_or_gpu_chunk, season_month)`` after one sort of in-season days.

    ``season_month`` is ``[N, T]`` int16 (0 = pad). Every k reuses the same packed chunk.
    """
    from training_runtime import h2d_pin_host

    skip = skip_muni_indices or set()
    mmap_lookahead = bool(getattr(args, "mmap_lookahead", True))
    packed_prefetch = min(max(prefetch_depth(args), 1), 4)
    pin_host = h2d_pin_host(args)
    use_pipeline = pipeline_h2d(args, device)

    def _packed():
        for muni_idx, raw in iter_npy_array_batch_chunks(
            dataset,
            municipalities,
            years,
            num_pixels_list,
            chunk_size=chunk_size,
            prefetch_depth=0,
            skip_muni_indices=skip,
            mmap_lookahead=mmap_lookahead,
        ):
            packed = dataset.pack_sorted_season_chunk(raw, reference_date)
            if packed is None:
                continue
            chunk, season_p = packed
            yield muni_idx, chunk, torch.from_numpy(season_p)

    stream = PrefetchIterator(_packed(), max_pending=packed_prefetch)
    if use_pipeline:
        for muni_idx, chunk, season_p in stream:
            yield (
                muni_idx,
                prepare_chunk_on_device(chunk, device, pin_host=pin_host),
                season_p,
            )
        return
    yield from stream
