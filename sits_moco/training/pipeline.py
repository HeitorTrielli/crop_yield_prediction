"""Pixel-chunk pipeline configuration and batch-level chunk iterators."""

from __future__ import annotations

import torch

from datasets.pixel_chunk import iter_batch_municipality_chunks


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
