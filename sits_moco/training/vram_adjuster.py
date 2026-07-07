"""Resolve chunk pipeline settings before training via a startup VRAM binary search."""

from __future__ import annotations

from typing import Any

import torch


def read_vram_pressure(device: torch.device) -> dict[str, float]:
    """
    Snapshot dedicated VRAM use after synchronizing the current stream.

    ``driver_used_frac`` reflects driver-reported dedicated VRAM use
    (``mem_get_info``), which is the signal for spill into shared/system memory.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "free_mb": 0.0,
            "driver_total_mb": 0.0,
            "reserved_mb": 0.0,
            "props_total_mb": 0.0,
            "driver_used_frac": 0.0,
            "reserved_frac": 0.0,
        }

    torch.cuda.current_stream(device).synchronize()
    free, total = torch.cuda.mem_get_info(device)
    reserved = torch.cuda.memory_reserved(device)
    props_total = float(torch.cuda.get_device_properties(device).total_memory)
    used_driver = float(total - free)
    total_f = float(total)
    mb = 1024.0 * 1024.0
    return {
        "free_mb": free / mb,
        "driver_total_mb": total_f / mb,
        "reserved_mb": reserved / mb,
        "props_total_mb": props_total / mb,
        "driver_used_frac": used_driver / total_f if total_f > 0 else 0.0,
        "reserved_frac": float(reserved) / props_total if props_total > 0 else 0.0,
    }


def cleanup_device_vram(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def reinitialize_training_weights(model: torch.nn.Module) -> None:
    from models.weight_init import weight_init_regression

    base = model._orig_mod if hasattr(model, "_orig_mod") else model
    base.apply(weight_init_regression)


def max_pixels_per_backward(chunks_per_grad: int, pixel_chunk_size: int) -> int:
    return max(1, int(chunks_per_grad)) * max(1, int(pixel_chunk_size))


def probe_chunk_target(args, chunks_per_grad: int) -> int:
    """
    Chunks to process in a VRAM probe before judging fit.

    Full training accumulates gradients across the whole dataloader batch (no
    optimizer step until all municipalities finish). The probe must run through
    at least two backward groups plus prefetch/H2D warmup and record *peak*
    dedicated VRAM — not the post-backward dip after the first group.
    """
    from training.pipeline import prefetch_depth

    cpg = max(1, int(chunks_per_grad))
    prefetch = max(0, prefetch_depth(args))
    return cpg * 2 + prefetch


def _pressure_peak(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    if a.get("driver_used_frac", 0.0) >= b.get("driver_used_frac", 0.0):
        return a
    return b


def _pixel_chunk_grid(p_hi: int, p_lo: int, step: int) -> list[int]:
    step = max(1, int(step))
    sizes: list[int] = []
    p = p_hi
    while p >= p_lo:
        sizes.append(p)
        if p == p_lo:
            break
        p -= step
        if p < p_lo:
            p = p_lo
    return sizes


def build_search_candidates(
    *,
    c_lo: int,
    c_hi: int,
    p_lo: int,
    p_hi: int,
    pixel_step: int,
) -> list[tuple[int, int, int]]:
    """
    Discrete (chunks_per_grad, pixel_chunk_size) grid sorted by product ascending.

    ``pixel_chunk_size`` uses stepped values (default step 100 px) on the search grid.
    """
    p_values = _pixel_chunk_grid(p_hi, p_lo, pixel_step)
    seen: set[tuple[int, int]] = set()
    candidates: list[tuple[int, int, int]] = []
    for c in range(c_hi, c_lo - 1, -1):
        for p in p_values:
            key = (c, p)
            if key in seen:
                continue
            seen.add(key)
            candidates.append((c, p, c * p))
    candidates.sort(key=lambda item: (item[2], item[0]))
    return candidates


class ChunksPerGradResolver:
    """
    Pick the largest chunks_per_grad * pixel_chunk_size that fits dedicated VRAM.

    Binary-searches a coarse discrete grid, then linearly scans the final narrow
    band and a local index window around the best fit to try more nearby pairs.
    """

    def __init__(self, args) -> None:
        self.enabled = bool(getattr(args, "auto_chunks_per_grad", False))
        self.requested = max(1, int(getattr(args, "chunks_per_grad", 1)))
        self.current = self.requested
        self.min_chunks = max(1, int(getattr(args, "auto_chunks_per_grad_min", 1)))
        if self.min_chunks > self.requested:
            self.min_chunks = self.requested
        self.requested_pixel_chunk = max(1, int(getattr(args, "pixel_chunk_size", 1)))
        self.pixel_chunk_size = self.requested_pixel_chunk
        explicit_min_pixel = getattr(args, "auto_pixel_chunk_size_min", None)
        if explicit_min_pixel is not None:
            self.min_pixel_chunk = max(1, int(explicit_min_pixel))
        else:
            self.min_pixel_chunk = max(1000, self.requested_pixel_chunk // 2)
        if self.min_pixel_chunk > self.requested_pixel_chunk:
            self.min_pixel_chunk = self.requested_pixel_chunk
        self.pixel_step = max(
            1, int(getattr(args, "auto_pixel_chunk_size_step", 100))
        )
        self.max_pixels_requested = max_pixels_per_backward(
            self.requested, self.requested_pixel_chunk
        )
        self.vram_frac = float(
            getattr(args, "auto_chunks_per_grad_vram_frac", 0.95)
        )
        self.vram_target_frac = float(
            getattr(args, "auto_chunks_per_grad_vram_target_frac", 0.90)
        )
        if self.vram_target_frac > self.vram_frac:
            self.vram_target_frac = self.vram_frac
        self.candidates = build_search_candidates(
            c_lo=self.min_chunks,
            c_hi=self.requested,
            p_lo=self.min_pixel_chunk,
            p_hi=self.requested_pixel_chunk,
            pixel_step=self.pixel_step,
        )
        self.narrow_linear_threshold = max(
            8, int(getattr(args, "auto_chunks_per_grad_narrow_threshold", 32))
        )
        self.refine_index_radius = max(
            0, int(getattr(args, "auto_chunks_per_grad_refine_radius", 24))
        )
        self.search_trials: list[dict[str, Any]] = []
        self.probe_attempts = 0

    def fits_vram(self, pressure: dict[str, float], *, probe_over_limit: bool) -> bool:
        used = pressure["driver_used_frac"]
        if probe_over_limit or used > self.vram_frac:
            return False
        return used <= self.vram_target_frac

    def _record_trial(
        self,
        *,
        chunks_per_grad: int,
        pixel_chunk_size: int,
        product: int,
        pressure: dict[str, float],
        probe_over_limit: bool,
        chunks_probed: int,
        probe_chunk_goal: int,
        oom: bool = False,
    ) -> None:
        used = pressure["driver_used_frac"]
        last = pressure.get("last_driver_used_frac", used)
        fits = self.fits_vram(pressure, probe_over_limit=probe_over_limit)
        if oom:
            status = "oom"
        elif probe_over_limit or used > self.vram_frac:
            status = "over hard cap"
        elif used > self.vram_target_frac:
            status = "above target"
        else:
            status = "ok"
        self.search_trials.append(
            {
                "chunks_per_grad": chunks_per_grad,
                "pixel_chunk_size": pixel_chunk_size,
                "max_pixels_per_backward": product,
                "probe_chunk_goal": probe_chunk_goal,
                "driver_used_frac_peak": used,
                "driver_used_frac_last": last,
                "reserved_frac": pressure["reserved_frac"],
                "free_mb": pressure["free_mb"],
                "driver_total_mb": pressure["driver_total_mb"],
                "chunks_probed": chunks_probed,
                "probe_over_limit": probe_over_limit,
                "oom": oom,
                "fits": fits,
                "status": status,
            }
        )

    def to_config_dict(self) -> dict[str, Any]:
        return {
            "auto_chunks_per_grad": self.enabled,
            "chunks_per_grad_requested": self.requested,
            "chunks_per_grad_effective": self.current,
            "pixel_chunk_size_requested": self.requested_pixel_chunk,
            "pixel_chunk_size_effective": self.pixel_chunk_size,
            "max_pixels_per_backward_requested": self.max_pixels_requested,
            "max_pixels_per_backward_effective": max_pixels_per_backward(
                self.current, self.pixel_chunk_size
            ),
            "auto_chunks_per_grad_min": self.min_chunks,
            "auto_pixel_chunk_size_min": self.min_pixel_chunk,
            "auto_pixel_chunk_size_step": self.pixel_step,
            "auto_chunks_per_grad_vram_frac": self.vram_frac,
            "auto_chunks_per_grad_vram_target_frac": self.vram_target_frac,
            "auto_chunks_per_grad_narrow_threshold": self.narrow_linear_threshold,
            "auto_chunks_per_grad_refine_radius": self.refine_index_radius,
            "search_strategy": "binary_search_narrow_linear_refine",
            "search_candidate_count": len(self.candidates),
            "search_trials": self.search_trials,
            "probe_attempts": self.probe_attempts,
        }


def init_chunks_per_grad_resolver(args) -> ChunksPerGradResolver | None:
    existing = getattr(args, "_chunks_per_grad_resolver", None)
    if existing is not None:
        return existing
    resolver = ChunksPerGradResolver(args)
    args._chunks_per_grad_resolver = resolver
    return resolver if resolver.enabled else resolver


def effective_chunks_per_grad(args) -> int:
    return max(1, int(getattr(args, "chunks_per_grad", 1)))


def _run_probe_trial(
    *,
    resolver: ChunksPerGradResolver,
    args,
    device: torch.device,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion,
    traindataloader,
    target_mean,
    target_std,
    make_optimizer,
    chunks_per_grad: int,
    pixel_chunk_size: int,
    product: int,
) -> tuple[torch.optim.Optimizer, dict[str, float], bool]:
    """Run one backward-group probe. Returns (optimizer, pressure, fits)."""
    from utils_aggregated import run_training_vram_probe

    resolver.probe_attempts += 1
    args.chunks_per_grad = chunks_per_grad
    args.pixel_chunk_size = pixel_chunk_size
    reinitialize_training_weights(model)
    optimizer = make_optimizer()
    cleanup_device_vram(device)

    oom = False
    chunk_goal = probe_chunk_target(args, chunks_per_grad)
    try:
        probe_state = run_training_vram_probe(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataloader=traindataloader,
            device=device,
            args=args,
            target_mean=target_mean,
            target_std=target_std,
            vram_frac=resolver.vram_frac,
            probe_min_chunks=chunk_goal,
        )
    except torch.cuda.OutOfMemoryError:
        oom = True
        cleanup_device_vram(device)
        probe_state = {
            "over_limit": True,
            "complete": False,
            "chunks": 0,
            "last_pressure": None,
            "peak_pressure": {"driver_used_frac": 1.0},
        }

    peak = probe_state.get("peak_pressure") or probe_state.get("last_pressure")
    pressure = dict(peak or read_vram_pressure(device))
    last = probe_state.get("last_pressure")
    if last is not None:
        pressure["last_driver_used_frac"] = last["driver_used_frac"]
    if oom:
        pressure["driver_used_frac"] = 1.0
        probe_state["over_limit"] = True

    fits = resolver.fits_vram(
        pressure, probe_over_limit=bool(probe_state.get("over_limit") or oom)
    )
    resolver._record_trial(
        chunks_per_grad=chunks_per_grad,
        pixel_chunk_size=pixel_chunk_size,
        product=product,
        pressure=pressure,
        probe_over_limit=bool(probe_state.get("over_limit")),
        chunks_probed=int(probe_state.get("chunks", 0)),
        probe_chunk_goal=chunk_goal,
        oom=oom,
    )
    return optimizer, pressure, fits


def _trial_for_candidate(
    resolver: ChunksPerGradResolver, chunks_per_grad: int, pixel_chunk_size: int
) -> dict[str, Any] | None:
    for trial in reversed(resolver.search_trials):
        if (
            trial["chunks_per_grad"] == chunks_per_grad
            and trial["pixel_chunk_size"] == pixel_chunk_size
        ):
            return trial
    return None


def _print_probe_trial(
    resolver: ChunksPerGradResolver,
    idx: int,
    trial: dict[str, Any],
    *,
    phase: str,
) -> None:
    chunks = trial["chunks_per_grad"]
    pixel = trial["pixel_chunk_size"]
    product = trial["max_pixels_per_backward"]
    last_frac = trial.get("driver_used_frac_last")
    last_note = f", last {last_frac:.1%}" if last_frac is not None else ""
    peak_frac = trial["driver_used_frac_peak"]
    print(
        f"  try [{idx + 1}/{len(resolver.candidates)}] "
        f"chunks_per_grad={chunks} x pixel_chunk_size={pixel} "
        f"({product:,} px/backward, {trial['chunks_probed']}/"
        f"{trial['probe_chunk_goal']} probe chunks): "
        f"VRAM peak {peak_frac:.1%}{last_note} "
        f"({trial['status']}, {phase})",
        flush=True,
    )


def _consider_best(
    *,
    resolver: ChunksPerGradResolver,
    idx: int,
    trial: dict[str, Any],
    best_idx: int | None,
    best_pressure: dict[str, float] | None,
) -> tuple[int | None, dict[str, float] | None]:
    if not trial.get("fits"):
        return best_idx, best_pressure
    _, _, product = resolver.candidates[idx]
    if best_idx is None:
        return idx, {
            "driver_used_frac": trial["driver_used_frac_peak"],
            "last_driver_used_frac": trial.get("driver_used_frac_last"),
        }
    _, _, best_product = resolver.candidates[best_idx]
    if product > best_product:
        return idx, {
            "driver_used_frac": trial["driver_used_frac_peak"],
            "last_driver_used_frac": trial.get("driver_used_frac_last"),
        }
    return best_idx, best_pressure


def _probe_candidate_index(
    *,
    resolver: ChunksPerGradResolver,
    args,
    device: torch.device,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion,
    traindataloader,
    target_mean,
    target_std,
    make_optimizer,
    idx: int,
    probed_indices: set[int],
    phase: str,
) -> tuple[torch.optim.Optimizer, dict[str, Any] | None]:
    chunks, pixel, product = resolver.candidates[idx]
    existing = _trial_for_candidate(resolver, chunks, pixel)
    if existing is not None:
        probed_indices.add(idx)
        return optimizer, existing

    optimizer, pressure, fits = _run_probe_trial(
        resolver=resolver,
        args=args,
        device=device,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        traindataloader=traindataloader,
        target_mean=target_mean,
        target_std=target_std,
        make_optimizer=make_optimizer,
        chunks_per_grad=chunks,
        pixel_chunk_size=pixel,
        product=product,
    )
    probed_indices.add(idx)
    trial = resolver.search_trials[-1]
    _print_probe_trial(resolver, idx, trial, phase=phase)
    cleanup_device_vram(device)
    return optimizer, trial


def resolve_chunks_per_grad_before_training(
    *,
    args,
    device: torch.device,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion,
    traindataloader,
    target_mean,
    target_std,
    make_optimizer,
) -> torch.optim.Optimizer:
    """
    Search a coarse grid for the largest (chunks_per_grad, pixel_chunk_size) pair
    that fits dedicated VRAM.

    Mutates ``args.chunks_per_grad`` and ``args.pixel_chunk_size`` for all epochs.
    """
    resolver = init_chunks_per_grad_resolver(args)
    if resolver is None or not resolver.enabled:
        return optimizer

    candidates = resolver.candidates
    if not candidates:
        return optimizer

    print(
        "Auto chunk pipeline: binary search + local refine on "
        f"{len(candidates)} coarse candidates "
        f"(chunks {resolver.min_chunks}-{resolver.requested}, "
        f"pixel_chunk_size {resolver.min_pixel_chunk}-"
        f"{resolver.requested_pixel_chunk} step {resolver.pixel_step}, "
        f"narrow<={resolver.narrow_linear_threshold}, "
        f"refine±{resolver.refine_index_radius}, "
        f"hard cap={resolver.vram_frac:.1%}, target={resolver.vram_target_frac:.1%})"
    )
    print(
        "  probe simulates mid-batch training: 2 backward groups + prefetch warmup, "
        "peak dedicated VRAM (no optimizer step until probe ends)",
        flush=True,
    )

    lo, hi = 0, len(candidates) - 1
    best_idx: int | None = None
    best_pressure: dict[str, float] | None = None
    probed_indices: set[int] = set()

    while lo <= hi:
        if hi - lo + 1 <= resolver.narrow_linear_threshold:
            print(
                f"  narrow band [{lo + 1}-{hi + 1}/{len(candidates)}]: "
                f"trying all {hi - lo + 1} remaining candidate(s)",
                flush=True,
            )
            for idx in range(lo, hi + 1):
                optimizer, trial = _probe_candidate_index(
                    resolver=resolver,
                    args=args,
                    device=device,
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    traindataloader=traindataloader,
                    target_mean=target_mean,
                    target_std=target_std,
                    make_optimizer=make_optimizer,
                    idx=idx,
                    probed_indices=probed_indices,
                    phase="narrow",
                )
                if trial is None:
                    continue
                best_idx, best_pressure = _consider_best(
                    resolver=resolver,
                    idx=idx,
                    trial=trial,
                    best_idx=best_idx,
                    best_pressure=best_pressure,
                )
            break

        mid = (lo + hi) // 2
        optimizer, trial = _probe_candidate_index(
            resolver=resolver,
            args=args,
            device=device,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            traindataloader=traindataloader,
            target_mean=target_mean,
            target_std=target_std,
            make_optimizer=make_optimizer,
            idx=mid,
            probed_indices=probed_indices,
            phase="binary",
        )
        if trial is None:
            break
        best_idx, best_pressure = _consider_best(
            resolver=resolver,
            idx=mid,
            trial=trial,
            best_idx=best_idx,
            best_pressure=best_pressure,
        )
        if trial["fits"]:
            lo = mid + 1
        else:
            hi = mid - 1

    if resolver.refine_index_radius > 0:
        if best_idx is None:
            refine_lo = 0
            refine_hi = min(len(candidates) - 1, resolver.refine_index_radius)
        else:
            refine_lo = max(0, best_idx - resolver.refine_index_radius)
            refine_hi = min(
                len(candidates) - 1, best_idx + resolver.refine_index_radius
            )
        pending = [
            idx
            for idx in range(refine_lo, refine_hi + 1)
            if idx not in probed_indices
        ]
        if pending:
            print(
                f"  local refine [{refine_lo + 1}-{refine_hi + 1}/"
                f"{len(candidates)}]: trying {len(pending)} more "
                f"candidate(s) around best",
                flush=True,
            )
            for idx in pending:
                optimizer, trial = _probe_candidate_index(
                    resolver=resolver,
                    args=args,
                    device=device,
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    traindataloader=traindataloader,
                    target_mean=target_mean,
                    target_std=target_std,
                    make_optimizer=make_optimizer,
                    idx=idx,
                    probed_indices=probed_indices,
                    phase="refine",
                )
                if trial is None:
                    continue
                best_idx, best_pressure = _consider_best(
                    resolver=resolver,
                    idx=idx,
                    trial=trial,
                    best_idx=best_idx,
                    best_pressure=best_pressure,
                )

    if best_idx is None:
        best_idx = 0
        best_chunks, best_pixel, best_product = candidates[0]
        print(
            "  auto chunk pipeline: nothing fit under target; "
            f"using smallest candidate chunks_per_grad={best_chunks}, "
            f"pixel_chunk_size={best_pixel} ({best_product:,} px/backward)",
            flush=True,
        )
    else:
        best_chunks, best_pixel, best_product = candidates[best_idx]
        print(
            "  auto chunk pipeline: resolved to "
            f"chunks_per_grad={best_chunks}, pixel_chunk_size={best_pixel} "
            f"({best_product:,} px/backward; requested "
            f"{resolver.requested}x{resolver.requested_pixel_chunk}="
            f"{resolver.max_pixels_requested:,} px, "
            f"VRAM peak {best_pressure['driver_used_frac']:.1%} <= target "
            f"{resolver.vram_target_frac:.1%}, "
            f"{resolver.probe_attempts} probe(s))",
            flush=True,
        )

    resolver.current = best_chunks
    resolver.pixel_chunk_size = best_pixel
    args.chunks_per_grad = best_chunks
    args.pixel_chunk_size = best_pixel
    reinitialize_training_weights(model)
    optimizer = make_optimizer()
    cleanup_device_vram(device)
    return optimizer
