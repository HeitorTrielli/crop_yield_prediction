"""
Shared VRAM measurement helpers for training-loop profiling.

These utilities mirror the real aggregated training path in utils_aggregated.py
so we can tell apart:
  - live tensor retention (autograd / references)
  - PyTorch CUDA caching-allocator growth (reserved >> active)
  - pipeline overlap (H2D double-buffer + prefetch)
"""

from __future__ import annotations

import csv
import gc
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import torch


@dataclass
class VramSample:
    """One point on a VRAM timeline."""

    step: str
    chunk_idx: int = -1
    muni_idx: int = -1
    allocated_mb: float = 0.0
    reserved_mb: float = 0.0
    active_mb: float = 0.0
    inactive_mb: float = 0.0
    peak_allocated_mb: float = 0.0
    elapsed_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def cache_overhead_mb(self) -> float:
        """Reserved minus actively used — classic PyTorch pool growth."""
        return max(0.0, self.reserved_mb - self.active_mb)


def _bytes_to_mb(n: int | float) -> float:
    return float(n) / (1024.0 * 1024.0)


def read_vram(device: torch.device) -> dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "active_mb": 0.0,
            "inactive_mb": 0.0,
            "peak_allocated_mb": 0.0,
        }
    stats = torch.cuda.memory_stats(device)
    active = stats.get("active_bytes.all.current", 0)
    reserved = torch.cuda.memory_reserved(device)
    return {
        "allocated_mb": _bytes_to_mb(torch.cuda.memory_allocated(device)),
        "reserved_mb": _bytes_to_mb(reserved),
        "active_mb": _bytes_to_mb(active),
        "inactive_mb": _bytes_to_mb(max(0, reserved - active)),
        "peak_allocated_mb": _bytes_to_mb(torch.cuda.max_memory_allocated(device)),
    }


class VramTracer:
    """Record VRAM samples with monotonic timestamps."""

    def __init__(self, device: torch.device):
        self.device = device
        self.samples: list[VramSample] = []
        self._t0 = time.perf_counter()

    def reset_peak(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def record(
        self,
        step: str,
        *,
        chunk_idx: int = -1,
        muni_idx: int = -1,
        sync: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> VramSample:
        if sync and self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        vram = read_vram(self.device)
        sample = VramSample(
            step=step,
            chunk_idx=chunk_idx,
            muni_idx=muni_idx,
            elapsed_s=time.perf_counter() - self._t0,
            extra=extra or {},
            **vram,
        )
        self.samples.append(sample)
        return sample

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for s in self.samples:
            row = asdict(s)
            extra = row.pop("extra", {})
            row.update(extra)
            rows.append(row)
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(s) for s in self.samples]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def cleanup_vram(
    device: torch.device,
    *,
    empty_cache: bool = False,
    gc_collect: bool = False,
    synchronize: bool = False,
) -> None:
    """Configurable release — mirrors / extends utils_aggregated._batch_vram_cleanup."""
    if synchronize and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    if gc_collect:
        gc.collect()
    if empty_cache and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def summarize_trace(samples: list[VramSample]) -> dict[str, Any]:
    if not samples:
        return {}
    reserved = [s.reserved_mb for s in samples]
    active = [s.active_mb for s in samples]
    cache = [s.cache_overhead_mb for s in samples]
    by_step: dict[str, list[float]] = {}
    for s in samples:
        by_step.setdefault(s.step, []).append(s.reserved_mb)
    return {
        "num_samples": len(samples),
        "reserved_mb_min": min(reserved),
        "reserved_mb_max": max(reserved),
        "reserved_mb_end": reserved[-1],
        "active_mb_max": max(active),
        "cache_overhead_mb_max": max(cache),
        "reserved_ramp_mb": max(reserved) - min(reserved),
        "reserved_by_step_max": {k: max(v) for k, v in by_step.items()},
    }


def print_summary(title: str, summary: dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    if not summary:
        print("(no samples)")
        return
    print(
        f"  reserved: {summary['reserved_mb_min']:.0f} → {summary['reserved_mb_max']:.0f} MB "
        f"(ramp {summary['reserved_ramp_mb']:.0f} MB, end {summary['reserved_mb_end']:.0f} MB)"
    )
    print(
        f"  active peak {summary['active_mb_max']:.0f} MB, "
        f"cache overhead peak {summary['cache_overhead_mb_max']:.0f} MB"
    )
    print("  reserved peak by step:")
    for step, peak in sorted(
        summary.get("reserved_by_step_max", {}).items()),
        key=lambda x: -x[1],
    ):
        print(f"    {step:28s} {peak:8.0f} MB")


CleanupFn = Callable[[torch.device], None]


def make_cleanup_fn(
    *,
    empty_cache: bool = False,
    gc_collect: bool = False,
    synchronize: bool = False,
) -> CleanupFn:
    def _fn(device: torch.device) -> None:
        cleanup_vram(
            device,
            empty_cache=empty_cache,
            gc_collect=gc_collect,
            synchronize=synchronize,
        )

    return _fn


# Named strategies aligned with suspected Task Manager behaviour.
CLEANUP_STRATEGIES: dict[str, CleanupFn] = {
    "none": make_cleanup_fn(),
    "sync_after_chunk": make_cleanup_fn(synchronize=True),
    "gc_after_chunk": make_cleanup_fn(gc_collect=True),
    "empty_cache_after_chunk": make_cleanup_fn(empty_cache=True, gc_collect=True),
    "batch_end_only": make_cleanup_fn(),  # caller invokes empty_cache once at batch end
    "batch_end_empty_cache": make_cleanup_fn(empty_cache=True, gc_collect=True),
}


def batch_end_cleanup_current(device: torch.device) -> None:
    """Same as utils_aggregated._batch_vram_cleanup (production default)."""
    cleanup_vram(device, empty_cache=True, gc_collect=True)
