"""
Training runtime options for I/O, H2D, and VRAM behavior.

Production defaults (Jun 2026, ~454k px/s on RTX 5090 / WSL, batchsize 32 benchmark):
  cross-muni pixel pipeline + mmap lookahead, prefetch_chunks=4, workers=4,
  skip_batch_empty_cache, zero_grad(set_to_none=True), persistent_workers,
  dataloader_prefetch_factor=4, TF32. pixel_chunk_size/chunks_per_grad defaults are CLI-only.

Use --batch-empty-cache / --legacy-zero-grad to restore pre-optimization behavior.
Rejected on WSL in benchmarks (not forced): chunks_per_grad>1, h2d_pin_host, compile reduce-overhead, 24k/32k chunks.
"""

from __future__ import annotations

import gc
from typing import Any

import torch

DEFAULTS: dict[str, Any] = {
    "zero_grad_set_to_none": True,
    "skip_batch_empty_cache": True,
    "dataloader_persistent_workers": True,
    "dataloader_prefetch_factor": 4,
    "npy_cache_size": 20,
    "h2d_pin_host": False,
    "no_dataloader_pin_memory": False,
    "compile_mode": "default",
    "batch_empty_cache": False,
    "legacy_zero_grad": False,
    "prefetch_chunks": 4,
}

_CUDAGRAPH_MARK = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)


def _get(args, name: str) -> Any:
    if hasattr(args, name):
        return getattr(args, name)
    return DEFAULTS[name]


def _flag(args, name: str) -> bool:
    return bool(_get(args, name))


def _int(args, name: str) -> int:
    return int(_get(args, name))


def enable_cuda_training_kernels() -> None:
    """Matmul/conv kernel prefs for throughput (no change to bf16 training numerics)."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def apply_runtime_defaults(params: dict) -> dict:
    """Fill missing optimization keys (for YAML / harness)."""
    out = dict(params)
    for key, value in DEFAULTS.items():
        out.setdefault(key, value)
    return out


def dataloader_options_from_args(args) -> dict:
    workers = int(getattr(args, "workers", 0))
    persistent = _flag(args, "dataloader_persistent_workers") and workers > 0
    prefetch = max(1, _int(args, "dataloader_prefetch_factor"))
    return {
        "npy_cache_size": _int(args, "npy_cache_size"),
        "pin_memory": not _flag(args, "no_dataloader_pin_memory"),
        "persistent_workers": persistent,
        "prefetch_factor": prefetch if workers > 0 else None,
    }


def h2d_pin_host(args) -> bool:
    return _flag(args, "h2d_pin_host")


def compile_mode(args) -> str:
    return str(_get(args, "compile_mode") or "default")


def mark_cudagraph_step() -> None:
    """Mark step boundary for torch.compile CUDA graphs (chunked training loops)."""
    if _CUDAGRAPH_MARK is not None:
        _CUDAGRAPH_MARK()


def compile_model(model: torch.nn.Module, args) -> torch.nn.Module:
    """
    torch.compile wrapper.

    ``reduce-overhead`` disables Triton CUDA graphs: our per-chunk forwards reuse
    dynamic buffers and otherwise hit CUDAGraph overwrite errors.
    """
    if not hasattr(torch, "compile"):
        return model
    mode = compile_mode(args)
    if mode == "default":
        return torch.compile(model)
    options: dict[str, Any] | None = None
    if mode == "reduce-overhead":
        options = {"triton.cudagraphs": False}
    return torch.compile(model, mode=mode, options=options)


def zero_grad(optimizer, args) -> None:
    if _flag(args, "legacy_zero_grad") or not _flag(args, "zero_grad_set_to_none"):
        optimizer.zero_grad()
    else:
        optimizer.zero_grad(set_to_none=True)


def _vram_collect_and_empty(device: torch.device) -> None:
    gc.collect()
    torch.cuda.empty_cache()


def batch_vram_cleanup(device: torch.device, args) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    if _flag(args, "batch_empty_cache"):
        _vram_collect_and_empty(device)
        return
    if _flag(args, "skip_batch_empty_cache"):
        return
    _vram_collect_and_empty(device)
