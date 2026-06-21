"""Runtime CUDA memory monitoring and automatic throttling for training."""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch


@dataclass
class MemorySnapshot:
    free_bytes: int
    total_bytes: int
    allocated_bytes: int
    reserved_bytes: int

    @property
    def free_gb(self) -> float:
        return self.free_bytes / (1024**3)

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024**3)

    @property
    def used_fraction(self) -> float:
        if self.total_bytes <= 0:
            return 0.0
        return 1.0 - (self.free_bytes / self.total_bytes)


class CudaMemoryGuard:
    """
    Watch VRAM during training and throttle automatically when pressure builds.

    Actions (escalating):
      1. gc + empty_cache
      2. halve chunks_per_grad (more frequent backward → smaller autograd graphs)
      3. halve pixel chunk_size (smaller forwards)
    """

    def __init__(
        self,
        device: torch.device,
        *,
        enabled: bool = True,
        initial_chunk_size: int = 512,
        initial_chunks_per_grad: int = 10,
        min_chunk_size: int = 128,
        min_chunks_per_grad: int = 1,
        warn_used_fraction: float = 0.82,
        critical_used_fraction: float = 0.90,
        min_free_gb: float = 1.5,
    ) -> None:
        self.device = device
        self.enabled = enabled and device.type == "cuda"
        self.min_chunk_size = min_chunk_size
        self.min_chunks_per_grad = min_chunks_per_grad
        self.warn_used_fraction = warn_used_fraction
        self.critical_used_fraction = critical_used_fraction
        self.min_free_gb = min_free_gb

        self.chunk_size = initial_chunk_size
        self.chunks_per_grad = initial_chunks_per_grad
        self._initial_chunk_size = initial_chunk_size
        self._initial_chunks_per_grad = initial_chunks_per_grad
        self._critical = False
        self._last_status_line: str | None = None

    def snapshot(self) -> MemorySnapshot | None:
        if not self.enabled:
            return None
        free, total = torch.cuda.mem_get_info(self.device)
        return MemorySnapshot(
            free_bytes=free,
            total_bytes=total,
            allocated_bytes=torch.cuda.memory_allocated(self.device),
            reserved_bytes=torch.cuda.memory_reserved(self.device),
        )

    def _pressure_level(self, snap: MemorySnapshot) -> int:
        if snap.free_gb < self.min_free_gb:
            return 2
        if snap.used_fraction >= self.critical_used_fraction:
            return 2
        if snap.used_fraction >= self.warn_used_fraction:
            return 1
        return 0

    def _cleanup(self) -> None:
        gc.collect()
        torch.cuda.empty_cache()

    def _log(self, message: str) -> None:
        if message != self._last_status_line:
            print(message)
            self._last_status_line = message

    def relieve(self, *, reason: str = "periodic") -> bool:
        """
        Run cleanup and downgrade settings if VRAM is tight.
        Returns True if settings were downgraded.
        """
        if not self.enabled:
            return False

        snap = self.snapshot()
        assert snap is not None
        level = self._pressure_level(snap)

        if level == 0:
            self._critical = False
            return False

        self._cleanup()
        snap = self.snapshot()
        assert snap is not None
        level = self._pressure_level(snap)

        if level == 0:
            self._critical = False
            self._log(
                f"  ✓ VRAM recovered after cleanup ({snap.free_gb:.1f} GB free, "
                f"{100 * snap.used_fraction:.0f}% used)"
            )
            return False

        downgraded = False
        self._critical = level >= 2

        if level >= 1 and self.chunks_per_grad > self.min_chunks_per_grad:
            old = self.chunks_per_grad
            self.chunks_per_grad = max(
                self.min_chunks_per_grad,
                self.chunks_per_grad // 2 if level >= 2 else max(self.min_chunks_per_grad, self.chunks_per_grad - 2),
            )
            if self.chunks_per_grad != old:
                downgraded = True

        snap = self.snapshot()
        assert snap is not None
        if self._pressure_level(snap) >= 2 and self.chunk_size > self.min_chunk_size:
            old = self.chunk_size
            self.chunk_size = max(self.min_chunk_size, self.chunk_size // 2)
            if self.chunk_size != old:
                downgraded = True

        if downgraded or level >= 1:
            snap = self.snapshot()
            assert snap is not None
            self._log(
                f"  ⚠️  VRAM pressure ({reason}): {snap.free_gb:.1f} GB free, "
                f"{100 * snap.used_fraction:.0f}% used → "
                f"chunk_size={self.chunk_size}, chunks_per_grad={self.chunks_per_grad}"
                + (" [critical]" if self._critical else "")
            )
        return downgraded

    def before_batch(self) -> None:
        self.relieve(reason="batch start")

    def after_optimizer_step(self) -> None:
        self.relieve(reason="optimizer step")

    def after_backward(self) -> None:
        if self._critical:
            self._cleanup()

    def effective_chunks_per_grad(self) -> int:
        if self._critical:
            return self.min_chunks_per_grad
        return self.chunks_per_grad

    def status_suffix(self) -> str:
        if not self.enabled:
            return ""
        if self.chunk_size == self._initial_chunk_size and self.chunks_per_grad == self._initial_chunks_per_grad:
            return ""
        return f" | vram chunk={self.chunk_size} grad={self.effective_chunks_per_grad()}"
