#!/usr/bin/env python3
"""
Synthetic micro-benchmark: VRAM vs number of live pixel chunks (no dataset).

Useful to separate "allocator cache ramp" from "N chunks kept in autograd graph"
without loading .npy files. Run on an idle GPU only.

  python scripts/profile_vram_autograd_depth.py --max-chunks 8 --pixels-per-chunk 8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.amp import autocast

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
for _p in (_ROOT, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from models import STNetRegression  # noqa: E402
from models.weight_init import weight_init_regression  # noqa: E402
from profile_vram_utils import cleanup_vram, print_summary, summarize_trace, VramTracer  # noqa: E402
from utils_aggregated import MunicipalityPixelAccumulator  # noqa: E402


def fake_chunk(batch: int, seq: int, fdim: int, device: torch.device):
    x = torch.randn(batch, seq, fdim, device=device, dtype=torch.float32)
    mask = torch.ones(batch, seq, device=device, dtype=torch.float32)
    doy = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
    weight = torch.full((batch, seq), 1.0 / seq, device=device)
    return x, mask, doy, weight


def run_once(
    *,
    model,
    device: torch.device,
    pixels_per_chunk: int,
    seq: int,
    fdim: int,
    max_chunks: int,
    detach_after_backward: bool,
) -> list:
    tracer = VramTracer(device)
    tracer.reset_peak()
    cleanup_vram(device, empty_cache=True, gc_collect=True, synchronize=True)

    acc = MunicipalityPixelAccumulator("sum")
    target = torch.tensor([1e6], device=device)

    for i in range(max_chunks):
        x = fake_chunk(pixels_per_chunk, seq, fdim, device)
        with autocast("cuda", dtype=torch.bfloat16):
            preds = model(x)
        acc.add(preds)
        tracer.record(f"after_forward_{i}", chunk_idx=i)

        loss = torch.nn.functional.mse_loss(
            acc.value().squeeze().float(), target.squeeze()
        )
        loss.backward()
        tracer.record(f"after_backward_{i}", chunk_idx=i)

        if detach_after_backward:
            acc.start_new_grad_segment()
        del x, preds, loss
        tracer.record(f"after_del_{i}", chunk_idx=i)

    cleanup_vram(device, synchronize=True)
    tracer.record("end_sync")
    cleanup_vram(device, empty_cache=True, gc_collect=True, synchronize=True)
    tracer.record("end_empty_cache")
    return tracer.samples


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pixels-per-chunk", type=int, default=8000)
    p.add_argument("--sequencelength", type=int, default=45)
    p.add_argument("--feature-dim", type=int, default=10)
    p.add_argument("--max-chunks", type=int, default=8)
    ns = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")

    device = torch.device("cuda")
    model = STNetRegression(
        input_dim=ns.feature_dim,
        num_outputs=1,
        max_seq_len=ns.sequencelength,
    ).to(device)
    model.apply(weight_init_regression)
    model.train()

    print("=== With detach after each backward (production CHUNKS_PER_GRAD=1) ===")
    s1 = run_once(
        model=model,
        device=device,
        pixels_per_chunk=ns.pixels_per_chunk,
        seq=ns.sequencelength,
        fdim=ns.feature_dim,
        max_chunks=ns.max_chunks,
        detach_after_backward=True,
    )
    print_summary("detach=True", summarize_trace(s1))

    model = STNetRegression(
        input_dim=ns.feature_dim,
        num_outputs=1,
        max_seq_len=ns.sequencelength,
    ).to(device)
    model.apply(weight_init_regression)
    model.train()

    print("\n=== Without detach (simulates CHUNKS_PER_GRAD > 1) ===")
    s2 = run_once(
        model=model,
        device=device,
        pixels_per_chunk=ns.pixels_per_chunk,
        seq=ns.sequencelength,
        fdim=ns.feature_dim,
        max_chunks=ns.max_chunks,
        detach_after_backward=False,
    )
    print_summary("detach=False", summarize_trace(s2))

    r1 = summarize_trace(s1).get("reserved_ramp_mb", 0)
    r2 = summarize_trace(s2).get("reserved_ramp_mb", 0)
    print(
        f"\nReserved ramp: detach=True {r1:.0f} MB vs detach=False {r2:.0f} MB\n"
        "Large gap → autograd graph retention; similar ramps → allocator cache only."
    )


if __name__ == "__main__":
    main()
