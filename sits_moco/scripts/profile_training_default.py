#!/usr/bin/env python3
"""
Profile the default tuning/production training path (no variant comparisons).

Uses the real ``train_epoch_aggregated`` from utils_aggregated.py with the same
model / dataloader / optimizer setup as main_yield_regression_polars.py.

Subcommands
-----------
  segments  — coarse timers (dataloader, CPU .npy, H2D, forward, backward, …)
  torch     — PyTorch Profiler (CPU + CUDA); exports Chrome trace for TensorBoard
  both      — run segments then torch profiler

Examples
--------
  python scripts/profile_training_default.py segments --batches 2
  python scripts/profile_training_default.py torch --batches 1 --warmup-batches 1
  python scripts/profile_training_default.py both --batchsize 32

View Chrome trace (after ``torch`` or ``both``):
  tensorboard --logdir results/training_profile

DO NOT run while another training job uses the same GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[1]
# Running as `python scripts/profile_training_default.py` puts `scripts/` on sys.path,
# which would shadow stdlib `profile` if we had a package named profile there.
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.profiling.limited_dataloader import BatchLimitedDataLoader
from scripts.profiling.production_stack import build_production_training_stack, describe_stack
from scripts.profiling.segment_probes import ProfileReport, SegmentProbeSession
from utils_aggregated import train_epoch_aggregated


def _profiler_device_time_us(evt) -> float:
    """CUDA/device time in microseconds (API differs across torch versions)."""
    for attr in (
        "device_time_total",
        "cuda_time_total",
        "self_device_time_total",
        "self_cuda_time_total",
        "device_time",
        "cuda_time",
    ):
        val = getattr(evt, attr, None)
        if val:
            return float(val)
    return 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Study YAML (default: productivity_baseline.yaml)",
    )
    common.add_argument("--batchsize", type=int, default=32)
    common.add_argument("--batches", type=int, default=2, help="Train batches to profile")
    common.add_argument("--workers", type=int, default=None)
    common.add_argument("--seed", type=int, default=None)
    common.add_argument("--datapath", type=str, default=None)
    common.add_argument("--yield-csv", type=str, default=None)
    common.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "results" / "training_profile",
    )
    common.add_argument(
        "--warmup-batches",
        type=int,
        default=1,
        help="Warmup batches before torch profiler timed region (torch/both only)",
    )

    sub.add_parser(
        "segments",
        parents=[common],
        help="Segment timers on production train_epoch_aggregated",
    )
    sub.add_parser(
        "torch",
        parents=[common],
        help="PyTorch Profiler (CPU+CUDA) on production path",
    )
    sub.add_parser(
        "both",
        parents=[common],
        help="segments then torch profiler",
    )
    return p.parse_args()


def _limited_loader(stack: dict, num_batches: int):
    return BatchLimitedDataLoader(stack["train_dataloader"], num_batches)


def run_segments(ns: argparse.Namespace, stack: dict, out_dir: Path) -> ProfileReport:
    report = ProfileReport()
    loader = _limited_loader(stack, ns.batches)
    dataset = loader.dataset

    t0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    t1 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    import time

    wall_t0 = time.perf_counter()
    session = SegmentProbeSession(report)
    with session.install(dataset=dataset, model=stack["model"], optimizer=stack["optimizer"]):
        if t0 is not None:
            t0.record()
        train_epoch_aggregated(
            stack["model"],
            stack["optimizer"],
            stack["criterion"],
            loader,
            stack["device"],
            stack["args"],
            target_mean=stack["target_mean"],
            target_std=stack["target_std"],
        )
        if t1 is not None:
            t1.record()
            torch.cuda.synchronize()
    report.wall_seconds = time.perf_counter() - wall_t0

    report.print_summary()
    rows = report.to_rows()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"segments_{ts}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")
    return report


def run_torch_profiler(ns: argparse.Namespace, stack: dict, out_dir: Path) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("torch profiler CUDA mode requires a GPU.")

    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = out_dir / "torch_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    # Warmup on full loader (not limited) then profile limited batches
    if ns.warmup_batches > 0:
        warm_loader = _limited_loader(stack, ns.warmup_batches)
        print(f"Warmup: {ns.warmup_batches} batch(es) …")
        train_epoch_aggregated(
            stack["model"],
            stack["optimizer"],
            stack["criterion"],
            warm_loader,
            stack["device"],
            stack["args"],
            target_mean=stack["target_mean"],
            target_std=stack["target_std"],
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    loader = _limited_loader(stack, ns.batches)
    print(f"Torch profiler: {ns.batches} batch(es) …")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        train_epoch_aggregated(
            stack["model"],
            stack["optimizer"],
            stack["criterion"],
            loader,
            stack["device"],
            stack["args"],
            target_mean=stack["target_mean"],
            target_std=stack["target_std"],
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    print("\n=== PyTorch Profiler: top CUDA kernels ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    print("\n=== PyTorch Profiler: top CPU ops ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    trace_path = trace_dir / f"trace_{ts}"
    prof.export_chrome_trace(str(trace_path) + ".json")
    print(f"\nChrome trace: {trace_path}.json")
    print(f"  tensorboard --logdir {out_dir.resolve()}")

    # Summarize memcpy vs compute heuristically from kernel names
    cuda_rows = []
    for evt in prof.key_averages():
        name = evt.key
        cuda_us = _profiler_device_time_us(evt)
        if cuda_us <= 0:
            continue
        category = "other"
        lname = name.lower()
        if "memcpy" in lname or "copy" in lname:
            category = "memcpy_h2d_d2h"
        elif "gemm" in lname or "conv" in lname or "cutlass" in lname or "volta" in lname:
            category = "compute"
        elif "elementwise" in lname or "reduce" in lname or "softmax" in lname:
            category = "compute"
        elif "nccl" in lname:
            category = "comm"
        cuda_rows.append({"kernel": name, "cuda_us": cuda_us, "category": category})

    if cuda_rows:
        df = pd.DataFrame(cuda_rows)
        by_cat = df.groupby("category")["cuda_us"].sum()
        total = by_cat.sum()
        print("\n=== CUDA time by heuristic category ===")
        for cat, us in by_cat.sort_values(ascending=False).items():
            print(f"  {cat:20s} {us/1000:.1f} ms  ({100*us/total:.1f}%)")

        summary_path = out_dir / f"torch_summary_{ts}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "trace": str(trace_path) + ".json",
                    "cuda_by_category_ms": {k: v / 1000 for k, v in by_cat.items()},
                    "top_kernels": df.sort_values("cuda_us", ascending=False)
                    .head(30)
                    .to_dict(orient="records"),
                },
                f,
                indent=2,
            )
        print(f"Wrote {summary_path}")


def main() -> None:
    ns = parse_args()
    out_dir = Path(ns.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stack = build_production_training_stack(
        config_yaml=ns.config,
        batchsize=ns.batchsize,
        workers=ns.workers,
        seed=ns.seed,
        datapath=ns.datapath,
        yield_csv=ns.yield_csv,
    )

    print("Production training profile")
    print(describe_stack(stack))
    print(f"  profile batches : {ns.batches}")
    print(f"  output          : {out_dir.resolve()}\n")

    if ns.command in ("segments", "both"):
        run_segments(ns, stack, out_dir)
        if ns.command == "both":
            print("\n" + "=" * 72 + "\n")
            # Rebuild model/optimizer after segment run (clean autograd state)
            stack = build_production_training_stack(
                config_yaml=ns.config,
                batchsize=ns.batchsize,
                workers=ns.workers,
                seed=ns.seed,
                datapath=ns.datapath,
                yield_csv=ns.yield_csv,
            )

    if ns.command in ("torch", "both"):
        run_torch_profiler(ns, stack, out_dir)


if __name__ == "__main__":
    main()
