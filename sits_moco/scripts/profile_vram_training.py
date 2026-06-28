#!/usr/bin/env python3
"""
Profile VRAM during the aggregated pixel-chunk training loop.

Goal: verify whether GPU memory climbs because pixel tensors are kept alive
until a dataloader batch ends, or because PyTorch's CUDA caching allocator
holds freed blocks until empty_cache().

DO NOT run while a long training job is active on the same GPU.

Examples (after training finishes):

  # Timeline for one municipality — see ramp within a batch
  python scripts/profile_vram_training.py trace \\
      --yield-csv files/pam_soy_pr_2019_2025.csv \\
      --municipality 4109401 --num-chunks 20 --quiet-training

  # Compare cleanup strategies (current vs per-chunk empty_cache)
  python scripts/profile_vram_training.py compare-cleanup \\
      --yield-csv files/pam_soy_pr_2019_2025.csv \\
      --municipality 4109401 --num-chunks 15

  # Binary-search largest chunk size that fits in a VRAM budget
  python scripts/profile_vram_training.py max-chunk \\
      --yield-csv files/pam_soy_pr_2019_2025.csv \\
      --municipality 4109401 --vram-budget-mb 28000
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast

from env_config import datapath_default_str, resolve_datapath

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
for _p in (_ROOT, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from datasets.pixel_chunk import (  # noqa: E402
    iter_municipality_pixel_chunks,
    prepare_chunk_on_device,
)
from datasets.uscrops_aggregated_npy_polars import USCropsAggregatedNPY  # noqa: E402
from main_yield_regression_polars import (  # noqa: E402
    get_aggregated_dataloader,
    model_kwargs_from_args,
)
from models import STNetRegression  # noqa: E402
from models.weight_init import weight_init_regression  # noqa: E402
from profile_vram_utils import (  # noqa: E402
    CLEANUP_STRATEGIES,
    VramTracer,
    batch_end_cleanup_current,
    cleanup_vram,
    print_summary,
    read_vram,
    summarize_trace,
)
from utils_aggregated import (  # noqa: E402
    CHUNKS_PER_GRAD_UPDATE,
    MunicipalityPixelAccumulator,
)


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for VRAM profiling.")
    device = torch.device("cuda")
    return device


def _build_minimal_args(ns: argparse.Namespace) -> argparse.Namespace:
    """Namespace compatible with utils_aggregated helpers."""
    if ns.datapath is None:
        ns.datapath = str(resolve_datapath())
    else:
        ns.datapath = str(resolve_datapath(ns.datapath))
    defaults = {
        "pixel_chunk_size": ns.chunk_size,
        "prefetch_chunks": ns.prefetch_chunks,
        "quiet_training": ns.quiet_training,
        "disable_pipeline_h2d": ns.disable_pipeline_h2d,
        "sequencelength": ns.sequencelength,
        "feature_layout": ns.feature_layout,
        "target_column": "production_t",
        "batchsize": 1,
        "workers": 0,
        "rc": ns.rc,
        "interp": ns.interp,
        "seed": ns.seed,
        "model_d_model": 128,
        "model_n_head": 16,
        "model_n_layers": 1,
        "model_d_inner": 128,
        "model_dropout": 0.2,
    }
    for k, v in defaults.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)
    return ns


def _load_model_and_dataset(ns: argparse.Namespace):
    device = _require_cuda()
    args = _build_minimal_args(ns)

    _, meta = get_aggregated_dataloader(
        ns.datapath,
        ns.yield_csv,
        batchsize=1,
        workers=0,
        sequencelength=ns.sequencelength,
        rc=ns.rc,
        interp=ns.interp,
        seed=ns.seed,
        mode="train",
        feature_layout=ns.feature_layout,
        target_column="production_t",
    )
    dataset = USCropsAggregatedNPY(
        mode="train",
        root=ns.datapath,
        yield_csv=ns.yield_csv,
        year=None,
        sequencelength=ns.sequencelength,
        dataaug=None,
        randomchoice=ns.rc,
        interp=ns.interp,
        seed=ns.seed,
        preload_ram=False,
        feature_layout=ns.feature_layout,
        target_column="production_t",
    )

    model_kw = model_kwargs_from_args(args)
    model = STNetRegression(
        input_dim=int(meta["ndims"]),
        num_outputs=1,
        max_seq_len=ns.sequencelength,
        **model_kw,
    ).to(device)
    model.apply(weight_init_regression)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler("cuda")
    return device, model, optimizer, scaler, dataset, args


def _pick_municipality(dataset, code: str | None) -> tuple[str, int, int | None]:
    if code is not None:
        for i, (muni, year) in enumerate(dataset.municipality_list):
            if str(muni) == str(code):
                return str(muni), i, year
        raise SystemExit(f"Municipality {code!r} not found in dataset.")

    # Largest municipality by pixel count — stresses VRAM most.
    best = None
    best_pixels = -1
    for muni, year in dataset.municipality_list:
        path = dataset._resolve_npy_path(muni, year)
        if path is None:
            continue
        import numpy as np

        n = len(np.load(path, mmap_mode="r"))
        if n > best_pixels:
            best_pixels = n
            best = (str(muni), year)
    if best is None:
        raise SystemExit("No municipalities with .npy data found.")
    return best[0], 0, best[1]


def _chunk_loop_settings(args) -> tuple[bool, int, int]:
    from utils_aggregated import _pipeline_h2d, _pixel_chunk_size, _prefetch_depth

    device = torch.device("cuda")
    return (
        _pipeline_h2d(args, device),
        _pixel_chunk_size(args),
        _prefetch_depth(args),
    )


def run_chunk_training_steps(
    *,
    model,
    optimizer,
    scaler,
    dataset,
    args,
    municipality_code: str,
    year: int | None,
    num_chunks: int,
    tracer: VramTracer,
    chunk_cleanup,
    batch_cleanup,
    muni_idx: int = 0,
    do_backward: bool = True,
) -> int:
    """
    Run up to ``num_chunks`` forward (+ optional backward) steps.

    Returns number of chunks actually processed.
    """
    device = next(model.parameters()).device
    use_pipeline_h2d, chunk_size, prefetch_depth = _chunk_loop_settings(args)
    aggregation = "sum"
    pixel_acc = MunicipalityPixelAccumulator(aggregation)
    chunks_per_grad = CHUNKS_PER_GRAD_UPDATE
    import numpy as np

    target = torch.tensor([1000.0], device=device)
    npy_path = dataset._resolve_npy_path(municipality_code, year)
    num_pixels = (
        len(np.load(npy_path, mmap_mode="r"))
        if npy_path is not None
        else chunk_size * num_chunks
    )
    total_chunks = (num_pixels + chunk_size - 1) // chunk_size
    total_chunks = min(total_chunks, num_chunks)

    tracer.record("batch_start", muni_idx=muni_idx, sync=True)
    optimizer.zero_grad(set_to_none=True)

    chunk_source = iter_municipality_pixel_chunks(
        dataset,
        municipality_code,
        year=year,
        chunk_size=chunk_size,
        prefetch_depth=prefetch_depth,
        device=device,
        pipeline_h2d=use_pipeline_h2d,
    )

    processed = 0
    for chunk_idx, chunk_item in enumerate(chunk_source):
        if chunk_idx >= num_chunks:
            break

        if use_pipeline_h2d:
            municipality_X_chunk = chunk_item
        else:
            municipality_X_chunk = prepare_chunk_on_device(chunk_item, device)

        tracer.record(
            "after_h2d",
            chunk_idx=chunk_idx,
            muni_idx=muni_idx,
            extra={"pipeline_h2d": use_pipeline_h2d, "chunk_size": chunk_size},
        )

        with autocast("cuda", dtype=torch.bfloat16):
            chunk_predictions = model(municipality_X_chunk)
        chunk_predictions = torch.clamp(chunk_predictions, min=-1e4, max=1e4)

        tracer.record("after_forward", chunk_idx=chunk_idx, muni_idx=muni_idx)

        pixel_acc.add(chunk_predictions)
        processed += 1
        chunk_idx_1 = chunk_idx + 1
        should_update = (chunk_idx_1 % chunks_per_grad == 0) or (
            chunk_idx_1 == total_chunks
        )

        if do_backward and should_update and pixel_acc.valid:
            municipality_agg = pixel_acc.value().squeeze().to(torch.float32)
            if municipality_agg.dim() == 0:
                municipality_agg = municipality_agg.unsqueeze(0)
            fraction = pixel_acc.pixel_count / max(num_pixels, 1)
            expected = (target.squeeze() * fraction).to(torch.float32)
            loss = torch.nn.functional.mse_loss(municipality_agg, expected)
            num_updates = (total_chunks + chunks_per_grad - 1) // chunks_per_grad
            scaled = loss / max(num_updates, 1)
            scaler.scale(scaled).backward()
            pixel_acc.start_new_grad_segment()
            tracer.record("after_backward", chunk_idx=chunk_idx, muni_idx=muni_idx)

        del municipality_X_chunk, chunk_predictions
        chunk_cleanup(device)
        tracer.record("after_chunk_cleanup", chunk_idx=chunk_idx, muni_idx=muni_idx)

    if do_backward and processed > 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    tracer.record("batch_end_before_cleanup", muni_idx=muni_idx, sync=True)
    batch_cleanup(device)
    tracer.record("batch_end_after_cleanup", muni_idx=muni_idx, sync=True)
    return processed


def cmd_trace(ns: argparse.Namespace) -> None:
    device, model, optimizer, scaler, dataset, args = _load_model_and_dataset(ns)
    muni, _, year = _pick_municipality(dataset, ns.municipality)

    out_dir = Path(ns.output_dir)
    tracer = VramTracer(device)
    tracer.reset_peak()

    cleanup_vram(device, empty_cache=True, gc_collect=True, synchronize=True)
    tracer.record("warmup_empty_cache", sync=True)

    processed = run_chunk_training_steps(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        dataset=dataset,
        args=args,
        municipality_code=muni,
        year=year,
        num_chunks=ns.num_chunks,
        tracer=tracer,
        chunk_cleanup=CLEANUP_STRATEGIES["none"],
        batch_cleanup=batch_end_cleanup_current,
    )

    csv_path = out_dir / f"trace_{muni}_chunks{processed}.csv"
    tracer.write_csv(csv_path)
    summary = summarize_trace(tracer.samples)
    print_summary(f"trace muni={muni} chunks={processed}", summary)
    print(f"\nWrote {csv_path}")
    print(
        "\nInterpretation:\n"
        "  - If 'after_chunk_cleanup' reserved keeps climbing but 'active' is flat,\n"
        "    the ramp is PyTorch cache — not live pixel tensors.\n"
        "  - If 'after_forward' active grows linearly with chunk_idx, tensors or\n"
        "    autograd graphs are being retained (should not happen with CHUNKS_PER_GRAD=1)."
    )


def cmd_compare_cleanup(ns: argparse.Namespace) -> None:
    device, model, optimizer, scaler, dataset, args = _load_model_and_dataset(ns)
    muni, _, year = _pick_municipality(dataset, ns.municipality)
    out_dir = Path(ns.output_dir)

    strategies = ns.strategies or list(CLEANUP_STRATEGIES.keys())
    results = {}

    for name in strategies:
        if name not in CLEANUP_STRATEGIES:
            print(f"Skipping unknown strategy {name!r}")
            continue

        # Fresh model/optimizer state per strategy for fair comparison.
        m = copy.deepcopy(model)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        sc = GradScaler("cuda")
        tracer = VramTracer(device)
        tracer.reset_peak()

        cleanup_vram(device, empty_cache=True, gc_collect=True, synchronize=True)

        chunk_fn = CLEANUP_STRATEGIES[name]
        if name == "batch_end_only":
            batch_fn = CLEANUP_STRATEGIES["none"]
        elif name == "batch_end_empty_cache":
            batch_fn = batch_end_cleanup_current
        else:
            batch_fn = CLEANUP_STRATEGIES["none"]
            if name != "none":
                batch_fn = batch_end_cleanup_current

        run_chunk_training_steps(
            model=m,
            optimizer=opt,
            scaler=sc,
            dataset=dataset,
            args=args,
            municipality_code=muni,
            year=year,
            num_chunks=ns.num_chunks,
            tracer=tracer,
            chunk_cleanup=chunk_fn,
            batch_cleanup=batch_fn,
        )

        summary = summarize_trace(tracer.samples)
        results[name] = summary
        tracer.write_csv(out_dir / f"compare_{name}_{muni}.csv")
        print_summary(f"strategy={name}", summary)

    ranked = sorted(
        results.items(),
        key=lambda kv: kv[1].get("reserved_mb_max", 0),
    )
    print("\n=== Ranked by peak reserved VRAM (lower is better) ===")
    for name, s in ranked:
        print(
            f"  {name:28s} peak reserved {s['reserved_mb_max']:.0f} MB, "
            f"ramp {s['reserved_ramp_mb']:.0f} MB"
        )


def cmd_max_chunk(ns: argparse.Namespace) -> None:
    device, model, optimizer, scaler, dataset, args = _load_model_and_dataset(ns)
    muni, _, year = _pick_municipality(dataset, ns.municipality)

    lo, hi = ns.chunk_min, ns.chunk_max
    best = None

    cleanup_vram(device, empty_cache=True, gc_collect=True, synchronize=True)

    while lo <= hi:
        mid = (lo + hi) // 2
        args.pixel_chunk_size = mid
        m = copy.deepcopy(model)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        sc = GradScaler("cuda")
        tracer = VramTracer(device)
        tracer.reset_peak()
        cleanup_vram(device, empty_cache=True, gc_collect=True, synchronize=True)

        try:
            run_chunk_training_steps(
                model=m,
                optimizer=opt,
                scaler=sc,
                dataset=dataset,
                args=args,
                municipality_code=muni,
                year=year,
                num_chunks=ns.num_chunks,
                tracer=tracer,
                chunk_cleanup=CLEANUP_STRATEGIES[ns.chunk_cleanup_strategy],
                batch_cleanup=batch_end_cleanup_current,
            )
        except torch.cuda.OutOfMemoryError:
            cleanup_vram(device, empty_cache=True, gc_collect=True, synchronize=True)
            hi = mid - 1
            print(f"  OOM at chunk_size={mid}")
            continue

        peak = max(s.reserved_mb for s in tracer.samples)
        cleanup_vram(device, empty_cache=True, gc_collect=True, synchronize=True)
        if peak <= ns.vram_budget_mb:
            best = (mid, peak)
            lo = mid + 1
        else:
            hi = mid - 1
            print(f"  over budget at chunk_size={mid}: peak reserved {peak:.0f} MB")

    print(f"\n=== max-chunk muni={muni} budget={ns.vram_budget_mb:.0f} MB ===")
    if best:
        print(f"  Largest chunk_size: {best[0]} (peak reserved {best[1]:.0f} MB)")
    else:
        print("  No chunk size fit within budget — lower budget or use smaller model.")


def cmd_overlap(ns: argparse.Namespace) -> None:
    """Measure VRAM cost of prefetch_depth × pipeline_h2d combinations (forward only)."""
    device, model, _, _, dataset, args = _load_model_and_dataset(ns)
    muni, _, year = _pick_municipality(dataset, ns.municipality)
    model.eval()

    print(f"\n=== pipeline overlap grid (muni={muni}, forward-only) ===")
    rows = []
    for prefetch in ns.prefetch_grid:
        for pipeline in (False, True):
            args.prefetch_chunks = prefetch
            args.disable_pipeline_h2d = not pipeline
            args.quiet_training = pipeline  # pipeline requires quiet mode in production

            cleanup_vram(device, empty_cache=True, gc_collect=True, synchronize=True)
            torch.cuda.reset_peak_memory_stats(device)

            use_pipeline, chunk_size, prefetch_depth = _chunk_loop_settings(args)
            chunk_source = iter_municipality_pixel_chunks(
                dataset,
                muni,
                year=year,
                chunk_size=chunk_size,
                prefetch_depth=prefetch_depth,
                device=device,
                pipeline_h2d=use_pipeline,
            )
            n = 0
            with torch.no_grad():
                for chunk_item in chunk_source:
                    if use_pipeline:
                        x = chunk_item
                    else:
                        x = prepare_chunk_on_device(chunk_item, device)
                    with autocast("cuda", dtype=torch.bfloat16):
                        _ = model(x)
                    del x
                    n += 1
                    if n >= ns.num_chunks:
                        break

            cleanup_vram(device, synchronize=True)
            vram = read_vram(device)
            peak = vram["peak_allocated_mb"]
            rows.append((prefetch, pipeline, peak, vram["reserved_mb"]))
            print(
                f"  prefetch={prefetch} pipeline_h2d={pipeline}: "
                f"peak allocated {peak:.0f} MB, reserved {vram['reserved_mb']:.0f} MB"
            )

    best = min(rows, key=lambda r: r[2])
    print(
        f"\nLowest peak forward-only: prefetch={best[0]} pipeline_h2d={best[1]} "
        f"({best[2]:.0f} MB allocated)"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--datapath",
        type=str,
        default=None,
        help=f"NPY root (default: {datapath_default_str()})",
    )
    common.add_argument("--yield-csv", type=str, required=True)
    common.add_argument("--municipality", type=str, default=None)
    common.add_argument("--sequencelength", type=int, default=45)
    common.add_argument("--feature-layout", type=str, default="spectral")
    common.add_argument("--chunk-size", type=int, default=20000, dest="chunk_size")
    common.add_argument("--prefetch-chunks", type=int, default=2)
    common.add_argument("--quiet-training", action="store_true")
    common.add_argument("--disable-pipeline-h2d", action="store_true")
    common.add_argument("--rc", action="store_true")
    common.add_argument("--interp", action="store_true")
    common.add_argument("--seed", type=int, default=6007)
    common.add_argument(
        "--output-dir",
        type=str,
        default="results/vram_profile",
    )

    t = sub.add_parser("trace", parents=[common], help="VRAM timeline per chunk step")
    t.add_argument("--num-chunks", type=int, default=20)

    c = sub.add_parser(
        "compare-cleanup",
        parents=[common],
        help="Compare chunk vs batch VRAM cleanup strategies",
    )
    c.add_argument("--num-chunks", type=int, default=15)
    c.add_argument("--strategies", nargs="*", default=None)

    m = sub.add_parser(
        "max-chunk",
        parents=[common],
        help="Binary search max chunk_size under VRAM budget",
    )
    m.add_argument("--num-chunks", type=int, default=3)
    m.add_argument("--vram-budget-mb", type=float, default=28000)
    m.add_argument("--chunk-min", type=int, default=2000)
    m.add_argument("--chunk-max", type=int, default=80000)
    m.add_argument(
        "--chunk-cleanup-strategy",
        choices=list(CLEANUP_STRATEGIES.keys()),
        default="sync_after_chunk",
    )

    o = sub.add_parser(
        "overlap",
        parents=[common],
        help="Grid over prefetch × pipeline_h2d (forward-only)",
    )
    o.add_argument("--num-chunks", type=int, default=10)
    o.add_argument("--prefetch-grid", type=int, nargs="+", default=[0, 1, 2])

    return p.parse_args()


def main() -> None:
    ns = parse_args()
    cmds = {
        "trace": cmd_trace,
        "compare-cleanup": cmd_compare_cleanup,
        "max-chunk": cmd_max_chunk,
        "overlap": cmd_overlap,
    }
    cmds[ns.command](ns)


if __name__ == "__main__":
    main()
