"""One training dataloader batch: cross-muni pixel stream + backward."""

from __future__ import annotations

from datetime import datetime

import torch

from datasets.pixel_chunk import prepare_chunk_on_device
from training.pipeline import (
    chunk_debug_syncs,
    effective_chunks_per_grad,
    iter_training_batch_chunks,
    pipeline_h2d,
    scan_skip_municipalities,
)
from training.accumulator import MunicipalityPixelAccumulator
from training.log_util import log_training as _log_training


def _as_1d(t: torch.Tensor) -> torch.Tensor:
    t = t.squeeze()
    if t.dim() == 0:
        return t.unsqueeze(0)
    return t.reshape(-1)


def _stats_on(stat, ref: torch.Tensor):
    if stat is None:
        return None
    if not torch.is_tensor(stat):
        return torch.as_tensor(stat, dtype=ref.dtype, device=ref.device)
    return stat.to(device=ref.device, dtype=ref.dtype)


def _normalize_pred(pred: torch.Tensor, target_mean, target_std) -> torch.Tensor:
    if target_mean is None or target_std is None:
        return pred
    mean = _stats_on(target_mean, pred)
    std = _stats_on(target_std, pred)
    return (pred - mean) / std


def _partial_target(
    target_normalized: torch.Tensor,
    fraction_processed: float,
    aggregation: str | list | tuple,
) -> torch.Tensor:
    """For sum heads, scale target by fraction of pixels seen; mean heads keep full target."""
    if isinstance(aggregation, str):
        if aggregation == "sum":
            return (target_normalized * fraction_processed).to(torch.float32)
        return target_normalized
    out = target_normalized.clone().to(torch.float32)
    aggs = list(aggregation)
    if len(aggs) == 1:
        aggs = aggs * int(out.numel())
    for i, agg in enumerate(aggs):
        if agg == "sum":
            out[i] = out[i] * fraction_processed
    return out


def _fmt_agg(aggregation: str | list | tuple) -> str:
    if isinstance(aggregation, str):
        return ".2f" if aggregation == "mean" else ".0f"
    return ".4g"


def process_train_batch(
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
    aggregation: str | list | tuple,
    target_mean,
    target_std,
    chunk_size: int,
    batch_idx: int,
    run_stats: dict | None = None,
    max_municipalities: int | None = None,
    vram_probe_state: dict | None = None,
) -> tuple[float, bool, int]:
    """
    Process municipalities in one optimizer batch.

    Returns ``(total_loss, batch_has_gradients, municipalities_completed)``.

    When ``vram_probe_state`` is set, checks dedicated VRAM after each chunk and
    stops as soon as the budget is exceeded or ``min_chunks`` have been processed.
    """
    from torch.amp import autocast

    from training_runtime import mark_cudagraph_step, zero_grad

    targets = targets.to(device).float()
    total_loss = 0.0
    batch_has_gradients = False
    chunk_debug = chunk_debug_syncs(args)
    chunks_per_grad = effective_chunks_per_grad(args)
    use_pipeline_h2d = pipeline_h2d(args, device)
    log_fmt = _fmt_agg(aggregation)

    skip_muni_indices = scan_skip_municipalities(
        municipalities, targets, num_pixels_list
    )

    current_muni = -1
    municipality_code = ""
    target = None
    num_pixels = 0
    total_chunks = 0
    chunk_idx = 0
    pixel_acc = None
    year = None
    muni_break = False
    municipalities_completed = 0
    stop_after_municipalities = False
    stop_vram_probe = False

    def _check_vram_probe() -> None:
        nonlocal stop_vram_probe
        if vram_probe_state is None or device.type != "cuda":
            return
        from training.vram_adjuster import _pressure_peak, read_vram_pressure

        vram_probe_state["chunks"] = int(vram_probe_state.get("chunks", 0)) + 1
        pressure = read_vram_pressure(device)
        vram_probe_state["last_pressure"] = pressure
        peak = vram_probe_state.get("peak_pressure")
        if peak is None:
            vram_probe_state["peak_pressure"] = dict(pressure)
        else:
            vram_probe_state["peak_pressure"] = _pressure_peak(peak, pressure)
        if pressure["driver_used_frac"] > float(vram_probe_state["vram_frac"]):
            vram_probe_state["over_limit"] = True
            stop_vram_probe = True
            return
        min_chunks = int(vram_probe_state.get("min_chunks", 1))
        if vram_probe_state["chunks"] >= min_chunks:
            vram_probe_state["complete"] = True
            stop_vram_probe = True

    def _finalize_muni() -> None:
        nonlocal total_loss
        if current_muni < 0:
            return
        if chunk_idx == 0 and num_pixels > 0:
            print(
                f"  ⚠️  Muni {municipality_code}: no pixel chunks loaded "
                f"(year={year}, pixels={num_pixels})",
                flush=True,
            )
        if pixel_acc is not None and pixel_acc.valid and chunk_idx > 0:
            municipality_agg = _as_1d(pixel_acc.value())
            target_normalized = _as_1d(target).to(torch.float32)
            municipality_agg_normalized = _normalize_pred(
                municipality_agg.to(torch.float32), target_mean, target_std
            )
            final_loss = torch.nn.functional.mse_loss(
                municipality_agg_normalized, target_normalized
            )

            if torch.isnan(final_loss) or torch.isinf(final_loss):
                print(
                    f"  ❌ DEBUG: Municipality {municipality_code} final loss is invalid: {final_loss.item()}"
                )
            else:
                if target_mean is not None and target_std is not None:
                    mean_t = _stats_on(target_mean, target_normalized)
                    std_t = _stats_on(target_std, target_normalized)
                    target_denorm = target_normalized * std_t + mean_t
                    pred_denorm = municipality_agg_normalized * std_t + mean_t
                else:
                    target_denorm = target_normalized
                    pred_denorm = municipality_agg_normalized
                total_loss += final_loss.item()
                if vram_probe_state is None:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    if target_denorm.numel() == 1:
                        tgt_s = f"{target_denorm.item():{log_fmt}}"
                        pred_s = f"{pred_denorm.item():{log_fmt}}"
                    else:
                        tgt_s = (
                            "["
                            + ", ".join(f"{v:{log_fmt}}" for v in target_denorm.tolist())
                            + "]"
                        )
                        pred_s = (
                            "["
                            + ", ".join(f"{v:{log_fmt}}" for v in pred_denorm.tolist())
                            + "]"
                        )
                    _log_training(
                        f"[{timestamp}] Muni {municipality_code}: "
                        f"target={tgt_s}, pred={pred_s}, "
                        f"chunks={chunk_idx}/{total_chunks}, loss={final_loss.item():.2e}"
                    )

    def _begin_muni(muni_idx: int) -> None:
        nonlocal current_muni, municipality_code, target, num_pixels, total_chunks
        nonlocal chunk_idx, pixel_acc, year, muni_break
        current_muni = muni_idx
        municipality_code = municipalities[muni_idx]
        num_pixels = num_pixels_list[muni_idx]
        target = targets[muni_idx]
        pixel_acc = MunicipalityPixelAccumulator(aggregation)
        chunk_idx = 0
        total_chunks = (num_pixels + chunk_size - 1) // chunk_size
        year = years[muni_idx] if years[muni_idx] is not None else None
        muni_break = False

    chunk_stream = iter_training_batch_chunks(
        dataset,
        municipalities,
        years,
        num_pixels_list,
        chunk_size,
        args,
        device,
        skip_muni_indices=skip_muni_indices,
    )

    for stream_muni_idx, chunk_item in chunk_stream:
        if stop_after_municipalities or stop_vram_probe:
            break

        if stream_muni_idx != current_muni:
            if current_muni >= 0:
                _finalize_muni()
            _begin_muni(stream_muni_idx)

        if muni_break:
            continue

        if use_pipeline_h2d:
            municipality_X_chunk = chunk_item
        else:
            unpacked = chunk_item
            chunk_x, chunk_mask, chunk_doy, chunk_weight = unpacked

            if chunk_debug:
                if torch.isnan(chunk_x).any() or torch.isinf(chunk_x).any():
                    print(
                        f"  ❌ DEBUG: Municipality {municipality_code} chunk {chunk_idx} has invalid input data (x)"
                    )
                    continue
                if torch.isnan(chunk_mask).any() or torch.isinf(chunk_mask).any():
                    print(
                        f"  ❌ DEBUG: Municipality {municipality_code} chunk {chunk_idx} has invalid mask"
                    )
                    continue
                if torch.isnan(chunk_doy).any() or torch.isinf(chunk_doy).any():
                    print(
                        f"  ❌ DEBUG: Municipality {municipality_code} chunk {chunk_idx} has invalid DOY"
                    )
                    continue
                if torch.isnan(chunk_weight).any() or torch.isinf(chunk_weight).any():
                    print(
                        f"  ❌ DEBUG: Municipality {municipality_code} chunk {chunk_idx} has invalid weight"
                    )
                    continue

            municipality_X_chunk = prepare_chunk_on_device(unpacked, device)

        mark_cudagraph_step()
        with autocast("cuda", dtype=torch.bfloat16):
            chunk_predictions = model(municipality_X_chunk)
        if run_stats is not None:
            run_stats["forward_calls"] = run_stats.get("forward_calls", 0) + 1
            run_stats["chunks_processed"] = run_stats.get("chunks_processed", 0) + 1
            run_stats["pixels_processed"] = run_stats.get(
                "pixels_processed", 0
            ) + int(chunk_predictions.shape[0])

        chunk_predictions = torch.clamp(chunk_predictions, min=-1e4, max=1e4)
        chunk_predictions = torch.nan_to_num(
            chunk_predictions, nan=0.0, posinf=1e4, neginf=-1e4
        )

        if chunk_debug and (
            torch.isnan(chunk_predictions).any() or torch.isinf(chunk_predictions).any()
        ):
            print(
                f"  ❌ DEBUG: Municipality {municipality_code} chunk {chunk_idx} has invalid predictions"
            )
            continue

        pixel_acc.add(chunk_predictions)
        if chunk_debug and pixel_acc.is_extreme():
            print(
                f"  ⚠️  DEBUG: Municipality {municipality_code} has extreme "
                f"{aggregation} at chunk {chunk_idx}"
            )
            pixel_acc = MunicipalityPixelAccumulator(aggregation)
            muni_break = True
            continue

        chunk_idx += 1

        chunks_in_group = chunk_idx % chunks_per_grad
        is_last_chunk = chunk_idx == total_chunks
        should_update = (chunks_in_group == 0) or is_last_chunk

        if should_update and pixel_acc.valid:
            if chunk_debug and pixel_acc.is_extreme():
                print(
                    f"  ⚠️  DEBUG: Municipality {municipality_code} extreme "
                    f"{aggregation} at chunk {chunk_idx}/{total_chunks}, skipping backward"
                )
                pixel_acc = MunicipalityPixelAccumulator(aggregation)
                muni_break = True
                continue

            municipality_agg = _as_1d(pixel_acc.value())
            target_normalized = _as_1d(target).to(torch.float32)
            municipality_agg_normalized = _normalize_pred(
                municipality_agg.to(torch.float32), target_mean, target_std
            )

            fraction_processed = (
                pixel_acc.pixel_count / num_pixels if num_pixels > 0 else 1.0
            )
            expected_partial_target = _partial_target(
                target_normalized, fraction_processed, aggregation
            )

            muni_loss = torch.nn.functional.mse_loss(
                municipality_agg_normalized, expected_partial_target
            )

            if torch.isnan(muni_loss) or torch.isinf(muni_loss):
                print(
                    f"  ❌ DEBUG: Municipality {municipality_code} has invalid loss: {muni_loss.item()}"
                )
                print(
                    f"      Expected partial target: {expected_partial_target.detach().cpu().tolist()}, "
                    f"Pred: {municipality_agg_normalized.detach().cpu().tolist()}"
                )
                print(
                    f"      Chunks: {chunk_idx}/{total_chunks}, Fraction processed: {fraction_processed:.4f}"
                )
                muni_break = True
                continue

            num_updates = (total_chunks + chunks_per_grad - 1) // chunks_per_grad
            update_weight = 1.0 / num_updates if num_updates > 0 else 1.0
            scaled_loss = muni_loss * update_weight

            if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
                print(
                    f"  ❌ DEBUG: Skipping backward for municipality {municipality_code} - scaled_loss is invalid: {scaled_loss.item()}"
                )
                muni_break = True
                continue

            scaler.scale(scaled_loss).backward()
            if run_stats is not None:
                run_stats["backward_calls"] = run_stats.get("backward_calls", 0) + 1
            batch_has_gradients = True

            pixel_acc.start_new_grad_segment()
            del municipality_X_chunk, chunk_predictions
            if is_last_chunk:
                muni_break = True
                municipalities_completed += 1
                if (
                    max_municipalities is not None
                    and municipalities_completed >= max_municipalities
                ):
                    stop_after_municipalities = True
            _check_vram_probe()
            continue

        del municipality_X_chunk, chunk_predictions
        _check_vram_probe()

    if current_muni >= 0 and not stop_vram_probe:
        _finalize_muni()

    if batch_has_gradients and vram_probe_state is None:
        has_nan_grad = any(
            p.grad is not None and torch.isnan(p.grad).any() for p in model.parameters()
        )
        if has_nan_grad:
            zero_grad(optimizer, args)
        else:
            max_grad_norm = 5.0
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            zero_grad(optimizer, args)
            if run_stats is not None:
                run_stats["optimizer_steps"] = run_stats.get("optimizer_steps", 0) + 1
    elif not batch_has_gradients:
        zero_grad(optimizer, args)
        if vram_probe_state is None:
            print(
                f"  ⚠️  DEBUG: Skipping optimizer step for batch {batch_idx} - no valid gradients accumulated"
            )

    return total_loss, batch_has_gradients, municipalities_completed
