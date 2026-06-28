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
    aggregation: str,
    target_mean,
    target_std,
    chunk_size: int,
    batch_idx: int,
    run_stats: dict | None = None,
) -> tuple[float, bool]:
    """
    Process municipalities in one optimizer batch.

    Returns ``(total_loss, batch_has_gradients)``.
    """
    from torch.amp import autocast

    from training_runtime import mark_cudagraph_step, zero_grad

    targets = targets.to(device).float()
    total_loss = 0.0
    batch_has_gradients = False
    chunk_debug = chunk_debug_syncs(args)
    chunks_per_grad = effective_chunks_per_grad(args)
    use_pipeline_h2d = pipeline_h2d(args, device)

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
            municipality_agg = pixel_acc.value().squeeze()
            if municipality_agg.dim() == 0:
                municipality_agg = municipality_agg.unsqueeze(0)
            elif municipality_agg.dim() > 1:
                municipality_agg = municipality_agg.flatten()[0:1]

            target_normalized = target.squeeze().to(torch.float32)
            if target_normalized.dim() == 0:
                target_normalized = target_normalized.unsqueeze(0)

            municipality_agg_normalized = municipality_agg.to(torch.float32)
            if target_mean is not None and target_std is not None:
                municipality_agg_normalized = (
                    municipality_agg_normalized - target_mean
                ) / target_std

            final_loss = torch.nn.functional.mse_loss(
                municipality_agg_normalized, target_normalized
            )

            if torch.isnan(final_loss) or torch.isinf(final_loss):
                print(
                    f"  ❌ DEBUG: Municipality {municipality_code} final loss is invalid: {final_loss.item()}"
                )
            else:
                if target_mean is not None and target_std is not None:
                    target_denorm = target_normalized.item() * target_std + target_mean
                    pred_denorm = (
                        municipality_agg_normalized.item() * target_std + target_mean
                    )
                else:
                    target_denorm = target_normalized.item()
                    pred_denorm = municipality_agg_normalized.item()
                total_loss += final_loss.item()
                fmt = ".2f" if aggregation == "mean" else ".0f"
                timestamp = datetime.now().strftime("%H:%M:%S")
                _log_training(
                    f"[{timestamp}] Muni {municipality_code}: "
                    f"target={target_denorm:{fmt}}, pred={pred_denorm:{fmt}}, "
                    f"chunks={chunk_idx}/{total_chunks}, loss={final_loss.item():.2e}"
                )

    def _begin_muni(muni_idx: int) -> None:
        nonlocal current_muni, municipality_code, target, num_pixels, total_chunks
        nonlocal chunk_idx, pixel_acc, year, muni_break
        current_muni = muni_idx
        municipality_code = municipalities[muni_idx]
        num_pixels = num_pixels_list[muni_idx]
        target = targets[muni_idx : muni_idx + 1].to(device).float()
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

            municipality_agg = pixel_acc.value().squeeze()
            if municipality_agg.dim() == 0:
                municipality_agg = municipality_agg.unsqueeze(0)
            elif municipality_agg.dim() > 1:
                municipality_agg = municipality_agg.flatten()[0:1]

            target_normalized = target.squeeze().to(torch.float32)
            if target_normalized.dim() == 0:
                target_normalized = target_normalized.unsqueeze(0)

            municipality_agg_normalized = municipality_agg.to(torch.float32)
            if target_mean is not None and target_std is not None:
                municipality_agg_normalized = (
                    municipality_agg_normalized - target_mean
                ) / target_std

            fraction_processed = (
                pixel_acc.pixel_count / num_pixels if num_pixels > 0 else 1.0
            )
            if aggregation == "sum":
                expected_partial_target = (
                    target_normalized * fraction_processed
                ).to(torch.float32)
            else:
                expected_partial_target = target_normalized

            muni_loss = torch.nn.functional.mse_loss(
                municipality_agg_normalized, expected_partial_target
            )

            if torch.isnan(muni_loss) or torch.isinf(muni_loss):
                print(
                    f"  ❌ DEBUG: Municipality {municipality_code} has invalid loss: {muni_loss.item()}"
                )
                print(
                    f"      Expected partial target: {expected_partial_target.item():.4f}, "
                    f"Pred: {municipality_agg_normalized.item():.4f}"
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
            continue

        del municipality_X_chunk, chunk_predictions

    if current_muni >= 0:
        _finalize_muni()

    if batch_has_gradients:
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
    else:
        zero_grad(optimizer, args)
        print(
            f"  ⚠️  DEBUG: Skipping optimizer step for batch {batch_idx} - no valid gradients accumulated"
        )

    return total_loss, batch_has_gradients
