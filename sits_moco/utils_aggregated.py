"""
Utilities for aggregated regression: pixel-level predictions → municipality-level targets.
"""

from __future__ import annotations

import gc
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from datasets.pixel_chunk import (
    iter_municipality_pixel_chunks,
    prepare_chunk_on_device,
    unpack_pixel_chunk,
)

# Pixels per forward pass
MAX_PIXEL_BATCH_SIZE = 45000

# Max consecutive chunk forwards before backward; lower = less VRAM, more backward calls
CHUNKS_PER_GRAD_UPDATE = 1


def _pixel_chunk_size(args) -> int:
    return getattr(args, "pixel_chunk_size", None) or MAX_PIXEL_BATCH_SIZE


def _prefetch_depth(args) -> int:
    """Background CPU threads preparing upcoming pixel chunks (0 = disabled)."""
    return max(0, int(getattr(args, "prefetch_chunks", 2)))


def _chunk_debug_syncs(args) -> bool:
    """Per-chunk GPU NaN/Inf checks (--quiet-training disables these, not muni logs)."""
    return not getattr(args, "quiet_training", False)


def _pipeline_h2d(args, device: torch.device) -> bool:
    """Overlap H2D copies with GPU compute (same numerics as sync transfer)."""
    if device.type != "cuda" or getattr(args, "disable_pipeline_h2d", False):
        return False
    # Needs quiet mode so we skip per-chunk CPU tensor debug checks.
    return _chunk_debug_syncs(args) is False


def _batch_vram_cleanup(device: torch.device) -> None:
    """gc + empty_cache after each dataloader batch."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


TARGET_SPECS = {
    "total": {"column": "production_t", "aggregation": "sum", "unit": "tons"},
    "total_adj": {
        "column": "production_t_s2_adj",
        "aggregation": "sum",
        "unit": "tons",
    },
    "productivity": {"column": "yield_t_ha", "aggregation": "mean", "unit": "t/ha"},
}


def resolve_target(target: str) -> tuple[str, str, str]:
    """Return (target_column, aggregation, unit) for a registered target name."""
    spec = TARGET_SPECS.get(target)
    if spec is None:
        choices = ", ".join(TARGET_SPECS)
        raise ValueError(f"Unknown target {target!r}; expected one of: {choices}")
    return spec["column"], spec["aggregation"], spec["unit"]


def resolve_inference_target(
    run_config: dict,
    checkpoint: dict | None = None,
) -> tuple[str, str, str, str]:
    """
    Resolve (target, target_column, aggregation, unit) for inference.

    Reads training/config.json first, then checkpoint metadata saved at train time.
    """
    computed = run_config.get("computed") or {}
    cli = run_config.get("cli") or {}
    ck = checkpoint or {}

    target = computed.get("target") or cli.get("target") or ck.get("target")
    if target is not None:
        col, agg, unit = resolve_target(target)
        return target, col, agg, unit

    target_column = computed.get("yield_target_column") or ck.get("target_column")
    aggregation = computed.get("municipality_aggregation") or ck.get("aggregation")
    if target_column is not None:
        if aggregation is None:
            aggregation = aggregation_for_target_column(target_column)
        for name, spec in TARGET_SPECS.items():
            if spec["column"] == target_column:
                return name, target_column, aggregation, spec["unit"]

    if aggregation == "mean":
        col, agg, unit = resolve_target("productivity")
        return "productivity", col, agg, unit
    if aggregation == "sum":
        col, agg, unit = resolve_target("total")
        return "total", col, agg, unit

    config_path = run_config.get("config_path", "training/config.json")
    raise ValueError(
        f"Cannot determine training target from {config_path} or checkpoint. "
        "Re-train with --target total|total_adj|productivity."
    )


def aggregation_for_target_column(target_column: str) -> str:
    for spec in TARGET_SPECS.values():
        if spec["column"] == target_column:
            return spec["aggregation"]
    raise ValueError(
        f"Unknown target column {target_column!r}; "
        f"expected one of: {', '.join(s['column'] for s in TARGET_SPECS.values())}"
    )


def aggregate_pixels(predictions: torch.Tensor, aggregation: str) -> torch.Tensor:
    if aggregation == "sum":
        return predictions.sum(dim=0)
    if aggregation == "mean":
        return predictions.mean(dim=0)
    raise ValueError(f"aggregation must be 'sum' or 'mean', got {aggregation!r}")


class MunicipalityPixelAccumulator:
    """Accumulate pixel predictions into a municipality-level sum or mean."""

    def __init__(self, aggregation: str = "sum"):
        if aggregation not in ("sum", "mean"):
            raise ValueError(
                f"aggregation must be 'sum' or 'mean', got {aggregation!r}"
            )
        self.aggregation = aggregation
        self._sum: torch.Tensor | None = None
        self.pixel_count = 0

    def add(self, chunk_predictions: torch.Tensor) -> None:
        chunk_sum = chunk_predictions.sum(dim=0).to(torch.float32)
        if self._sum is None:
            self._sum = chunk_sum
        else:
            self._sum = self._sum + chunk_sum
        self.pixel_count += int(chunk_predictions.shape[0])

    @property
    def valid(self) -> bool:
        return self._sum is not None and self.pixel_count > 0

    def value(self) -> torch.Tensor | None:
        if not self.valid:
            return None
        if self.aggregation == "sum":
            return self._sum
        return self._sum / float(self.pixel_count)

    def detach(self) -> None:
        if self._sum is not None:
            self._sum = self._sum.detach()

    def start_new_grad_segment(self) -> None:
        """Break the autograd chain after backward while keeping running totals."""
        self.detach()

    def is_extreme(self) -> bool:
        val = self.value()
        if val is None or torch.isinf(val).any():
            return True
        limit = 1e7 if self.aggregation == "sum" else 1e3
        return torch.abs(val).max().item() > limit


def stnet_regression_input_dim_from_state_dict(state_dict: dict) -> int:
    """Infer STNetRegression MLP input width (10 spectral vs 12 with Xavier) from saved weights."""
    w = state_dict.get("mlp1.0.lin.weight")
    if w is not None:
        return int(w.shape[1])
    return 10


def aggregate_municipality_from_pixel_chunks(
    model, pixel_chunks, device, aggregation: str = "sum"
) -> float | None:
    """Run STNet on pixel chunks; aggregate to municipality prediction (sum or mean)."""
    from torch.amp import autocast

    from utils import recursive_todevice

    acc = MunicipalityPixelAccumulator(aggregation)
    model.eval()
    with torch.no_grad():
        for pixel_chunk in pixel_chunks:
            unpacked = unpack_pixel_chunk(pixel_chunk)
            if unpacked is None:
                continue
            municipality_X_chunk = prepare_chunk_on_device(unpacked, device)
            with (
                autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else torch.no_grad()
            ):
                chunk_predictions = model(municipality_X_chunk)
            acc.add(chunk_predictions)

    result = acc.value()
    if result is None:
        return None
    result = result.squeeze()
    if result.dim() > 0:
        result = result[0] if len(result) > 0 else torch.tensor(0.0)
    return float(result.item())


def sum_municipality_from_pixel_chunks(model, pixel_chunks, device) -> float | None:
    """Backward-compatible alias: sum aggregation (total production in tons)."""
    return aggregate_municipality_from_pixel_chunks(
        model, pixel_chunks, device, aggregation="sum"
    )


def regression_metrics(y_pred, y_true, *, target_column: str = "production_t"):
    """Calculate regression metrics: RMSE, MAE, R², MAPE."""
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()

    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    y_pred = y_pred[mask]
    y_true = y_true[mask]

    if len(y_pred) == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "mape": np.nan}

    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_true))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))

    # MAPE: Filter out values where y_true is too small (less than 1% of mean or absolute threshold)
    # This prevents division by near-zero values that cause MAPE to explode
    mean_y_true = np.mean(np.abs(y_true))
    floor = 0.1 if target_column == "yield_t_ha" else 100.0
    threshold = max(mean_y_true * 0.01, floor)
    mape_mask = np.abs(y_true) >= threshold
    if mape_mask.sum() > 0:
        mape = (
            np.mean(np.abs((y_true[mape_mask] - y_pred[mape_mask]) / y_true[mape_mask]))
            * 100
        )
    else:
        # If all values are too small, MAPE is not meaningful
        mape = np.nan

    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


class AggregatedMSELoss(nn.Module):
    """Loss: aggregate pixel predictions per municipality (sum or mean), compare to target."""

    def __init__(
        self,
        reduction="mean",
        target_mean=None,
        target_std=None,
        aggregation: str = "sum",
        target_column: str = "production_t",
    ):
        super().__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction="none")
        self.target_mean = target_mean if target_mean is not None else 0.0
        self.target_std = target_std if target_std is not None else 1.0
        self.normalize = target_mean is not None and target_std is not None
        self.aggregation = aggregation
        self.target_column = target_column

    def forward(self, predictions_list, targets, num_pixels_list=None):
        """
        Args:
            predictions_list: List of tensors, one per municipality [num_pixels, num_outputs]
            targets: [batch_size] - Municipality-level targets (already normalized if normalize=True)
        """
        aggregated_predictions = []
        for pred in predictions_list:
            aggregated_predictions.append(aggregate_pixels(pred, self.aggregation))

        aggregated = torch.stack(aggregated_predictions)  # [batch_size, num_outputs]

        if aggregated.dim() > 1 and aggregated.size(1) == 1:
            aggregated = aggregated.squeeze(1)
        if targets.dim() > 1 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        # Normalize predictions if normalization is enabled (targets are already normalized)
        if self.normalize:
            aggregated = (aggregated - self.target_mean) / self.target_std

        # Compute loss directly on normalized values
        loss = self.mse(aggregated, targets)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def aggregated_collate_fn(batch):
    """Custom collate function for variable number of pixels per municipality."""
    municipalities = []
    targets = []
    num_pixels_list = []
    years = []

    for item in batch:
        if len(item) == 4:
            # Multi-year mode: (municipality_code, target, num_pixels, year)
            municipality_code, target, num_pixels, year = item
            years.append(year)
        else:
            # Single-year mode: (municipality_code, target, num_pixels)
            municipality_code, target, num_pixels = item
            years.append(None)
        municipalities.append(municipality_code)
        targets.append(target)
        num_pixels_list.append(num_pixels)

    targets = torch.stack(targets)
    return municipalities, targets, num_pixels_list, years


def train_epoch_aggregated(
    model,
    optimizer,
    criterion,
    dataloader,
    device,
    args,
    target_mean=None,
    target_std=None,
):
    """Training epoch: process municipalities, sum pixel predictions, compare to targets."""
    from torch.amp import GradScaler, autocast
    from tqdm import tqdm

    from utils import AverageMeter

    losses = AverageMeter("Loss", ":.4e")
    model.train()
    aggregation = getattr(criterion, "aggregation", "sum")
    scaler = GradScaler("cuda")
    chunk_debug_syncs = _chunk_debug_syncs(args)
    chunk_size = _pixel_chunk_size(args)
    chunks_per_grad = CHUNKS_PER_GRAD_UPDATE

    pbar = tqdm(
        total=len(dataloader),
        desc="Train",
        unit="batch",
        leave=True,
        dynamic_ncols=True,
    )

    for idx, (municipalities, targets, num_pixels_list, years) in enumerate(dataloader):
        targets = targets.to(device).float()
        total_loss = 0.0
        num_municipalities = len(municipalities)
        batch_has_gradients = (
            False  # Track if any gradients were accumulated in this batch
        )

        optimizer.zero_grad()
        batch_total = len(dataloader)
        pbar.write(
            f"── batch {idx + 1}/{batch_total} ({num_municipalities} muni) ──"
        )

        for muni_idx, municipality_code in enumerate(municipalities):
            num_pixels = num_pixels_list[muni_idx]
            target = targets[muni_idx : muni_idx + 1].to(device).float()

            # Debug: Check target values
            if torch.isnan(target).any() or torch.isinf(target).any():
                print(
                    f"  ❌ DEBUG: Municipality {municipality_code} has invalid target: {target}"
                )
                continue
            # Note: Negative targets are normal after normalization, so we don't warn about them

            dataset = dataloader.dataset
            pixel_acc = MunicipalityPixelAccumulator(aggregation)
            chunk_idx = 0
            total_chunks = (num_pixels + chunk_size - 1) // chunk_size

            # Skip municipalities with zero pixels
            if total_chunks == 0 or num_pixels == 0:
                print(
                    f"  ⚠️  Warning: Municipality {municipality_code} has 0 pixels, skipping"
                )
                continue

            # Get year for this municipality (if multi-year mode)
            year = years[muni_idx] if years[muni_idx] is not None else None
            use_pipeline_h2d = _pipeline_h2d(args, device)
            chunk_source = iter_municipality_pixel_chunks(
                dataset,
                municipality_code,
                year=year,
                chunk_size=chunk_size,
                prefetch_depth=_prefetch_depth(args),
                device=device,
                pipeline_h2d=use_pipeline_h2d,
            )
            for chunk_item in chunk_source:
                if use_pipeline_h2d:
                    municipality_X_chunk = chunk_item
                else:
                    unpacked = chunk_item
                    chunk_x, chunk_mask, chunk_doy, chunk_weight = unpacked

                    if chunk_debug_syncs:
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

                # Use bfloat16 for faster computation with Tensor Cores
                with autocast("cuda", dtype=torch.bfloat16):
                    chunk_predictions = model(municipality_X_chunk)

                # Clip predictions to prevent extreme values that could cause overflow
                chunk_predictions = torch.clamp(chunk_predictions, min=-1e4, max=1e4)
                chunk_predictions = torch.nan_to_num(
                    chunk_predictions, nan=0.0, posinf=1e4, neginf=-1e4
                )

                if chunk_debug_syncs and (
                    torch.isnan(chunk_predictions).any()
                    or torch.isinf(chunk_predictions).any()
                ):
                    print(
                        f"  ❌ DEBUG: Municipality {municipality_code} chunk {chunk_idx} has invalid predictions"
                    )
                    continue

                pixel_acc.add(chunk_predictions)
                if chunk_debug_syncs and pixel_acc.is_extreme():
                    print(
                        f"  ⚠️  DEBUG: Municipality {municipality_code} has extreme "
                        f"{aggregation} at chunk {chunk_idx}"
                    )
                    pixel_acc = MunicipalityPixelAccumulator(aggregation)
                    break

                chunk_idx += 1

                # Do gradient updates periodically to cap autograd graph depth
                chunks_in_group = chunk_idx % chunks_per_grad
                is_last_chunk = chunk_idx == total_chunks
                should_update = (chunks_in_group == 0) or is_last_chunk

                if should_update and pixel_acc.valid:
                    if chunk_debug_syncs and pixel_acc.is_extreme():
                        print(
                            f"  ⚠️  DEBUG: Municipality {municipality_code} extreme "
                            f"{aggregation} at chunk {chunk_idx}/{total_chunks}, skipping backward"
                        )
                        pixel_acc = MunicipalityPixelAccumulator(aggregation)
                        break

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

                    # Compute loss directly on normalized values
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
                        break

                    # Scale loss by 1/num_updates for gradient accumulation across multiple updates
                    num_updates = (
                        total_chunks + chunks_per_grad - 1
                    ) // chunks_per_grad
                    update_weight = 1.0 / num_updates if num_updates > 0 else 1.0
                    scaled_loss = muni_loss * update_weight

                    # Skip backward if loss is invalid
                    if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
                        print(
                            f"  ❌ DEBUG: Skipping backward for municipality {municipality_code} - scaled_loss is invalid: {scaled_loss.item()}"
                        )
                        break

                    scaler.scale(scaled_loss).backward()
                    batch_has_gradients = True  # Mark that gradients were accumulated

                    pixel_acc.start_new_grad_segment()
                    del municipality_X_chunk, chunk_predictions
                    if is_last_chunk:
                        break
                    continue

                del municipality_X_chunk, chunk_predictions

            if chunk_idx == 0 and num_pixels > 0:
                print(
                    f"  ⚠️  Muni {municipality_code}: no pixel chunks loaded "
                    f"(year={year}, pixels={num_pixels})",
                    flush=True,
                )

            if pixel_acc.valid and chunk_idx > 0:
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
                        target_denorm = (
                            target_normalized.item() * target_std + target_mean
                        )
                        pred_denorm = (
                            municipality_agg_normalized.item() * target_std
                            + target_mean
                        )
                    else:
                        target_denorm = target_normalized.item()
                        pred_denorm = municipality_agg_normalized.item()
                    total_loss += final_loss.item()
                    fmt = ".2f" if aggregation == "mean" else ".0f"
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    pbar.write(
                        f"[{timestamp}] Muni {municipality_code}: "
                        f"target={target_denorm:{fmt}}, pred={pred_denorm:{fmt}}, "
                        f"chunks={chunk_idx}/{total_chunks}, loss={final_loss.item():.2e}"
                    )

            # Step optimizer after processing all municipalities in the batch
            # Only if at least one municipality accumulated gradients
            if (muni_idx + 1) == num_municipalities:
                if batch_has_gradients:
                    has_nan_grad = any(
                        p.grad is not None and torch.isnan(p.grad).any()
                        for p in model.parameters()
                    )

                    if has_nan_grad:
                        optimizer.zero_grad()
                    else:
                        max_grad_norm = 5.0
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=max_grad_norm
                        )
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    # No gradients accumulated in this batch, skip optimizer step
                    optimizer.zero_grad()
                    print(
                        f"  ⚠️  DEBUG: Skipping optimizer step for batch {idx} - no valid gradients accumulated"
                    )

        # Calculate average loss only if we have valid municipalities
        if num_municipalities > 0 and total_loss >= 0:
            loss_value = total_loss / num_municipalities
        else:
            loss_value = 0.0
            if num_municipalities == 0:
                print(f"  ⚠️  DEBUG: No municipalities in batch")
            elif total_loss < 0:
                print(f"  ⚠️  DEBUG: Negative total loss: {total_loss}")

        # Check for NaN or Inf
        if isinstance(loss_value, float) and (
            loss_value != loss_value or abs(loss_value) == float("inf")
        ):
            print(f"  ❌ DEBUG: Batch loss is invalid: {loss_value}")
            print(
                f"      Total loss: {total_loss}, Num municipalities: {num_municipalities}"
            )
            loss_value = 0.0

        if not batch_has_gradients:
            pbar.write(
                f"  ⚠️  batch {idx + 1}: no gradients "
                f"({num_municipalities} municipalities, chunk_size={chunk_size})"
            )

        pbar.set_description(f"train loss={loss_value:.2f}")
        pbar.update(1)
        losses.update(loss_value, len(municipalities))
        _batch_vram_cleanup(device)

    pbar.close()
    return losses.avg


def test_epoch_aggregated(
    model,
    criterion,
    dataloader,
    device,
    args,
    target_mean=None,
    target_std=None,
):
    """Test/validation epoch."""
    from torch.amp import autocast
    from tqdm import tqdm

    from utils import AverageMeter

    losses = AverageMeter("Loss", ":.4e")
    model.eval()
    aggregation = getattr(criterion, "aggregation", "sum")
    target_column = getattr(criterion, "target_column", "production_t")
    all_aggregated_preds = []
    all_targets = []
    chunk_size = _pixel_chunk_size(args)

    iterator_ctx = tqdm(enumerate(dataloader), total=len(dataloader), leave=True)

    with torch.no_grad():
        for idx, (municipalities, targets, num_pixels_list, years) in iterator_ctx:
            targets = targets.to(device).float()
            predictions_list = []

            for muni_idx, municipality_code in enumerate(municipalities):
                dataset = dataloader.dataset
                pixel_predictions_chunks = []

                # Get year for this municipality (if multi-year mode)
                year = years[muni_idx] if years[muni_idx] is not None else None
                use_pipeline_h2d = _pipeline_h2d(args, device)
                chunk_source = iter_municipality_pixel_chunks(
                    dataset,
                    municipality_code,
                    year=year,
                    chunk_size=chunk_size,
                    prefetch_depth=_prefetch_depth(args),
                    device=device,
                    pipeline_h2d=use_pipeline_h2d,
                )
                for chunk_item in chunk_source:
                    if use_pipeline_h2d:
                        municipality_X_chunk = chunk_item
                    else:
                        municipality_X_chunk = prepare_chunk_on_device(chunk_item, device)

                    with autocast("cuda", dtype=torch.bfloat16):
                        chunk_predictions = model(municipality_X_chunk)
                    pixel_predictions_chunks.append(chunk_predictions.cpu())

                pixel_predictions = torch.cat(pixel_predictions_chunks, dim=0).to(
                    device
                )
                predictions_list.append(pixel_predictions)

            # Use bfloat16 for faster computation with Tensor Cores
            # Note: criterion handles aggregation internally
            with autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(predictions_list, targets, num_pixels_list)
            losses.update(loss.item(), len(municipalities))
            _batch_vram_cleanup(device)

            aggregated_preds = torch.stack(
                [aggregate_pixels(pred, aggregation) for pred in predictions_list]
            )
            if aggregated_preds.dim() > 1 and aggregated_preds.size(1) == 1:
                aggregated_preds = aggregated_preds.squeeze(1)

            # Normalize predictions first (model outputs raw values, but targets are normalized)
            # Then denormalize both for metrics computation in original scale
            if target_mean is not None and target_std is not None:
                # Normalize predictions to match normalized targets
                aggregated_preds_normalized = (
                    aggregated_preds - target_mean
                ) / target_std
                # Now both are in normalized space, denormalize for metrics
                aggregated_preds = (
                    aggregated_preds_normalized * target_std + target_mean
                )
                targets = targets * target_std + target_mean

            # Convert to float32 before numpy conversion (bfloat16 not supported by NumPy)
            all_aggregated_preds.append(aggregated_preds.cpu().float().numpy())
            all_targets.append(targets.cpu().numpy())

        all_aggregated_preds = np.concatenate(all_aggregated_preds)
        all_targets = np.concatenate(all_targets)
        scores = regression_metrics(
            all_aggregated_preds, all_targets, target_column=target_column
        )

    return losses.avg, scores
