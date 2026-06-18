"""
Utilities for aggregated regression: pixel-level predictions → municipality-level targets.
"""

from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

# Pixels per forward pass
MAX_PIXEL_BATCH_SIZE = 160

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
            if not pixel_chunk:
                continue
            chunk_x = torch.stack([p[0] for p in pixel_chunk])
            chunk_mask = torch.stack([p[1] for p in pixel_chunk])
            chunk_doy = torch.stack([p[2] for p in pixel_chunk])
            chunk_weight = torch.stack([p[3] for p in pixel_chunk])
            municipality_X_chunk = recursive_todevice(
                (chunk_x, chunk_mask, chunk_doy, chunk_weight), device
            )
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

    from utils import AverageMeter, recursive_todevice

    losses = AverageMeter("Loss", ":.4e")
    model.train()
    aggregation = getattr(criterion, "aggregation", "sum")
    scaler = GradScaler("cuda")

    # Debug: Store initial weights for first batch to check if they change
    if hasattr(train_epoch_aggregated, "_first_batch"):
        first_batch = False
    else:
        first_batch = True
        train_epoch_aggregated._first_batch = False
        # Store a sample weight to check if it changes
        sample_param = next(iter(model.parameters()))
        train_epoch_aggregated._initial_weight = sample_param.data.clone()

    CHUNKS_PER_GRAD_UPDATE = (
        50  # Lower is fine: mainly affects gradient accumulation, not computation speed
    )

    with tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        leave=True,
    ) as iterator:
        for idx, (municipalities, targets, num_pixels_list, years) in iterator:
            targets = targets.to(device).float()
            total_loss = 0.0
            num_municipalities = len(municipalities)
            batch_has_gradients = (
                False  # Track if any gradients were accumulated in this batch
            )

            optimizer.zero_grad()

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
                total_chunks = (
                    num_pixels + MAX_PIXEL_BATCH_SIZE - 1
                ) // MAX_PIXEL_BATCH_SIZE

                # Skip municipalities with zero pixels
                if total_chunks == 0 or num_pixels == 0:
                    print(
                        f"  ⚠️  Warning: Municipality {municipality_code} has 0 pixels, skipping"
                    )
                    continue

                # Get year for this municipality (if multi-year mode)
                year = years[muni_idx] if years[muni_idx] is not None else None
                for pixel_chunk in dataset.load_pixels_from_municipality(
                    municipality_code, year=year, chunk_size=MAX_PIXEL_BATCH_SIZE
                ):
                    # Skip empty or single-sample chunks (BatchNorm needs batch_size > 1)
                    if len(pixel_chunk) < 2:
                        continue

                    chunk_x = torch.stack([p[0] for p in pixel_chunk])
                    chunk_mask = torch.stack([p[1] for p in pixel_chunk])
                    chunk_doy = torch.stack([p[2] for p in pixel_chunk])
                    chunk_weight = torch.stack([p[3] for p in pixel_chunk])

                    # Debug: Check input data for NaN/Inf
                    if torch.isnan(chunk_x).any() or torch.isinf(chunk_x).any():
                        print(
                            f"  ❌ DEBUG: Municipality {municipality_code} chunk {chunk_idx} has invalid input data (x)"
                        )
                        print(
                            f"      X: min={chunk_x.min().item():.4f}, max={chunk_x.max().item():.4f}, NaN={torch.isnan(chunk_x).sum().item()}, Inf={torch.isinf(chunk_x).sum().item()}"
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
                    if (
                        torch.isnan(chunk_weight).any()
                        or torch.isinf(chunk_weight).any()
                    ):
                        print(
                            f"  ❌ DEBUG: Municipality {municipality_code} chunk {chunk_idx} has invalid weight"
                        )
                        continue

                    municipality_X_chunk = (
                        chunk_x,
                        chunk_mask,
                        chunk_doy,
                        chunk_weight,
                    )
                    municipality_X_chunk = recursive_todevice(
                        municipality_X_chunk, device
                    )

                    # Use bfloat16 for faster computation with Tensor Cores
                    with autocast("cuda", dtype=torch.bfloat16):
                        chunk_predictions = model(municipality_X_chunk)

                    # Clip predictions to prevent extreme values that could cause overflow
                    chunk_predictions = torch.clamp(
                        chunk_predictions, min=-1e4, max=1e4
                    )

                    # Debug: Check predictions
                    if (
                        torch.isnan(chunk_predictions).any()
                        or torch.isinf(chunk_predictions).any()
                    ):
                        print(
                            f"  ❌ DEBUG: Municipality {municipality_code} chunk {chunk_idx} has invalid predictions"
                        )
                        print(
                            f"      Predictions: min={chunk_predictions.min().item():.4f}, max={chunk_predictions.max().item():.4f}, mean={chunk_predictions.mean().item():.4f}"
                        )
                        print(
                            f"      Input X stats: min={chunk_x.min().item():.4f}, max={chunk_x.max().item():.4f}, mean={chunk_x.mean().item():.4f}"
                        )
                        print(
                            f"      Input shape: {chunk_x.shape}, mask shape: {chunk_mask.shape}"
                        )
                        continue

                    pixel_acc.add(chunk_predictions)
                    if pixel_acc.is_extreme():
                        agg_val = pixel_acc.value()
                        abs_val = (
                            torch.abs(agg_val).max().item()
                            if agg_val is not None
                            else float("nan")
                        )
                        print(
                            f"  ⚠️  DEBUG: Municipality {municipality_code} has extreme "
                            f"{aggregation} at chunk {chunk_idx}: {abs_val:.2f}"
                        )
                        pixel_acc = MunicipalityPixelAccumulator(aggregation)
                        break

                    chunk_idx += 1

                    # Do gradient updates periodically to avoid memory issues
                    chunks_in_group = chunk_idx % CHUNKS_PER_GRAD_UPDATE
                    is_last_chunk = chunk_idx == total_chunks
                    should_update = (chunks_in_group == 0) or is_last_chunk

                    if should_update and pixel_acc.valid:
                        if pixel_acc.is_extreme():
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
                            pixel_acc.pixel_count / num_pixels
                            if num_pixels > 0
                            else 1.0
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
                            total_chunks + CHUNKS_PER_GRAD_UPDATE - 1
                        ) // CHUNKS_PER_GRAD_UPDATE
                        update_weight = 1.0 / num_updates if num_updates > 0 else 1.0
                        scaled_loss = muni_loss * update_weight

                        # Skip backward if loss is invalid
                        if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
                            print(
                                f"  ❌ DEBUG: Skipping backward for municipality {municipality_code} - scaled_loss is invalid: {scaled_loss.item()}"
                            )
                            break

                        scaler.scale(scaled_loss).backward()
                        batch_has_gradients = (
                            True  # Mark that gradients were accumulated
                        )

                        pixel_acc.detach()

                        # Continue processing remaining chunks (don't exit loop early)
                        # Only break if this was the last chunk
                        if is_last_chunk:
                            break
                        continue

                if pixel_acc.valid and chunk_idx > 0 and not pixel_acc.is_extreme():
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
                        fmt = ".2f" if aggregation == "mean" else ".0f"
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(
                            f"[{timestamp}] Muni {municipality_code}: "
                            f"target={target_denorm:{fmt}}, pred={pred_denorm:{fmt}}, "
                            f"chunks={chunk_idx}/{total_chunks}, loss={final_loss.item():.2e}"
                        )
                        total_loss += final_loss.item()

                # Step optimizer after processing all municipalities in the batch
                # Only if at least one municipality accumulated gradients
                if (muni_idx + 1) == num_municipalities:
                    if batch_has_gradients:
                        # Debug: Check if gradients exist before stepping
                        has_grad = any(
                            p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.parameters()
                        )
                        # Check for NaN gradients before optimizer step
                        has_nan_grad = any(
                            p.grad is not None and torch.isnan(p.grad).any()
                            for p in model.parameters()
                        )

                        if has_nan_grad:
                            optimizer.zero_grad()
                        else:
                            # Clip gradients to prevent explosion
                            # Max norm of 5.0 is reasonable for normalized loss values
                            # Allows gradients to flow while preventing extreme values
                            max_grad_norm = 5.0
                            scaler.unscale_(
                                optimizer
                            )  # Unscale gradients before clipping
                            # clip_grad_norm_ returns the total norm before clipping
                            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=max_grad_norm
                            )

                            clipped_status = (
                                "clipped"
                                if total_grad_norm > max_grad_norm
                                else "not clipped"
                            )
                            print(
                                f"  🔧 Optimizer step: grad_norm={total_grad_norm:.2e} ({clipped_status}, max={max_grad_norm:.2e}), lr={optimizer.param_groups[0]['lr']:.2e}"
                            )

                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()

                            # Debug: Check if weights changed (only for first batch)
                            if first_batch and idx == 0:
                                sample_param = next(iter(model.parameters()))
                                weight_diff = (
                                    (
                                        sample_param.data
                                        - train_epoch_aggregated._initial_weight
                                    )
                                    .abs()
                                    .max()
                                    .item()
                                )
                                if weight_diff < 1e-10:
                                    print(f"      ⚠️  WARNING: Weights did not change!")
                                else:
                                    print(
                                        f"      ✓ Weights updated (change: {weight_diff:.2e})"
                                    )
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

            iterator.set_description(f"train loss={loss_value:.2f}")
            losses.update(loss_value, len(municipalities))

    return losses.avg


def test_epoch_aggregated(
    model, criterion, dataloader, device, args, target_mean=None, target_std=None
):
    """Test/validation epoch."""
    from torch.amp import autocast
    from tqdm import tqdm

    from utils import AverageMeter, recursive_todevice

    losses = AverageMeter("Loss", ":.4e")
    model.eval()
    aggregation = getattr(criterion, "aggregation", "sum")
    target_column = getattr(criterion, "target_column", "production_t")
    all_aggregated_preds = []
    all_targets = []

    with torch.no_grad():
        with tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            leave=True,
        ) as iterator:
            for idx, (municipalities, targets, num_pixels_list, years) in iterator:
                targets = targets.to(device).float()
                predictions_list = []

                for muni_idx, municipality_code in enumerate(municipalities):
                    dataset = dataloader.dataset
                    pixel_predictions_chunks = []

                    # Get year for this municipality (if multi-year mode)
                    year = years[muni_idx] if years[muni_idx] is not None else None
                    for pixel_chunk in dataset.load_pixels_from_municipality(
                        municipality_code, year=year, chunk_size=MAX_PIXEL_BATCH_SIZE
                    ):
                        # Skip empty or single-sample chunks
                        if len(pixel_chunk) < 2:
                            continue

                        chunk_x = torch.stack([p[0] for p in pixel_chunk])
                        chunk_mask = torch.stack([p[1] for p in pixel_chunk])
                        chunk_doy = torch.stack([p[2] for p in pixel_chunk])
                        chunk_weight = torch.stack([p[3] for p in pixel_chunk])

                        # Ensure chunk_x has correct shape: [batch_size, seq_len, features]
                        # If it's 2D, add batch dimension
                        if chunk_x.dim() == 2:
                            chunk_x = chunk_x.unsqueeze(0)
                            chunk_mask = chunk_mask.unsqueeze(0)
                            chunk_doy = chunk_doy.unsqueeze(0)
                            chunk_weight = chunk_weight.unsqueeze(0)

                        municipality_X_chunk = (
                            chunk_x,
                            chunk_mask,
                            chunk_doy,
                            chunk_weight,
                        )
                        municipality_X_chunk = recursive_todevice(
                            municipality_X_chunk, device
                        )

                        # Use bfloat16 for faster computation with Tensor Cores
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
