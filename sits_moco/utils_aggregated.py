"""
Utilities for aggregated regression: pixel-level predictions → municipality-level targets.
"""

import numpy as np
import torch
import torch.nn as nn


def regression_metrics(y_pred, y_true):
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

    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


class AggregatedMSELoss(nn.Module):
    """Loss function: sums pixel predictions per municipality, compares to municipality target."""

    def __init__(
        self, reduction="mean", loss_scale_factor=1e4, target_mean=None, target_std=None
    ):
        super().__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction="none")
        self.loss_scale_factor = loss_scale_factor
        self.target_mean = target_mean if target_mean is not None else 0.0
        self.target_std = target_std if target_std is not None else 1.0
        self.normalize = target_mean is not None and target_std is not None

    def forward(self, predictions_list, targets, num_pixels_list=None):
        """
        Args:
            predictions_list: List of tensors, one per municipality [num_pixels, num_outputs]
            targets: [batch_size] - Municipality-level targets (already normalized if normalize=True)
        """
        aggregated_predictions = []
        for pred in predictions_list:
            municipality_sum = pred.sum(dim=0)  # Sum over pixels: [num_outputs]
            aggregated_predictions.append(municipality_sum)

        aggregated = torch.stack(aggregated_predictions)  # [batch_size, num_outputs]

        if aggregated.dim() > 1 and aggregated.size(1) == 1:
            aggregated = aggregated.squeeze(1)
        if targets.dim() > 1 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        # Normalize predictions if normalization is enabled (targets are already normalized)
        if self.normalize:
            aggregated = (aggregated - self.target_mean) / self.target_std

        # Scale predictions and targets before computing loss to avoid overflow in mixed precision
        aggregated_scaled = aggregated / self.loss_scale_factor
        targets_scaled = targets / self.loss_scale_factor

        # Compute loss on scaled values
        loss_scaled = self.mse(aggregated_scaled, targets_scaled)

        # Scale loss back to original scale
        loss = loss_scaled * (self.loss_scale_factor**2)

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

    for X_tuple, target, num_pixels in batch:
        municipalities.append(X_tuple)
        targets.append(target)
        num_pixels_list.append(num_pixels)

    targets = torch.stack(targets)
    return municipalities, targets, num_pixels_list


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

    # Prioritize MAX_PIXEL_BATCH_SIZE for speed (reduces forward passes)
    # Increase CHUNKS_PER_GRAD_UPDATE for stability (doesn't affect speed much)
    MAX_PIXEL_BATCH_SIZE = (
        2000  # Reduced to stay within dedicated VRAM (avoid slower shared memory)
    )
    CHUNKS_PER_GRAD_UPDATE = 200  # Lower is fine: mainly affects gradient accumulation, not computation speed

    # Loss scaling factor for mixed precision (bfloat16/float16)
    # Values are scaled down before loss computation to avoid overflow, then scaled back
    LOSS_SCALE_FACTOR = 1e4  # Scale predictions/targets by this before computing loss

    with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
        for idx, (municipalities, targets, num_pixels_list) in iterator:
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
                municipality_sum = None
                chunk_idx = 0
                # Ensure municipality_sum will be float32 (not float16 from autocast)
                municipality_sum_dtype = torch.float32
                total_chunks = (
                    num_pixels + MAX_PIXEL_BATCH_SIZE - 1
                ) // MAX_PIXEL_BATCH_SIZE

                # Skip municipalities with zero pixels
                if total_chunks == 0 or num_pixels == 0:
                    print(
                        f"  ⚠️  Warning: Municipality {municipality_code} has 0 pixels, skipping"
                    )
                    continue

                for pixel_chunk in dataset.load_pixels_from_municipality(
                    municipality_code, chunk_size=MAX_PIXEL_BATCH_SIZE
                ):
                    # Skip empty chunks
                    if len(pixel_chunk) == 0:
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

                    chunk_sum = chunk_predictions.sum(dim=0)
                    # Convert to float32 to avoid float16 overflow (float16 max is 65504)
                    chunk_sum = chunk_sum.to(torch.float32)

                    # Check if chunk_sum is inf before accumulating
                    chunk_sum_is_inf = torch.isinf(chunk_sum)
                    if chunk_sum.dim() == 0:
                        chunk_sum_is_inf_value = chunk_sum_is_inf.item()
                        chunk_sum_value = chunk_sum.item()
                    else:
                        chunk_sum_is_inf_value = chunk_sum_is_inf.any().item()
                        chunk_sum_value = (
                            chunk_sum.item()
                            if chunk_sum.numel() == 1
                            else chunk_sum.max().item()
                        )

                    if chunk_sum_is_inf_value:
                        print(
                            f"  ❌ DEBUG: Municipality {municipality_code} chunk {chunk_idx} has inf chunk_sum!"
                        )
                        print(
                            f"      Chunk sum: {chunk_sum_value}, Predictions range: [{chunk_predictions.min().item():.2f}, {chunk_predictions.max().item():.2f}]"
                        )
                        print(
                            f"      Predictions sum check: {chunk_predictions.sum().item():.2f}, is_inf: {torch.isinf(chunk_predictions.sum()).item()}"
                        )
                        municipality_sum = None
                        break

                    # Check previous accumulated sum before adding
                    if municipality_sum is not None:
                        prev_sum_is_inf = torch.isinf(municipality_sum)
                        if municipality_sum.dim() == 0:
                            prev_sum_is_inf_value = prev_sum_is_inf.item()
                            prev_sum_value = municipality_sum.item()
                        else:
                            prev_sum_is_inf_value = prev_sum_is_inf.any().item()
                            prev_sum_value = municipality_sum.max().item()

                        if prev_sum_is_inf_value:
                            print(
                                f"  ❌ DEBUG: Municipality {municipality_code} chunk {chunk_idx} - previous sum was already inf: {prev_sum_value}"
                            )
                            municipality_sum = None
                            break

                    # Ensure float32 dtype to avoid float16 overflow
                    if municipality_sum is None:
                        municipality_sum = chunk_sum.to(torch.float32)
                    else:
                        # Ensure both are float32 before adding
                        municipality_sum = municipality_sum.to(
                            torch.float32
                        ) + chunk_sum.to(torch.float32)

                    # Check if accumulated sum is becoming extreme (inf or very large)
                    # Handle both scalar and tensor cases
                    is_inf = torch.isinf(municipality_sum)
                    if municipality_sum.dim() == 0:
                        is_inf_value = is_inf.item()
                        abs_sum = torch.abs(municipality_sum).item()
                    else:
                        is_inf_value = is_inf.any().item()
                        abs_sum = torch.abs(municipality_sum).max().item()

                    if (
                        is_inf_value or abs_sum > 1e8
                    ):  # Increased threshold from 1e7 to 1e8
                        print(
                            f"  ⚠️  DEBUG: Municipality {municipality_code} has extreme accumulated sum at chunk {chunk_idx}"
                        )
                        print(
                            f"      Accumulated sum: {abs_sum:.2f} (inf={is_inf_value})"
                        )
                        print(
                            f"      Chunk sum: {chunk_sum_value:.2f}, Predictions range: [{chunk_predictions.min().item():.2f}, {chunk_predictions.max().item():.2f}]"
                        )
                        print(
                            f"      Chunk size: {len(pixel_chunk)}, Total chunks so far: {chunk_idx}"
                        )
                        if municipality_sum is not None:
                            print(
                                f"      Previous sum before adding: {prev_sum_value:.2f} (was inf: {prev_sum_is_inf_value})"
                            )
                        municipality_sum = (
                            None  # Mark as invalid to skip this municipality
                        )
                        break

                    chunk_idx += 1

                    # Do gradient updates periodically to avoid memory issues
                    chunks_in_group = chunk_idx % CHUNKS_PER_GRAD_UPDATE
                    is_last_chunk = chunk_idx == total_chunks
                    should_update = (chunks_in_group == 0) or is_last_chunk

                    if should_update and municipality_sum is not None:
                        # Check if sum is extreme before using it
                        if (
                            torch.isinf(municipality_sum).any()
                            or torch.abs(municipality_sum).max() > 1e7
                        ):
                            print(
                                f"  ⚠️  DEBUG: Municipality {municipality_code} has extreme sum at update: {municipality_sum.item():.2f}"
                            )
                            municipality_sum = None
                            break

                        # Normalize the accumulated sum
                        municipality_sum_normalized = municipality_sum.squeeze()
                        if municipality_sum_normalized.dim() == 0:
                            municipality_sum_normalized = (
                                municipality_sum_normalized.unsqueeze(0)
                            )
                        elif municipality_sum_normalized.dim() > 1:
                            municipality_sum_normalized = (
                                municipality_sum_normalized.flatten()[0:1]
                            )

                        # Ensure target has same shape and dtype
                        target_normalized = target.squeeze().to(torch.float32)
                        if target_normalized.dim() == 0:
                            target_normalized = target_normalized.unsqueeze(0)

                        # Ensure municipality_sum_normalized is float32 for accumulation
                        municipality_sum_normalized = municipality_sum_normalized.to(
                            torch.float32
                        )

                        # Normalize predictions if normalization is enabled (targets are already normalized)
                        if target_mean is not None and target_std is not None:
                            municipality_sum_normalized = (
                                municipality_sum_normalized - target_mean
                            ) / target_std

                        # Compare accumulated sum to proportional target
                        # For intermediate updates, scale target by fraction of chunks processed
                        # This ensures we're comparing partial sum to partial target
                        fraction_processed = (
                            chunk_idx / total_chunks if total_chunks > 0 else 1.0
                        )
                        expected_partial_target = (
                            target_normalized * fraction_processed
                        ).to(torch.float32)

                        # Scale predictions and targets before computing loss to avoid overflow in mixed precision
                        # This allows us to use bfloat16 autocast while avoiding numerical issues
                        municipality_sum_scaled = (
                            municipality_sum_normalized / LOSS_SCALE_FACTOR
                        )
                        expected_partial_target_scaled = (
                            expected_partial_target / LOSS_SCALE_FACTOR
                        )

                        # Compute loss on scaled values (will be much smaller, avoiding overflow)
                        muni_loss_scaled = torch.nn.functional.mse_loss(
                            municipality_sum_scaled, expected_partial_target_scaled
                        )

                        # Scale loss back to original scale: loss = loss_scaled * scale^2
                        # This ensures gradients are computed correctly
                        muni_loss = muni_loss_scaled * (LOSS_SCALE_FACTOR**2)

                        # Don't print intermediate updates - only print final result per municipality

                        # Debug: Check loss
                        if torch.isnan(muni_loss) or torch.isinf(muni_loss):
                            print(
                                f"  ❌ DEBUG: Municipality {municipality_code} has invalid loss: {muni_loss.item()}"
                            )
                            print(
                                f"      Expected partial target: {expected_partial_target.item():.4f}, Sum: {municipality_sum_normalized.item():.4f}"
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

                        # Detach to break computational graph for next iteration
                        municipality_sum = municipality_sum.detach()

                        # Continue processing remaining chunks (don't exit loop early)
                        # Only break if this was the last chunk
                        if is_last_chunk:
                            break
                        continue

                # Calculate final loss for this municipality (for logging)
                if municipality_sum is not None and chunk_idx > 0:
                    # Check if sum is extreme before calculating loss
                    if (
                        torch.isinf(municipality_sum).any()
                        or torch.abs(municipality_sum).max() > 1e7
                    ):
                        print(
                            f"  ⚠️  DEBUG: Municipality {municipality_code} skipped - extreme sum: {municipality_sum.item():.2f}"
                        )
                        municipality_sum = None

                    if municipality_sum is not None:
                        municipality_sum_normalized = municipality_sum.squeeze()
                        if municipality_sum_normalized.dim() == 0:
                            municipality_sum_normalized = (
                                municipality_sum_normalized.unsqueeze(0)
                            )
                        elif municipality_sum_normalized.dim() > 1:
                            municipality_sum_normalized = (
                                municipality_sum_normalized.flatten()[0:1]
                            )

                        target_normalized = target.squeeze().to(torch.float32)
                        if target_normalized.dim() == 0:
                            target_normalized = target_normalized.unsqueeze(0)

                        # Ensure municipality_sum_normalized is float32
                        municipality_sum_normalized = municipality_sum_normalized.to(
                            torch.float32
                        )

                        # Normalize predictions if normalization is enabled (targets are already normalized)
                        if target_mean is not None and target_std is not None:
                            municipality_sum_normalized = (
                                municipality_sum_normalized - target_mean
                            ) / target_std

                        # Scale predictions and targets before computing loss to avoid overflow
                        municipality_sum_scaled = (
                            municipality_sum_normalized / LOSS_SCALE_FACTOR
                        )
                        target_scaled = target_normalized / LOSS_SCALE_FACTOR

                        # Compute loss on scaled values
                        final_loss_scaled = torch.nn.functional.mse_loss(
                            municipality_sum_scaled, target_scaled
                        )

                        # Scale loss back to original scale
                        final_loss = final_loss_scaled * (LOSS_SCALE_FACTOR**2)

                        if torch.isnan(final_loss) or torch.isinf(final_loss):
                            print(
                                f"  ❌ DEBUG: Municipality {municipality_code} final loss is invalid: {final_loss.item()}"
                            )
                            print(
                                f"      Target: {target_normalized.item()}, Sum: {municipality_sum_normalized.item()}"
                            )
                        else:
                            # Debug: Print final result for this municipality (one line only)
                            # Show denormalized values for readability
                            if target_mean is not None and target_std is not None:
                                target_denorm = (
                                    target_normalized.item() * target_std + target_mean
                                )
                                pred_denorm = (
                                    municipality_sum_normalized.item() * target_std
                                    + target_mean
                                )
                            else:
                                target_denorm = target_normalized.item()
                                pred_denorm = municipality_sum_normalized.item()
                            print(
                                f"  Muni {municipality_code}: target={target_denorm:.0f}, pred={pred_denorm:.0f}, chunks={chunk_idx}/{total_chunks}, loss={final_loss.item():.2e}"
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
                            # Max norm of 1.0 is standard; scale by loss_scale_factor to account for loss scaling
                            max_grad_norm = (
                                1.0 * LOSS_SCALE_FACTOR
                            )  # Scale clipping threshold (1e4)
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
    all_aggregated_preds = []
    all_targets = []

    MAX_PIXEL_BATCH_SIZE = (
        1000  # Increased to use more GPU memory (process 1000 pixels at once)
    )

    with torch.no_grad():
        with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
            for idx, (municipalities, targets, num_pixels_list) in iterator:
                targets = targets.to(device).float()
                predictions_list = []

                for muni_idx, municipality_code in enumerate(municipalities):
                    dataset = dataloader.dataset
                    pixel_predictions_chunks = []

                    for pixel_chunk in dataset.load_pixels_from_municipality(
                        municipality_code, chunk_size=MAX_PIXEL_BATCH_SIZE
                    ):
                        chunk_x = torch.stack([p[0] for p in pixel_chunk])
                        chunk_mask = torch.stack([p[1] for p in pixel_chunk])
                        chunk_doy = torch.stack([p[2] for p in pixel_chunk])
                        chunk_weight = torch.stack([p[3] for p in pixel_chunk])

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
                    [pred.sum(dim=0) for pred in predictions_list]
                )
                if aggregated_preds.dim() > 1 and aggregated_preds.size(1) == 1:
                    aggregated_preds = aggregated_preds.squeeze(1)

                # Denormalize predictions and targets for evaluation metrics
                if target_mean is not None and target_std is not None:
                    aggregated_preds = aggregated_preds * target_std + target_mean
                    targets = targets * target_std + target_mean

                # Convert to float32 before numpy conversion (bfloat16 not supported by NumPy)
                all_aggregated_preds.append(aggregated_preds.cpu().float().numpy())
                all_targets.append(targets.cpu().numpy())

        all_aggregated_preds = np.concatenate(all_aggregated_preds)
        all_targets = np.concatenate(all_targets)
        scores = regression_metrics(all_aggregated_preds, all_targets)

    return losses.avg, scores
