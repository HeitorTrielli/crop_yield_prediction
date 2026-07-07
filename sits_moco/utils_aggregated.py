"""
Utilities for aggregated regression: pixel-level predictions → municipality-level targets.
"""

from __future__ import annotations

import gc
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from datasets.pixel_chunk import prepare_chunk_on_device, unpack_pixel_chunk
from training.accumulator import MunicipalityPixelAccumulator


def _pixel_chunk_size(args) -> int:
    from training.pipeline import pixel_chunk_size

    return pixel_chunk_size(args)


def _prefetch_depth(args) -> int:
    from training.pipeline import prefetch_depth

    return prefetch_depth(args)


def _pipeline_h2d(args, device: torch.device) -> bool:
    from training.pipeline import pipeline_h2d

    return pipeline_h2d(args, device)


def _chunk_debug_syncs(args) -> bool:
    from training.pipeline import chunk_debug_syncs

    return chunk_debug_syncs(args)


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _log_training(msg: str) -> None:
    print(msg, flush=True)


def _format_epoch_batch_bar(
    *,
    prefix: str,
    loss_value: float,
    batch_idx: int,
    batch_total: int,
    epoch_start: float,
    batch_start: float,
    batch_pixels: int | None = None,
) -> str:
    """One-line batch summary (printed once per batch, not a live tqdm bar)."""
    done = batch_idx + 1
    pct = 100.0 * done / batch_total if batch_total else 100.0
    epoch_elapsed = time.perf_counter() - epoch_start
    batch_elapsed = time.perf_counter() - batch_start
    avg_batch = epoch_elapsed / done if done else batch_elapsed
    remaining = avg_batch * max(batch_total - done, 0)
    timing = (
        f"{_format_duration(epoch_elapsed)}<{_format_duration(remaining)}, "
        f"{batch_elapsed:.1f}s/batch"
    )
    if batch_pixels is not None and batch_elapsed > 0:
        pixels_per_sec = batch_pixels / batch_elapsed
        timing += f", {pixels_per_sec:.1f} pixels/s"
    return (
        f"{prefix} loss={loss_value:.2f}: "
        f"{done}/{batch_total} ({pct:.0f}%) "
        f"[{timing}]"
    )


def _batch_vram_cleanup(device: torch.device, args=None) -> None:
    """gc + empty_cache after each dataloader batch."""
    from training_runtime import batch_vram_cleanup

    if args is not None:
        batch_vram_cleanup(device, args)
        return
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    gc.collect()
    torch.cuda.empty_cache()


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
    run_stats: dict | None = None,
):
    """Training epoch: process municipalities, sum pixel predictions, compare to targets."""
    from torch.amp import GradScaler

    from training.train_batch import process_train_batch
    from training_runtime import zero_grad
    from utils import AverageMeter

    losses = AverageMeter("Loss", ":.4e")
    model.train()
    aggregation = getattr(criterion, "aggregation", "sum")
    scaler = GradScaler("cuda")
    chunk_size = _pixel_chunk_size(args)

    batch_total = len(dataloader)
    epoch_start = time.perf_counter()

    for idx, (municipalities, targets, num_pixels_list, years) in enumerate(dataloader):
        batch_start = time.perf_counter()
        num_municipalities = len(municipalities)
        batch_has_gradients = False

        zero_grad(optimizer, args)
        _log_training(f"── batch {idx + 1}/{batch_total} ({num_municipalities} muni) ──")

        total_loss, batch_has_gradients, _ = process_train_batch(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            dataset=dataloader.dataset,
            municipalities=municipalities,
            targets=targets,
            num_pixels_list=num_pixels_list,
            years=years,
            device=device,
            args=args,
            aggregation=aggregation,
            target_mean=target_mean,
            target_std=target_std,
            chunk_size=chunk_size,
            batch_idx=idx,
            run_stats=run_stats,
        )

        if num_municipalities > 0 and total_loss >= 0:
            loss_value = total_loss / num_municipalities
        else:
            loss_value = 0.0
            if num_municipalities == 0:
                print(f"  ⚠️  DEBUG: No municipalities in batch")
            elif total_loss < 0:
                print(f"  ⚠️  DEBUG: Negative total loss: {total_loss}")

        if isinstance(loss_value, float) and (
            loss_value != loss_value or abs(loss_value) == float("inf")
        ):
            print(f"  ❌ DEBUG: Batch loss is invalid: {loss_value}")
            print(
                f"      Total loss: {total_loss}, Num municipalities: {num_municipalities}"
            )
            loss_value = 0.0

        if not batch_has_gradients:
            _log_training(
                f"  ⚠️  batch {idx + 1}: no gradients "
                f"({num_municipalities} municipalities, chunk_size={chunk_size})"
            )

        _log_training(
            _format_epoch_batch_bar(
                prefix="train",
                loss_value=loss_value,
                batch_idx=idx,
                batch_total=batch_total,
                epoch_start=epoch_start,
                batch_start=batch_start,
                batch_pixels=sum(num_pixels_list),
            )
        )
        losses.update(loss_value, len(municipalities))
        if run_stats is not None:
            run_stats["batches_processed"] = run_stats.get("batches_processed", 0) + 1
            run_stats.setdefault("batch_times_s", []).append(
                time.perf_counter() - batch_start
            )
        _batch_vram_cleanup(device, args)

    return losses.avg


def run_training_vram_probe(
    *,
    model,
    optimizer,
    criterion,
    dataloader,
    device,
    args,
    target_mean=None,
    target_std=None,
    vram_frac: float,
    probe_min_chunks: int,
) -> dict:
    """
    Process pixel chunks until dedicated VRAM peak is measured under budget.

    Runs long enough to simulate mid-batch training (two backward groups plus
    prefetch/H2D warmup). Tracks peak dedicated VRAM across chunks; skips the
    optimizer step so gradients accumulate like a real dataloader batch.
    """
    from torch.amp import GradScaler

    from training.train_batch import process_train_batch
    from training_runtime import zero_grad

    model.train()
    aggregation = getattr(criterion, "aggregation", "sum")
    scaler = GradScaler("cuda")
    chunk_size = _pixel_chunk_size(args)
    state = {
        "vram_frac": float(vram_frac),
        "min_chunks": max(1, int(probe_min_chunks)),
        "chunks": 0,
        "over_limit": False,
        "complete": False,
        "last_pressure": None,
        "peak_pressure": None,
    }

    for batch_idx, (municipalities, targets, num_pixels_list, years) in enumerate(
        dataloader
    ):
        if state["over_limit"] or state["complete"]:
            break

        zero_grad(optimizer, args)
        process_train_batch(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            dataset=dataloader.dataset,
            municipalities=municipalities,
            targets=targets,
            num_pixels_list=num_pixels_list,
            years=years,
            device=device,
            args=args,
            aggregation=aggregation,
            target_mean=target_mean,
            target_std=target_std,
            chunk_size=chunk_size,
            batch_idx=batch_idx,
            vram_probe_state=state,
        )
        if state["over_limit"] or state["complete"]:
            break

    return state


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

    from training.eval_batch import run_eval_batch
    from utils import AverageMeter

    losses = AverageMeter("Loss", ":.4e")
    model.eval()
    aggregation = getattr(criterion, "aggregation", "sum")
    target_column = getattr(criterion, "target_column", "production_t")
    all_aggregated_preds = []
    all_targets = []
    chunk_size = _pixel_chunk_size(args)
    batch_total = len(dataloader)
    epoch_start = time.perf_counter()

    with torch.no_grad():
        for idx, (municipalities, targets, num_pixels_list, years) in enumerate(
            dataloader
        ):
            batch_start = time.perf_counter()
            targets = targets.to(device).float()

            predictions_list = run_eval_batch(
                model=model,
                dataset=dataloader.dataset,
                municipalities=municipalities,
                targets=targets,
                num_pixels_list=num_pixels_list,
                years=years,
                device=device,
                args=args,
                chunk_size=chunk_size,
            )

            with autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(predictions_list, targets, num_pixels_list)
            loss_value = loss.item()
            losses.update(loss_value, len(municipalities))
            _batch_vram_cleanup(device, args)
            _log_training(
                _format_epoch_batch_bar(
                    prefix="val",
                    loss_value=loss_value,
                    batch_idx=idx,
                    batch_total=batch_total,
                    epoch_start=epoch_start,
                    batch_start=batch_start,
                    batch_pixels=sum(num_pixels_list),
                )
            )

            aggregated_preds = torch.stack(
                [aggregate_pixels(pred, aggregation) for pred in predictions_list]
            )
            if aggregated_preds.dim() > 1 and aggregated_preds.size(1) == 1:
                aggregated_preds = aggregated_preds.squeeze(1)

            if target_mean is not None and target_std is not None:
                aggregated_preds_normalized = (
                    aggregated_preds - target_mean
                ) / target_std
                aggregated_preds = (
                    aggregated_preds_normalized * target_std + target_mean
                )
                targets = targets * target_std + target_mean

            all_aggregated_preds.append(aggregated_preds.cpu().float().numpy())
            all_targets.append(targets.cpu().numpy())

        all_aggregated_preds = np.concatenate(all_aggregated_preds)
        all_targets = np.concatenate(all_targets)
        scores = regression_metrics(
            all_aggregated_preds, all_targets, target_column=target_column
        )

    return losses.avg, scores
