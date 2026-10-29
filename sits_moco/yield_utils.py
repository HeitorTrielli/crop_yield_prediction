# Yield / productivity helpers (formerly utils_aggregated).
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
    # Deviation from per-year train-set mean of yield_t_ha (yield anomaly).
    # Column is derived at load time; see ensure_productivity_dev_column().
    "productivity_dev": {
        "column": "yield_t_ha_dev",
        "aggregation": "mean",
        "unit": "t/ha_dev",
    },
}


def resolve_target(target: str) -> tuple[str, str, str]:
    """Return (target_column, aggregation, unit) for a target name."""
    spec = TARGET_SPECS.get(target)
    if spec is None:
        choices = ", ".join(TARGET_SPECS)
        raise ValueError(f"Unknown target {target!r}; expected one of: {choices}")
    return spec["column"], spec["aggregation"], spec["unit"]


def resolve_target_bundle(target: str) -> dict:
    """
    Resolve a training target.

    Returns dict with:
      target, target_names, target_columns, aggregations, units, num_outputs
    Lists have length 1; scalar aliases are also set
    (target_column, aggregation, target_unit).
    """
    col, agg, unit = resolve_target(target)
    return {
        "target": target,
        "target_names": [target],
        "target_columns": [col],
        "aggregations": [agg],
        "units": [unit],
        "num_outputs": 1,
        "target_column": col,
        "aggregation": agg,
        "target_unit": unit,
    }


HEAD_OUTPUT_ZSCORE = "zscore"
HEAD_OUTPUT_RAW = "raw"


def resolve_head_output(
    checkpoint: dict | None = None,
    run_config: dict | None = None,
    *,
    default: str | None = None,
) -> str:
    """Return ``zscore`` or ``raw`` for the regression decoder.

    New training emits z-scores of the municipal target (bias 0 = climatology).
    Checkpoints and run configs without ``head_output`` are treated as ``raw``
    so older models that emitted t/ha or tons keep working.
    """
    ck = checkpoint or {}
    if ck.get("head_output"):
        return str(ck["head_output"])
    computed = (run_config or {}).get("computed") or {}
    if computed.get("head_output"):
        return str(computed["head_output"])
    cli = (run_config or {}).get("cli") or {}
    if cli.get("head_output"):
        return str(cli["head_output"])
    if default is not None:
        return str(default)
    if checkpoint is not None or run_config is not None:
        return HEAD_OUTPUT_RAW
    return HEAD_OUTPUT_ZSCORE


def pixel_pool_for_head(aggregation, head_output: str):
    """Z-score heads vote for the municipal scalar, so pixels are mean-pooled."""
    if head_output != HEAD_OUTPUT_ZSCORE:
        return aggregation
    if isinstance(aggregation, str):
        return "mean"
    return type(aggregation)("mean" for _ in aggregation)


def _stat_to_numpy(stat) -> np.ndarray:
    if torch.is_tensor(stat):
        return stat.detach().cpu().numpy()
    return np.asarray(stat)


def denormalize_head_output(values, target_mean, target_std, head_output: str):
    """Map decoder outputs to original target units (z * σ + μ)."""
    if head_output != HEAD_OUTPUT_ZSCORE:
        return values
    if target_mean is None or target_std is None:
        return values

    if torch.is_tensor(values):
        mean = target_mean
        std = target_std
        if not torch.is_tensor(mean):
            mean = torch.as_tensor(mean, dtype=values.dtype, device=values.device)
        else:
            mean = mean.to(device=values.device, dtype=values.dtype)
        if not torch.is_tensor(std):
            std = torch.as_tensor(std, dtype=values.dtype, device=values.device)
        else:
            std = std.to(device=values.device, dtype=values.dtype)
        return values * std + mean

    if isinstance(values, np.ndarray):
        mean = np.asarray(_stat_to_numpy(target_mean), dtype=values.dtype)
        std = np.asarray(_stat_to_numpy(target_std), dtype=values.dtype)
        return values * std + mean

    mean = float(np.asarray(_stat_to_numpy(target_mean)).reshape(-1)[0])
    std = float(np.asarray(_stat_to_numpy(target_std)).reshape(-1)[0])
    return float(values) * std + mean


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
        "Re-train with --target total|total_adj|productivity|productivity_dev."
    )


def aggregation_for_target_column(target_column: str) -> str:
    for spec in TARGET_SPECS.values():
        if spec["column"] == target_column:
            return spec["aggregation"]
    raise ValueError(
        f"Unknown target column {target_column!r}; "
        f"expected one of: {', '.join(s['column'] for s in TARGET_SPECS.values())}"
    )


_MODEL_KWARG_FIELDS = (
    ("model_d_model", "d_model", int),
    ("model_n_head", "n_head", int),
    ("model_n_layers", "n_layers", int),
    ("model_d_inner", "d_inner", int),
    ("model_dropout", "dropout", float),
    ("decoder_type", "decoder_type", str),
    ("decoder_hidden_dims", "decoder_hidden_dims", str),
    ("decoder_dropout", "decoder_dropout", float),
    ("decoder_activation", "decoder_activation", str),
    ("decoder_norm", "decoder_norm", str),
    ("decoder_num_experts", "decoder_num_experts", int),
    ("temporal_pooling", "temporal_pooling", str),
    ("attn_pool_queries", "attn_pool_queries", int),
    ("soil_fusion", "soil_fusion", str),
    ("climate_pooling", "climate_pooling", str),
    ("climate_d_model", "climate_d_model", int),
    ("climate_n_head", "climate_n_head", int),
    ("climate_d_inner", "climate_d_inner", int),
    ("climate_max_seq_len", "climate_max_seq_len", int),
    ("climate_sequencelength", "climate_max_seq_len", int),
    ("static_input", "static_input", bool),
)


def resolve_model_kwargs(
    run_config: dict | None = None,
    checkpoint: dict | None = None,
) -> dict:
    """
    Return STNetRegression constructor kwargs from training config.

    Precedence: computed.model_kwargs > cli model_* fields > {} (class defaults).
    """
    computed = (run_config or {}).get("computed") or {}
    cli = (run_config or {}).get("cli") or {}

    mk = computed.get("model_kwargs")
    if mk:
        kwargs = {
            "d_model": int(mk["d_model"]),
            "n_head": int(mk["n_head"]),
            "n_layers": int(mk["n_layers"]),
            "d_inner": int(mk["d_inner"]),
            "dropout": float(mk["dropout"]),
            # absent in configs from before learned attention pooling existed
            "temporal_pooling": str(mk.get("temporal_pooling", "ndvi")),
            "attn_pool_queries": int(mk.get("attn_pool_queries", 4)),
            "soil_fusion": str(mk.get("soil_fusion", "early")),
            "static_input": bool(mk.get("static_input", False)),
        }
        if mk.get("climate_pooling") is not None:
            kwargs["climate_pooling"] = str(mk["climate_pooling"])
        for key in ("climate_d_model", "climate_n_head", "climate_d_inner"):
            if mk.get(key) is not None:
                kwargs[key] = int(mk[key])
        if mk.get("climate_max_seq_len") is not None:
            kwargs["climate_max_seq_len"] = int(mk["climate_max_seq_len"])
        for key in (
            "decoder_type",
            "decoder_hidden_dims",
            "decoder_activation",
            "decoder_norm",
        ):
            if mk.get(key) is not None:
                kwargs[key] = str(mk[key])
        if mk.get("decoder_dropout") is not None:
            kwargs["decoder_dropout"] = float(mk["decoder_dropout"])
        if mk.get("decoder_num_experts") is not None:
            kwargs["decoder_num_experts"] = int(mk["decoder_num_experts"])
        if mk.get("recon_mode") is not None:
            kwargs["recon_mode"] = str(mk["recon_mode"])
        if mk.get("aux_yield_probe") is not None:
            kwargs["aux_yield_probe"] = str(mk["aux_yield_probe"])
    else:
        kwargs = {}
        for cli_key, ctor_key, caster in _MODEL_KWARG_FIELDS:
            if cli_key in cli and cli[cli_key] is not None:
                kwargs[ctor_key] = caster(cli[cli_key])

    # Older checkpoints may omit soil_fusion; infer from weight shapes.
    if checkpoint is not None:
        state = checkpoint.get("model_state") or checkpoint
        if isinstance(state, dict):
            inferred = soil_fusion_from_state_dict(state)
            if is_dual_stnet_state_dict(state):
                kwargs["soil_fusion"] = inferred
            elif kwargs.get("soil_fusion", "early") == "early" and inferred == "late":
                kwargs["soil_fusion"] = "late"
    return kwargs


def is_dual_stnet_state_dict(state_dict: dict) -> bool:
    """True when weights belong to DualSTNet (spectral and/or climate trunks)."""
    has_spec = any(k.startswith("spectral.") for k in state_dict)
    has_clim = any(k.startswith("climate.") for k in state_dict)
    return has_spec or has_clim


def soil_fusion_from_state_dict(state_dict: dict, *, soil_dim: int = 4) -> str:
    """Infer early vs late soil fusion from STNet / DualSTNet weight shapes."""
    if is_dual_stnet_state_dict(state_dict):
        decoder_input_keys = (
            "decoder.0.weight",  # legacy
            "decoder.weight",  # linear
            "decoder.0.0.weight",  # MLP
            "decoder.blocks.0.branch.0.weight",  # residual
            "decoder.linear.weight",  # linear + nonlinear correction
            "decoder.gate.0.weight",  # gated
            "decoder.joint.0.0.weight",  # modality additive + interaction
            "decoder.experts.0.0.0.weight",  # mixture of MLP experts
            "decoder.layers.0.0.weight",  # dense
            "decoder.input.0.weight",  # highway
            "decoder.projection.weight",  # Fourier features
            "decoder.layers.0.linear.weight",  # maxout
        )
        dec_w = next(
            (state_dict[key] for key in decoder_input_keys if key in state_dict),
            None,
        )
        if dec_w is not None:
            decoder_in = int(dec_w.shape[1])
        else:
            # Pure additive heads expose the modality widths separately.
            additive = [
                value
                for key, value in state_dict.items()
                if key.startswith("decoder.additive.") and key.endswith(".weight")
            ]
            if additive:
                decoder_in = sum(int(weight.shape[1]) for weight in additive)
            else:
                # Modality-attention heads project each modality separately.
                projections = [
                    value
                    for key, value in state_dict.items()
                    if key.startswith("decoder.projections.")
                    and key.endswith(".weight")
                ]
                if not projections:
                    return "none"
                decoder_in = sum(int(weight.shape[1]) for weight in projections)
        pooled = 0
        d_model_w = state_dict.get("spectral.mlp1.2.lin.weight")
        if d_model_w is not None:
            spec_d = int(d_model_w.shape[0])
            if "spectral.attn_pool.queries" in state_dict:
                n_q = int(state_dict["spectral.attn_pool.queries"].shape[0])
                pooled += n_q * spec_d
            else:
                pooled += spec_d
        c_model_w = state_dict.get("climate.mlp1.2.lin.weight")
        if c_model_w is not None:
            clim_d = int(c_model_w.shape[0])
            if "climate.attn_pool.queries" in state_dict:
                n_qc = int(state_dict["climate.attn_pool.queries"].shape[0])
                pooled += n_qc * clim_d
            else:
                pooled += clim_d
        if decoder_in == pooled + int(soil_dim):
            return "late"
        return "none"

    mlp_w = state_dict.get("mlp1.0.lin.weight")
    dec_w = state_dict.get("decoder.0.weight")
    d_model_w = state_dict.get("mlp1.2.lin.weight")
    if mlp_w is None or dec_w is None or d_model_w is None:
        return "early"
    mlp_in = int(mlp_w.shape[1])
    decoder_in = int(dec_w.shape[1])
    d_model = int(d_model_w.shape[0])
    if "attn_pool.queries" in state_dict:
        n_q = int(state_dict["attn_pool.queries"].shape[0])
        pooled = n_q * d_model
    else:
        pooled = d_model
    if decoder_in == pooled + int(soil_dim):
        return "late"
    _ = mlp_in
    return "early"


def resolve_inference_chunk_size(
    run_config: dict | None = None,
    chunk_size: int | None = None,
) -> int:
    """
    Pixels per GPU forward pass during inference.

    Defaults to training ``pixel_chunk_size`` (one training micro-batch), **not**
    ``pixel_chunk_size * chunks_per_grad`` (that product is the gradient
    accumulation budget and can exhaust VRAM during results generation).
    """
    if chunk_size is not None:
        return int(chunk_size)
    computed = (run_config or {}).get("computed") or {}
    cli = (run_config or {}).get("cli") or {}
    chunk_pipeline = computed.get("chunk_pipeline") or {}
    px_eff = chunk_pipeline.get("pixel_chunk_size_effective")
    if px_eff is not None:
        return int(px_eff)
    px = cli.get("pixel_chunk_size")
    if px is not None:
        return int(px)
    return 400


def resolve_pixel_chunk_size(
    run_config: dict | None = None,
    chunk_size: int | None = None,
) -> int:
    """Alias for :func:`resolve_inference_chunk_size`."""
    return resolve_inference_chunk_size(run_config, chunk_size)


def aggregate_pixels(
    predictions: torch.Tensor,
    aggregation: str | list[str] | tuple[str, ...] = "sum",
) -> torch.Tensor:
    """Aggregate pixel predictions [N, C] → [C] with per-dim sum/mean."""
    if isinstance(aggregation, str):
        if aggregation == "sum":
            return predictions.sum(dim=0)
        if aggregation == "mean":
            return predictions.mean(dim=0)
        raise ValueError(f"aggregation must be 'sum' or 'mean', got {aggregation!r}")

    aggs = list(aggregation)
    summed = predictions.sum(dim=0)
    n = float(predictions.shape[0])
    if len(aggs) == 1:
        return summed if aggs[0] == "sum" else summed / n
    if len(aggs) != int(summed.numel()):
        raise ValueError(
            f"Got {summed.numel()} prediction dims but {len(aggs)} aggregations"
        )
    out = summed.clone()
    for i, agg in enumerate(aggs):
        if agg == "mean":
            out[i] = out[i] / n
        elif agg != "sum":
            raise ValueError(f"aggregation must be 'sum' or 'mean', got {agg!r}")
    return out


def stnet_regression_input_dim_from_state_dict(
    state_dict: dict, *, soil_dim: int = 4
) -> int:
    """
    Infer STNetRegression **feature-cube** width from saved weights.

    For DualSTNet this is spectral_dim (+ soil_dim when late fusion).
    For early fusion this equals the MLP input width. For late soil fusion the
    MLP sees ``cube_dim - soil_dim`` while the batch still has ``cube_dim``
    channels (soil concatenated after temporal pooling).
    """
    if is_dual_stnet_state_dict(state_dict):
        w = state_dict.get("spectral.mlp1.0.lin.weight")
        if w is not None:
            spec_in = int(w.shape[1])
        else:
            # Climate-only DualSTNet: spectral cube still has 10 S2 bands (± soil).
            spec_in = 10
        if soil_fusion_from_state_dict(state_dict, soil_dim=soil_dim) == "late":
            return spec_in + int(soil_dim)
        return spec_in
    w = state_dict.get("mlp1.0.lin.weight")
    if w is None:
        return 10
    mlp_in = int(w.shape[1])
    if soil_fusion_from_state_dict(state_dict, soil_dim=soil_dim) == "late":
        return mlp_in + int(soil_dim)
    return mlp_in


def aggregate_municipality_from_pixel_chunks(
    model,
    pixel_chunks,
    device,
    aggregation: str = "sum",
    *,
    head_output: str = HEAD_OUTPUT_RAW,
    target_mean=None,
    target_std=None,
    debug_callback=None,
) -> float | None:
    """Run STNet on pixel chunks; aggregate to municipality prediction (sum or mean)."""
    from torch.amp import autocast

    from utils import recursive_todevice

    pool = pixel_pool_for_head(aggregation, head_output)
    acc = MunicipalityPixelAccumulator(pool)
    model.eval()
    with torch.no_grad():
        for chunk_index, pixel_chunk in enumerate(pixel_chunks):
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
            if debug_callback is not None:
                debug_callback(chunk_index, unpacked, chunk_predictions)
            acc.add(chunk_predictions)

    result = acc.value()
    if result is None:
        return None
    result = result.squeeze()
    if result.dim() > 0:
        result = result[0] if len(result) > 0 else torch.tensor(0.0)
    return float(
        denormalize_head_output(
            result.item(), target_mean, target_std, head_output
        )
    )


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
    # This prevents division by near-zero values that cause MAPE to explode.
    # Deviation targets cross zero often — MAPE is usually NaN / unreliable there.
    mean_y_true = np.mean(np.abs(y_true))
    if target_column == "yield_t_ha":
        floor = 0.1
    elif target_column == "yield_t_ha_dev":
        floor = 0.5
    else:
        floor = 100.0
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


def within_state_year_metrics(
    y_pred,
    y_true,
    municipality_codes,
    years,
    *,
    min_count: int = 20,
) -> dict[str, float]:
    """n-weighted mean R² / Spearman inside each (year × IBGE UF) cell.

    Ignores cells with fewer than ``min_count`` municipalities. Returns NaN when
    no cell qualifies. This is the honest multi-state ranking metric: pooled R²
    can look strong while within-state ranking is near zero.
    """
    from training.aux_losses import ibge_state_from_code

    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if len(municipality_codes) != len(y_pred) or len(years) != len(y_pred):
        return {
            "within_state_r2": float("nan"),
            "within_state_spearman": float("nan"),
            "within_state_n_cells": 0.0,
            "within_state_n": 0.0,
        }

    groups: dict[tuple, list[int]] = {}
    for i, (code, year) in enumerate(zip(municipality_codes, years)):
        if year is None:
            continue
        state = ibge_state_from_code(code)
        if state is None:
            continue
        if not np.isfinite(y_pred[i]) or not np.isfinite(y_true[i]):
            continue
        groups.setdefault((int(year), state), []).append(i)

    r2_w = 0.0
    sp_w = 0.0
    n_tot = 0
    n_cells = 0
    for indices in groups.values():
        if len(indices) < int(min_count):
            continue
        yp = y_pred[indices]
        yt = y_true[indices]
        cell = regression_metrics(yp, yt, target_column="yield_t_ha")
        # Spearman via rank Pearson (no scipy dependency here).
        pred_rank = yp.argsort().argsort().astype(np.float64)
        true_rank = yt.argsort().argsort().astype(np.float64)
        pred_c = pred_rank - pred_rank.mean()
        true_c = true_rank - true_rank.mean()
        denom = np.sqrt((pred_c**2).sum() * (true_c**2).sum())
        spearman = float((pred_c * true_c).sum() / denom) if denom > 1e-12 else 0.0
        n = len(indices)
        r2_w += float(cell["r2"]) * n
        sp_w += spearman * n
        n_tot += n
        n_cells += 1

    if n_tot == 0:
        return {
            "within_state_r2": float("nan"),
            "within_state_spearman": float("nan"),
            "within_state_n_cells": 0.0,
            "within_state_n": 0.0,
        }
    return {
        "within_state_r2": r2_w / n_tot,
        "within_state_spearman": sp_w / n_tot,
        "within_state_n_cells": float(n_cells),
        "within_state_n": float(n_tot),
    }


def _metric_prefix_for_column(target_column: str) -> str:
    if target_column == "yield_t_ha":
        return "prod"
    if target_column == "yield_t_ha_dev":
        return "prod_dev"
    if target_column == "production_t_s2_adj":
        return "total_adj"
    if target_column == "production_t":
        return "total"
    return "out"


def regression_metrics_bundle(
    y_pred,
    y_true,
    *,
    target_columns: str | list[str] | tuple[str, ...],
):
    """
    Metrics for one or more output heads.

    The first column fills rmse/mae/r2/mape for trainlog compatibility.
    Additional columns add prefixed keys (e.g. r2_total_adj).
    """
    if isinstance(target_columns, str):
        cols = [target_columns]
    else:
        cols = list(target_columns)

    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)

    if y_pred.shape[1] != len(cols):
        raise ValueError(
            f"y_pred has {y_pred.shape[1]} dims but {len(cols)} target columns"
        )

    out: dict = {}
    for i, col in enumerate(cols):
        m = regression_metrics(y_pred[:, i], y_true[:, i], target_column=col)
        if i == 0:
            out.update(m)
        prefix = _metric_prefix_for_column(col)
        for k, v in m.items():
            out[f"{k}_{prefix}"] = v
    return out


class AggregatedMSELoss(nn.Module):
    """Loss: aggregate pixel predictions per municipality, compare to target.

    When ``head_output='zscore'`` the decoder already emits municipal z-scores,
    so pixels are mean-pooled and the aggregate is not z-scored again.
    When ``head_output='raw'`` and ``normalize_targets=False``, both the
    decoder and labels stay in original units (t/ha).
    """

    def __init__(
        self,
        reduction="mean",
        target_mean=None,
        target_std=None,
        aggregation: str | list[str] | tuple[str, ...] = "sum",
        target_column: str | list[str] | tuple[str, ...] = "production_t",
        head_output: str = HEAD_OUTPUT_ZSCORE,
        normalize_targets: bool | None = None,
    ):
        super().__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction="none")
        self.target_mean = target_mean if target_mean is not None else 0.0
        self.target_std = target_std if target_std is not None else 1.0
        self.normalize = target_mean is not None and target_std is not None
        self.normalize_targets = (
            self.normalize if normalize_targets is None else bool(normalize_targets)
        )
        self.aggregation = aggregation
        self.target_column = target_column
        self.head_output = head_output

    def _norm_stats_tensors(self, ref: torch.Tensor):
        mean = self.target_mean
        std = self.target_std
        if not torch.is_tensor(mean):
            mean = torch.as_tensor(mean, dtype=ref.dtype, device=ref.device)
        else:
            mean = mean.to(device=ref.device, dtype=ref.dtype)
        if not torch.is_tensor(std):
            std = torch.as_tensor(std, dtype=ref.dtype, device=ref.device)
        else:
            std = std.to(device=ref.device, dtype=ref.dtype)
        return mean, std

    def forward(self, predictions_list, targets, num_pixels_list=None):
        """
        Args:
            predictions_list: List of tensors, one per municipality [num_pixels, num_outputs]
            targets: [batch_size] or [batch_size, num_outputs] (already normalized if normalize=True)
        """
        pool = pixel_pool_for_head(
            self.aggregation, getattr(self, "head_output", HEAD_OUTPUT_RAW)
        )
        aggregated_predictions = []
        for pred in predictions_list:
            aggregated_predictions.append(aggregate_pixels(pred, pool))

        aggregated = torch.stack(aggregated_predictions)  # [batch_size, num_outputs]

        if aggregated.dim() > 1 and aggregated.size(1) == 1:
            aggregated = aggregated.squeeze(1)
        if targets.dim() > 1 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        head_output = getattr(self, "head_output", HEAD_OUTPUT_RAW)
        labels_zscored = bool(getattr(self, "normalize_targets", self.normalize))
        if head_output != HEAD_OUTPUT_ZSCORE and labels_zscored:
            mean, std = self._norm_stats_tensors(aggregated)
            aggregated = (aggregated - mean) / std

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
    """Training epoch: mean-pool pixel z-scores per municipality, MSE vs z-scored labels."""
    from torch.amp import GradScaler

    from training.train_batch import process_train_batch
    from training_runtime import zero_grad
    from utils import AverageMeter

    losses = AverageMeter("Loss", ":.4e")
    model.train()
    aggregation = getattr(criterion, "aggregation", "sum")
    head_output = getattr(criterion, "head_output", HEAD_OUTPUT_RAW)
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
            head_output=head_output,
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
    head_output = getattr(criterion, "head_output", HEAD_OUTPUT_RAW)
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
            head_output=head_output,
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
    head_output = getattr(criterion, "head_output", HEAD_OUTPUT_RAW)
    pixel_pool = pixel_pool_for_head(aggregation, head_output)
    target_column = getattr(criterion, "target_column", "production_t")
    all_aggregated_preds = []
    all_targets = []
    all_codes: list = []
    all_years: list = []
    chunk_size = _pixel_chunk_size(args)
    batch_total = len(dataloader)
    epoch_start = time.perf_counter()

    def _to_stats_tensor(stat, ref: torch.Tensor):
        if stat is None:
            return None
        if not torch.is_tensor(stat):
            return torch.as_tensor(stat, dtype=ref.dtype, device=ref.device)
        return stat.to(device=ref.device, dtype=ref.dtype)

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
                [aggregate_pixels(pred, pixel_pool) for pred in predictions_list]
            )
            if aggregated_preds.dim() > 1 and aggregated_preds.size(1) == 1:
                aggregated_preds = aggregated_preds.squeeze(1)
            if targets.dim() > 1 and targets.size(1) == 1:
                targets = targets.squeeze(1)

            if target_mean is not None and target_std is not None:
                labels_zscored = bool(getattr(args, "normalize_targets", True))
                if head_output == HEAD_OUTPUT_ZSCORE:
                    aggregated_preds = denormalize_head_output(
                        aggregated_preds, target_mean, target_std, head_output
                    )
                    targets = denormalize_head_output(
                        targets, target_mean, target_std, head_output
                    )
                elif labels_zscored:
                    mean_t = _to_stats_tensor(target_mean, aggregated_preds)
                    std_t = _to_stats_tensor(target_std, aggregated_preds)
                    aggregated_preds_normalized = (aggregated_preds - mean_t) / std_t
                    aggregated_preds = aggregated_preds_normalized * std_t + mean_t
                    targets = targets * std_t + mean_t

            all_aggregated_preds.append(aggregated_preds.cpu().float().numpy())
            all_targets.append(targets.cpu().numpy())
            all_codes.extend(list(municipalities))
            if years is None:
                all_years.extend([None] * len(municipalities))
            else:
                all_years.extend(list(years))

        all_aggregated_preds = np.concatenate(all_aggregated_preds)
        all_targets = np.concatenate(all_targets)
        scores = regression_metrics_bundle(
            all_aggregated_preds, all_targets, target_columns=target_column
        )
        scores.update(
            within_state_year_metrics(
                all_aggregated_preds,
                all_targets,
                all_codes,
                all_years,
                min_count=int(getattr(args, "within_state_min_count", 20) or 20),
            )
        )

    return losses.avg, scores
