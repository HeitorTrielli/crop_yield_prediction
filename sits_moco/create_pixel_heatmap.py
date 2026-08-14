"""
Script to create pixel-level prediction heatmaps for municipalities.
Maps predictions back to spatial locations using original TIFF files.
"""

import argparse
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from torch.amp import autocast
from tqdm import tqdm

from datasets.datautils import getWeight
from datasets.feature_layout import (
    feature_layout_choices,
    feature_layout_input_dim,
    normalize_feature_layout,
)
from datasets.pixel_transform import DOY_CHANNEL
from datasets.uscrops_aggregated_npy_polars import (
    _indices_first_n_months,
    scale_xavier_rain_channels,
)
from models import STNetRegression
from utils import recursive_todevice
from municipality_labels import (
    municipality_output_stem,
    resolve_municipality_display_name,
)
from run_paths import (
    apply_run_config_to_args,
    intramunicipal_heatmap_dir,
    load_latest_run_config,
    resolve_checkpoint_path,
    run_dir_from_path,
)
from utils_aggregated import (
    resolve_inference_target,
    resolve_model_kwargs,
    stnet_regression_input_dim_from_state_dict,
)


def _yield_stats_text(valid_predictions, unit: str, aggregation: str) -> str:
    lines = [
        f"Min: {valid_predictions.min():.2f} {unit}",
        f"Max: {valid_predictions.max():.2f} {unit}",
        f"Mean: {valid_predictions.mean():.2f} {unit}",
    ]
    if aggregation == "sum":
        lines.append(f"Total: {valid_predictions.sum():.2f} {unit}")
    lines.append(f"Pixels: {len(valid_predictions):,}")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create pixel-level prediction heatmaps for municipalities."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model_best.pth, or run directory (uses training/model_best.pth)",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default=None,
        help=(
            "Dataset root (default: training/config.json). "
            "Override with a season folder when using --year-range."
        ),
    )
    parser.add_argument(
        "--tiffpath",
        type=str,
        default="files/daily_tiff",
        help="Path to directory containing original TIFF files (for spatial mapping). For daily mode: .../daily_tiff/ (season/municipality subfolders).",
    )
    parser.add_argument(
        "--year-range",
        type=str,
        default=None,
        help="Season folder name (e.g. 2020-2021). Required unless --tiffpath already points to {muni}/{year-range}.",
    )
    parser.add_argument(
        "--municipality-code",
        type=str,
        nargs="+",
        required=True,
        help="Municipality code(s) to create heatmap for (e.g., '4100103' or '4100103 4109401' for multiple)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory for heatmap images (default: "
            "{run_dir}/figures/intramunicipal_heatmap/ from --checkpoint)"
        ),
    )
    parser.add_argument(
        "--bias-correct",
        action="store_true",
        help=(
            "Post-hoc: shift pixel predictions so the municipal mean matches PAM "
            "yield_t_ha for --year / --year-range (does not change the model). "
            "Useful for downscaled maps; municipal R² is not a fair metric after this."
        ),
    )
    parser.add_argument(
        "--bias-correct-mode",
        type=str,
        default="shift",
        choices=("shift", "scale"),
        help="Bias correction mode when --bias-correct is set (default: shift)",
    )
    parser.add_argument(
        "--yield-csv",
        type=str,
        default="files/pam_soy_pr_2019_2025_coverage_0.8_1.2.csv",
        help="PAM CSV used for --bias-correct municipal mean lookup",
    )
    parser.add_argument(
        "--figures-subdir",
        type=str,
        default=None,
        help=(
            "Optional tag subfolder under figures/intramunicipal_heatmap/ "
            "(e.g. 2020-2021 for harvest season)"
        ),
    )
    parser.add_argument(
        "-seq",
        "--sequencelength",
        type=int,
        default=None,
        help="Max time steps (default: training/config.json)",
    )
    parser.add_argument(
        "--rc",
        action="store_true",
        help="Whether to use random choice for time series data",
    )
    parser.add_argument(
        "--interp",
        action="store_true",
        help="Whether to interpolate the time series data",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Number of pixels to process at once (default: 2000)",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        help='torch.Device. either "cpu" or "cuda". default will check by torch.cuda.is_available()',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: training/config.json)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        help="Year of the data (default: 2023)",
    )
    parser.add_argument(
        "--reference-date",
        type=str,
        default=None,
        help="Date where DOY 1 falls for daily mode, YYYY-MM-DD (e.g. 2020-10-01). If omitted, inferred as Oct 1 of the first year in --year-range.",
    )
    parser.add_argument(
        "--daily-ndvi",
        action="store_true",
        help="Also save one NDVI heatmap PNG per observation date (from daily TIFFs).",
    )
    parser.add_argument(
        "--daily-ndvi-subdir",
        type=str,
        default="daily_ndvi",
        help="Subfolder under --output-dir for per-day NDVI PNGs (default: daily_ndvi).",
    )
    parser.add_argument(
        "--feature-layout",
        type=str,
        default=None,
        metavar="NAME",
        help="Override feature layout from training/config.json",
    )
    parser.add_argument(
        "--report-panel",
        action="store_true",
        help=(
            "Write one 2×2 PNG (yield sem/com + NDVI mean/peak) for the report. "
            "Requires exactly one --municipality-code and --compare-checkpoint."
        ),
    )
    parser.add_argument(
        "--compare-checkpoint",
        type=str,
        default=None,
        help="Second model checkpoint (e.g. com chuva) for --report-panel.",
    )
    parser.add_argument(
        "--panel-output",
        type=str,
        default=None,
        help="Output path for --report-panel (default: {output-dir}/{muni_stem}_panel.png).",
    )
    parser.add_argument(
        "--num-periods",
        type=int,
        nargs="+",
        metavar="K",
        help=(
            "Use only the first K season months (1–6), as in evaluate_incomplete_series.py. "
            "Pass one or more values, e.g. --num-periods 1 3 6. Saves yield heatmaps only."
        ),
    )
    parser.add_argument(
        "--incomplete-series",
        action="store_true",
        help="Shortcut for --num-periods 1 2 3 4 5 6 (one yield heatmap per K).",
    )
    parser.add_argument(
        "--panel-dpi",
        type=int,
        default=160,
        help="DPI for --report-panel (default: 160).",
    )
    args = parser.parse_args()

    args.checkpoint = resolve_checkpoint_path(args.checkpoint)
    run_config = load_latest_run_config(args.checkpoint)
    apply_run_config_to_args(args, run_config)
    if args.feature_layout is not None:
        normalize_feature_layout(args.feature_layout)

    args.datapath = Path(args.datapath)
    args.tiffpath = Path(args.tiffpath)
    if args.compare_checkpoint is not None:
        args.compare_checkpoint = resolve_checkpoint_path(args.compare_checkpoint)
    if args.output_dir is not None:
        args.output_dir = Path(args.output_dir)
    if args.panel_output is not None:
        args.panel_output = Path(args.panel_output)
    if args.reference_date is not None:
        args.reference_date = datetime.strptime(args.reference_date, "%Y-%m-%d").date()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args, run_config


def parse_filename(filename):
    """Parse TIFF filename: 4100103_2023-01_tile_00_01.tif"""
    stem = Path(filename).stem
    parts = stem.split("_")
    municipality_code = parts[0]
    year_month = parts[1]
    year, month = year_month.split("-")
    tile_x = parts[3]
    tile_y = parts[4]
    return municipality_code, int(year), int(month), tile_x, tile_y


def parse_daily_filename(filename: str):
    """Parse daily TIFF filename: 4109401_2020_10_04.tiff -> (code, date(2020,10,4))."""
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 4:
        return None, None
    code = parts[0]
    try:
        y, m, d = int(parts[1]), int(parts[2]), int(parts[3])
        return code, date(y, m, d)
    except Exception:
        return None, None


def season_start_from_year_range(year_range: str) -> date:
    """Return Oct 1 of the first year in a 'YYYY-YYYY' season folder."""
    parts = (year_range or "").split("-")
    if len(parts) >= 2 and parts[0].isdigit():
        return date(int(parts[0]), 10, 1)
    raise ValueError(f"Could not parse --year-range={year_range!r} (expected like 2020-2021)")


def harvest_year_from_year_range(year_range: str | None) -> int | None:
    """Calendar year of harvest (second year in a 'YYYY-YYYY' season folder)."""
    parts = (year_range or "").strip().split("-")
    if len(parts) >= 2 and parts[1].isdigit():
        return int(parts[1])
    if parts and parts[0].isdigit():
        return int(parts[0])
    return None


def date_to_season_doy(d: date, season_start: date) -> int:
    """1-based day index from season_start."""
    return (d - season_start).days + 1


def doy_to_season_month(doy: int, reference_date: date) -> int:
    """Map season DOY (1-based, DOY1=reference_date) to season month index 1..6, else 0."""
    d = reference_date.toordinal() + int(doy) - 1
    cal_month = date.fromordinal(d).month
    ref_month = reference_date.month
    season_cal_months = [(ref_month - 1 + i) % 12 + 1 for i in range(6)]
    if cal_month not in season_cal_months:
        return 0
    return season_cal_months.index(cal_month) + 1


def ndvi_mean_timestep_mask(
    doys: np.ndarray,
    valid: np.ndarray,
    *,
    season_start: date,
    harvest_year: int,
    skip_season_months: int = 3,
) -> np.ndarray:
    """Timesteps used for mean NDVI: after the first *skip* season months, in *harvest_year*."""
    mask = np.zeros(len(doys), dtype=bool)
    for i in np.flatnonzero(valid):
        doy = int(doys[i])
        if doy_to_season_month(doy, season_start) <= skip_season_months:
            continue
        if (season_start + timedelta(days=doy - 1)).year != harvest_year:
            continue
        mask[i] = True
    return mask


def compute_daily_ndvi_stats_from_npy_row(
    X_row: np.ndarray,
    # Keep literal defaults here because this function is defined before RED_BAND_IDX/NIR_BAND_IDX
    red_idx: int = 2,
    nir_idx: int = 6,
    *,
    season_start: date | None = None,
    harvest_year: int | None = None,
    skip_season_months: int = 3,
):
    """Compute mean/peak (max) NDVI over valid daily timesteps from one .npy pixel row (T, 11).

    Uses raw reflectance (bands * 1e-4). Treats 0 and -9999 as missing.
    Mean NDVI optionally restricts to the harvest calendar year after skipping the first
    *skip_season_months* season months (default 3: Oct--Dec). Peak is max NDVI on all valid days.
    Returns (mean_ndvi, peak_ndvi, ndvi_per_season_month_dict).
    """
    if X_row.ndim != 2 or X_row.shape[1] < 11:
        return np.nan, np.nan, {}

    bands = X_row[:, :10].astype(np.float32)
    doys = X_row[:, 10].astype(np.int32)

    # Missing markers from preprocess_daily_to_npy: 0 or -9999
    missing = (bands == 0) | (bands == -9999)
    red = bands[:, red_idx]
    nir = bands[:, nir_idx]
    miss_red = missing[:, red_idx]
    miss_nir = missing[:, nir_idx]

    red = red * 1e-4
    nir = nir * 1e-4

    denom = nir + red + 1e-8
    valid = np.isfinite(red) & np.isfinite(nir) & ~miss_red & ~miss_nir & (np.abs(denom) > 1e-8)
    if not np.any(valid):
        return np.nan, np.nan, {}

    ndvi = (nir - red) / denom
    ndvi = np.clip(ndvi, -1.0, 1.0)
    peak_ndvi = float(np.nanmax(ndvi[valid])) if np.any(valid) else np.nan

    mean_mask = valid
    if season_start is not None and harvest_year is not None:
        mean_mask = ndvi_mean_timestep_mask(
            doys,
            valid,
            season_start=season_start,
            harvest_year=harvest_year,
            skip_season_months=skip_season_months,
        )
    elif season_start is not None and skip_season_months > 0:
        mean_mask = valid & np.array(
            [
                doy_to_season_month(int(d), season_start) > skip_season_months
                for d in doys
            ],
            dtype=bool,
        )

    mean_ndvi = float(np.nanmean(ndvi[mean_mask])) if np.any(mean_mask) else np.nan

    # Per-season-month mean NDVI
    per_month = {}
    for m in range(1, 7):
        per_month[m] = np.nan
    return mean_ndvi, peak_ndvi, per_month


def resolve_muni_npy(
    datapath: Path, municipality_code: str, year_range: str | None = None
) -> Path | None:
    """Find municipality .npy under datapath (flat or {year-range}/{muni}/ layout)."""
    candidates = [datapath / municipality_code / f"{municipality_code}.npy"]
    if year_range is not None:
        candidates.insert(
            0, datapath / year_range / municipality_code / f"{municipality_code}.npy"
        )
    candidates.append(datapath / f"{municipality_code}.npy")
    for p in candidates:
        if p.is_file():
            return p
    return None


def resolve_daily_tiff_dir(tiffpath: Path, municipality_code: str, year_range: str | None) -> Path | None:
    """Return directory containing daily TIFFs for a municipality/season."""
    if tiffpath.is_dir():
        # If user already points directly to .../<muni>/<year-range>
        if year_range is not None and tiffpath.name == year_range and tiffpath.parent.name == municipality_code:
            return tiffpath
        # Canonical layout: .../<year-range>/<muni>/
        if year_range is not None:
            candidate = tiffpath / year_range / municipality_code
            if candidate.is_dir():
                return candidate
        # Fallback: .../<muni>/ (no season in path)
        candidate = tiffpath / municipality_code
        if candidate.is_dir():
            return candidate
    return None


def filter_timeseries_for_periods(x, num_periods, reference_date):
    """Keep daily rows in season months 1..num_periods (sorted by DOY). None = full series."""
    if num_periods is None:
        return np.asarray(x, dtype=np.float32)
    if reference_date is None:
        raise ValueError("reference_date is required when num_periods is set")
    if isinstance(reference_date, str):
        reference_date = datetime.strptime(reference_date, "%Y-%m-%d").date()
    raw = np.asarray(x, dtype=np.float32)
    idx = _indices_first_n_months(raw, num_periods, reference_date)
    if len(idx) == 0:
        return None
    sub = raw[idx]
    doys = sub[:, DOY_CHANNEL].astype(np.float64)
    return sub[np.argsort(doys, kind="stable")]


def transform_pixel(
    x,
    sequencelength,
    rc,
    interp,
    seed=None,
    input_dim: int = 10,
    deterministic_head=False,
):
    """Transform pixel data: normalize, pad/sample to sequencelength, extract DOY.
    If deterministic_head is True and x has more than sequencelength rows, keeps the earliest sequencelength rows.
    ``input_dim`` must match the trained model (10 or 12 with Xavier).
    """
    if seed is not None:
        np.random.seed(seed)

    raw = np.asarray(x, dtype=np.float32)
    t_len, c_in = raw.shape
    doy_col = 10 if c_in >= 11 else c_in - 1
    doy = raw[:, doy_col].astype(np.int32)
    x_spec = raw[:, :10] * 1e-4

    mean = np.array(
        [[0.147, 0.169, 0.186, 0.221, 0.273, 0.297, 0.308, 0.316, 0.256, 0.188]]
    )
    std = np.array([0.227, 0.219, 0.222, 0.22, 0.2, 0.193, 0.192, 0.182, 0.123, 0.106])

    weight = getWeight(x_spec)
    x_spec_n = (x_spec - mean) / std
    if input_dim == 12:
        if c_in >= 13:
            rain = scale_xavier_rain_channels(raw[:, 11:13])
        else:
            rain = np.zeros((t_len, 2), dtype=np.float32)
        x = np.concatenate([x_spec_n, rain], axis=-1)
    else:
        x = x_spec_n

    if interp:
        doy_pad = np.linspace(0, 366, sequencelength).astype("int")
        fdim = x.shape[1]
        x_pad = np.array([np.interp(doy_pad, doy, x[:, i]) for i in range(fdim)]).T
        spec_denorm = x_pad[:, :10] * std + mean
        weight_pad = getWeight(spec_denorm)
        mask = np.ones((sequencelength,), dtype=int)
    elif rc:
        replace = False if x.shape[0] >= sequencelength else True
        idxs = np.random.choice(x.shape[0], sequencelength, replace=replace)
        idxs.sort()
        x_pad = x[idxs]
        mask = np.ones((sequencelength,), dtype=int)
        doy_pad = doy[idxs]
        weight_pad = weight[idxs]
        weight_pad /= weight_pad.sum()
    else:
        x_length, c_length = x.shape
        if x_length == sequencelength:
            mask = np.ones((sequencelength,), dtype=int)
            x_pad = x
            doy_pad = doy
            weight_pad = weight
            weight_pad /= weight_pad.sum()
        elif x_length < sequencelength:
            mask = np.zeros((sequencelength,), dtype=int)
            mask[:x_length] = 1
            x_pad = np.zeros((sequencelength, c_length))
            x_pad[:x_length, :] = x[:x_length, :]
            doy_pad = np.zeros((sequencelength,), dtype=int)
            doy_pad[:x_length] = doy[:x_length]
            weight_pad = np.zeros((sequencelength,), dtype=float)
            weight_pad[:x_length] = weight[:x_length]
            weight_pad /= weight_pad.sum() if weight_pad.sum() > 0 else 1
        else:
            if deterministic_head:
                x_pad = x[:sequencelength]
                mask = np.ones((sequencelength,), dtype=int)
                doy_pad = doy[:sequencelength]
                weight_pad = weight[:sequencelength]
                weight_pad /= weight_pad.sum() if weight_pad.sum() > 0 else 1
            else:
                idxs = np.random.choice(x.shape[0], sequencelength, replace=False)
                idxs.sort()
                x_pad = x[idxs]
                mask = np.ones((sequencelength,), dtype=int)
                doy_pad = doy[idxs]
                weight_pad = weight[idxs]
                weight_pad /= weight_pad.sum()

    return (
        torch.from_numpy(x_pad).type(torch.FloatTensor),
        torch.from_numpy(mask == 0),
        torch.from_numpy(doy_pad).type(torch.LongTensor),
        torch.from_numpy(weight_pad).type(torch.FloatTensor),
    )


def get_municipality_name(tiff_dir, municipality_code):
    """Human-readable municipality name for plots (yield CSV, then TIFF folder)."""
    return resolve_municipality_display_name(
        municipality_code, tiff_root=tiff_dir
    )


def compute_daily_valid_pixel_mask(daily_tiffs: list[Path]):
    """Match preprocess_daily_to_npy pixel filtering on the original daily TIFFs.

    Returns:
        keep_mask: (H, W) bool, True for pixels kept in the .npy (row-major order)
        height, width
    """
    if not daily_tiffs:
        return None, None, None

    # Determine dimensions and nodata from first file
    with rasterio.open(daily_tiffs[0]) as src0:
        height, width = src0.height, src0.width

    any_data = np.zeros((height, width), dtype=bool)
    days_with_data = np.zeros((height, width), dtype=np.uint16)

    for p in daily_tiffs:
        try:
            with rasterio.open(p) as src:
                nodata = src.nodata if src.nodata is not None else -9999
                data = src.read()[:10].astype(np.float32)  # [10, H, W]
        except Exception:
            # Treat missing/unreadable day as no data
            continue

        # Same cleaning logic as preprocess: nodata -> NaN
        if nodata is not None:
            data[data == nodata] = np.nan
        data[data == -9999] = np.nan

        # A pixel has data on this day if any band is finite and non-zero
        day_has = np.any(np.isfinite(data) & (data != 0), axis=0)
        any_data |= day_has
        days_with_data += day_has.astype(np.uint16)

    keep_mask = any_data & (days_with_data > 2)
    return keep_mask, height, width


# Band indices for NDVI (from preprocess_tiff_to_npy BANDNAMES: red=2, nir=6)
RED_BAND_IDX = 2
NIR_BAND_IDX = 6


def save_daily_ndvi_heatmaps_from_tiffs(
    municipality_code,
    daily_tiffs_sorted,
    output_parent,
    height,
    width,
    subdir_name="daily_ndvi",
):
    """Save one beige→green NDVI heatmap per observation date using daily TIFF rasters."""
    ndvi_root = output_parent / subdir_name / municipality_code
    ndvi_root.mkdir(parents=True, exist_ok=True)

    dated_paths = []
    for p in daily_tiffs_sorted:
        code, d = parse_daily_filename(p.name)
        if code == municipality_code and d is not None:
            dated_paths.append((d, p))

    for date_key, path in tqdm(
        dated_paths,
        total=len(dated_paths),
        desc="  Daily NDVI",
        unit="day",
    ):
        if date_key is None:
            continue
        fname = f"{municipality_code}_ndvi_{date_key.strftime('%Y-%m-%d')}.png"
        out_path = ndvi_root / fname

        try:
            with rasterio.open(path) as src:
                nodata = src.nodata if src.nodata is not None else -9999
                data = src.read()[:10].astype(np.float32)
        except Exception as e:
            print(f"  ⚠️  Skip daily NDVI {path.name}: {e}")
            continue

        if data.shape[1] != height or data.shape[2] != width:
            print(
                f"  ⚠️  Skip daily NDVI {path.name}: shape {data.shape} != {(10, height, width)}"
            )
            continue

        if nodata is not None:
            data[data == nodata] = np.nan
        data[data == -9999] = np.nan

        red = data[RED_BAND_IDX] * 1e-4
        nir = data[NIR_BAND_IDX] * 1e-4
        denom = nir + red + 1e-8
        ndvi_map = np.full((height, width), np.nan, dtype=np.float32)
        valid = np.isfinite(red) & np.isfinite(nir) & (np.abs(denom) > 1e-8)
        ndvi_map[valid] = np.clip((nir[valid] - red[valid]) / denom[valid], -1.0, 1.0)

        pred_like = {}
        for r in range(height):
            for c in range(width):
                pred_like[("grid", r, c)] = float(ndvi_map[r, c])

        create_ndvi_heatmap(
            pred_like,
            None,
            height,
            width,
            out_path,
            municipality_code,
            tiffpath=None,
            year=None,
            title_suffix=f"{date_key.strftime('%Y-%m-%d')}",
        )


def compute_ndvi_from_timeseries(
    timeseries, red_idx=RED_BAND_IDX, nir_idx=NIR_BAND_IDX
):
    """
    Compute mean NDVI from pixel timeseries [num_months, num_bands].
    NDVI = (NIR - Red) / (NIR + Red + eps). Returns scalar in [-1, 1] or np.nan if invalid.
    """
    red = timeseries[:, red_idx].astype(np.float64)
    nir = timeseries[:, nir_idx].astype(np.float64)
    denom = nir + red + 1e-8
    ndvi = (nir - red) / denom
    ndvi = np.clip(ndvi, -1.0, 1.0)
    valid = np.isfinite(ndvi) & (np.abs(denom) > 1e-8)
    if np.any(valid):
        return float(np.nanmean(ndvi[valid]))
    return np.nan


def compute_ndvi_single_band(bands, red_idx=RED_BAND_IDX, nir_idx=NIR_BAND_IDX):
    """
    Compute NDVI for a single time step (one row: 10 bands).
    Returns scalar in [-1, 1] or np.nan if invalid.
    """
    red = float(bands[red_idx])
    nir = float(bands[nir_idx])
    denom = nir + red + 1e-8
    if not np.isfinite(denom) or np.abs(denom) < 1e-8:
        return np.nan
    ndvi = (nir - red) / denom
    return float(np.clip(ndvi, -1.0, 1.0))


def should_include_pixel(timeseries):
    """
    Apply the same filtering logic as preprocessing to determine if a pixel should be included.
    Returns True if pixel should be in .npy file, False otherwise.
    """
    # Filter out pixels that are always zero
    if np.all(timeseries == 0):
        return False

    # Filter out pixels that have data in 2 months or fewer
    months_with_data = np.sum(np.any(timeseries != 0, axis=1))
    if months_with_data <= 2:
        return False

    return True


def reconstruct_spatial_predictions(
    model,
    municipality_code,
    datapath,
    tiffpath,
    year,
    sequencelength,
    rc,
    interp,
    chunk_size,
    device,
    seed=None,
    year_range=None,
    reference_date=None,
    input_dim: int = 10,
    num_periods=None,
):
    """
    Reconstruct spatial layout of predictions by processing pixels in same order as preprocessing.
    Uses TIFF files to get exact spatial structure and matches pixels to .npy data.
    Returns: prediction_map dict with (tile_x, tile_y, row, col) -> prediction, or None if mapping fails
    """
    if year_range is None:
        print("  ⚠️  Warning: --year-range is required (unless --tiffpath points directly to {muni}/{year-range})")
        return None, None, None, None, None, None, None

    daily_dir = resolve_daily_tiff_dir(tiffpath, municipality_code, year_range)
    if daily_dir is None:
        print(f"  ⚠️  Warning: Could not resolve daily TIFF dir from {tiffpath} for {municipality_code}/{year_range}")
        return None, None, None, None, None, None, None

    # Collect daily TIFFs and sort by date
    dated = []
    for p in sorted(daily_dir.glob("*.tif*")):
        code, d = parse_daily_filename(p.name)
        if code == municipality_code and d is not None:
            dated.append((d, p))
    dated.sort(key=lambda x: x[0])
    daily_tiffs = [p for _, p in dated]
    if not daily_tiffs:
        print(f"  ⚠️  Warning: No daily TIFF files found in {daily_dir}")
        return None, None, None, None, None, None, None

    if reference_date is None:
        season_start = season_start_from_year_range(year_range)
    else:
        season_start = reference_date
    harvest_year = harvest_year_from_year_range(year_range)

    muni_npy_file = resolve_muni_npy(datapath, municipality_code, year_range)
    if muni_npy_file is None:
        print(
            f"  ⚠️  Warning: Could not find {municipality_code}.npy under {datapath}"
            + (f" (tried year-range {year_range!r})" if year_range else "")
        )
        return None, None, None, None, None, None, None

    try:
        municipality_data = np.load(muni_npy_file, mmap_mode="r")
    except Exception as e:
        print(f"  ⚠️  Warning: Could not load {muni_npy_file}: {e}")
        return None, None, None, None, None, None, None

    if len(municipality_data) == 0:
        print(f"  ⚠️  Warning: {municipality_code} has 0 pixels")
        return None, None, None, None, None, None, None

    print(f"  Loaded {muni_npy_file} shape={municipality_data.shape}")
    if input_dim == 12 and municipality_data.shape[-1] < 13:
        print(
            "  ⚠️  Warning: checkpoint uses spectral_xavier (12 inputs) but .npy has "
            f"{municipality_data.shape[-1]} channels per timestep. "
            "Run data_download/xavier_rain_for_daily_npy.py on this season's .npy before heatmapping."
        )

    # Recompute keep-mask in the exact same way as preprocess_daily_to_npy
    keep_mask, height, width = compute_daily_valid_pixel_mask(daily_tiffs)
    if keep_mask is None:
        return None, None, None, None, None, None, None

    expected_valid = int(keep_mask.sum())
    if expected_valid != len(municipality_data):
        print(
            f"  ⚠️  Warning: keep_mask pixels ({expected_valid}) != .npy pixels ({len(municipality_data)}). "
            "Heatmap will still be produced, but mapping may be off if inputs differ (nodata handling, reprojection, etc.)."
        )

    model.eval()
    prediction_map = {}
    ndvi_map = {}
    ndvi_map_median = {}
    ndvi_map_per_month = defaultdict(dict)
    npy_pixel_idx = 0

    with torch.no_grad():
        current_chunk = []
        chunk_indices = []

        # Iterate pixels in row-major order (same as preprocess_daily_to_npy reshape)
        for row in tqdm(range(height), desc="  Rows", unit="row"):
            for col in range(width):
                if keep_mask[row, col]:
                    if npy_pixel_idx >= len(municipality_data):
                        prediction_map[("grid", row, col)] = np.nan
                        continue
                    X = municipality_data[npy_pixel_idx]
                    # Daily NDVI stats computed from raw (unnormalized) .npy row
                    try:
                        mean_ndvi, peak_ndvi, _ = compute_daily_ndvi_stats_from_npy_row(
                            X,
                            season_start=season_start,
                            harvest_year=harvest_year,
                        )
                    except Exception:
                        mean_ndvi, peak_ndvi = np.nan, np.nan
                    ndvi_map[("grid", row, col)] = mean_ndvi
                    # ndvi_map_median slot holds peak (max) NDVI over time.
                    ndvi_map_median[("grid", row, col)] = peak_ndvi

                    X_filtered = filter_timeseries_for_periods(
                        X, num_periods, season_start
                    )
                    if X_filtered is None:
                        prediction_map[("grid", row, col)] = np.nan
                        npy_pixel_idx += 1
                        continue

                    X_tuple = transform_pixel(
                        X_filtered,
                        sequencelength,
                        rc,
                        interp,
                        seed=seed,
                        input_dim=input_dim,
                        deterministic_head=True,
                    )
                    current_chunk.append(X_tuple)
                    chunk_indices.append(("grid", row, col))
                    npy_pixel_idx += 1
                else:
                    prediction_map[("grid", row, col)] = np.nan
                    ndvi_map[("grid", row, col)] = np.nan
                    ndvi_map_median[("grid", row, col)] = np.nan

                is_last = (row == height - 1 and col == width - 1)
                if len(current_chunk) >= chunk_size or (len(current_chunk) >= 2 and is_last):
                    chunk_x = torch.stack([p[0] for p in current_chunk])
                    chunk_mask = torch.stack([p[1] for p in current_chunk])
                    chunk_doy = torch.stack([p[2] for p in current_chunk])
                    chunk_weight = torch.stack([p[3] for p in current_chunk])

                    municipality_X_chunk = (chunk_x, chunk_mask, chunk_doy, chunk_weight)
                    municipality_X_chunk = recursive_todevice(municipality_X_chunk, device)

                    with (
                        autocast("cuda", dtype=torch.bfloat16)
                        if device.type == "cuda"
                        else torch.no_grad()
                    ):
                        chunk_predictions = model(municipality_X_chunk)
                        if chunk_predictions.dim() == 1:
                            chunk_predictions = chunk_predictions.unsqueeze(0)

                    for i, key in enumerate(chunk_indices):
                        prediction_map[key] = float(chunk_predictions[i].item())

                    current_chunk = []
                    chunk_indices = []

    # In daily mode: ndvi_map = mean NDVI; ndvi_map_median slot holds peak (max) NDVI over time.
    return prediction_map, ndvi_map, ndvi_map_median, dict(ndvi_map_per_month), None, height, width

def create_heatmap_array(
    prediction_map,
    tiles,
    tile_height,
    tile_width,
    municipality_code,
    tiffpath=None,
    year=None,
):
    """Create heatmap array from spatial predictions (without visualization)."""
    if not prediction_map:
        print("  ⚠️  Warning: No predictions to visualize")
        return None

    return _create_heatmap_array_internal(
        prediction_map,
        tiles,
        tile_height,
        tile_width,
        municipality_code,
        tiffpath,
        year,
    )


def _create_heatmap_array_internal(
    prediction_map,
    tiles,
    tile_height,
    tile_width,
    municipality_code,
    tiffpath=None,
    year=None,
):
    """Internal function to create heatmap array."""

    # Check if using grid layout (no spatial info)
    is_grid = any(k[0] == "grid" for k in prediction_map.keys())

    if is_grid:
        # Sequential grid layout
        max_row = max(r for _, r, c in prediction_map.keys())
        max_col = max(c for _, r, c in prediction_map.keys())
        heatmap = np.full((max_row + 1, max_col + 1), np.nan)

        for (_, row, col), pred in prediction_map.items():
            if 0 <= row <= max_row and 0 <= col <= max_col:
                heatmap[row, col] = pred
    else:
        # Spatial tile layout - use geographic coordinates from TIFF files
        sorted_tiles = sorted([k for k in tiles.keys() if k != "grid"])
        if not sorted_tiles:
            return

        # Get geographic bounds for each tile
        tile_bounds = {}  # {(tile_x, tile_y): (min_x, min_y, max_x, max_y)}
        tile_dims = {}  # {(tile_x, tile_y): (height, width)}

        if tiffpath and year:
            # Find municipality directory
            muni_dir = None
            for dir_path in tiffpath.iterdir():
                if dir_path.is_dir() and dir_path.name.startswith(
                    municipality_code + "_"
                ):
                    muni_dir = dir_path
                    break

            if muni_dir is None:
                muni_dir = tiffpath / municipality_code

            if muni_dir.exists():
                # Get bounds from TIFF files
                for tile_x, tile_y in sorted_tiles:
                    # Find a TIFF file for this tile
                    tiff_files = list(muni_dir.glob(f"*_tile_{tile_x}_{tile_y}.tif"))
                    if tiff_files:
                        try:
                            with rasterio.open(tiff_files[0]) as src:
                                bounds = src.bounds
                                tile_bounds[(tile_x, tile_y)] = (
                                    bounds.left,
                                    bounds.bottom,
                                    bounds.right,
                                    bounds.top,
                                )
                                tile_dims[(tile_x, tile_y)] = (src.height, src.width)
                        except Exception as e:
                            print(
                                f"  ⚠️  Warning: Could not get bounds for tile ({tile_x}, {tile_y}): {e}"
                            )
                            # Fallback to index-based positioning
                            tile_bounds = None
                            break

        if tile_bounds and len(tile_bounds) == len(sorted_tiles):
            # Use geographic coordinates with rasterio transforms for exact positioning
            # Store transforms for each tile
            tile_transforms = {}  # {(tile_x, tile_y): Affine transform}

            if tiffpath and year and muni_dir.exists():
                for tile_x, tile_y in sorted_tiles:
                    tiff_files = list(muni_dir.glob(f"*_tile_{tile_x}_{tile_y}.tif"))
                    if tiff_files:
                        try:
                            with rasterio.open(tiff_files[0]) as src:
                                tile_transforms[(tile_x, tile_y)] = src.transform
                        except Exception:
                            pass

            if len(tile_transforms) == len(sorted_tiles):
                # Use transforms to get exact pixel coordinates
                all_min_x = min(b[0] for b in tile_bounds.values())
                all_min_y = min(b[1] for b in tile_bounds.values())
                all_max_x = max(b[2] for b in tile_bounds.values())
                all_max_y = max(b[3] for b in tile_bounds.values())

                # Use the most common pixel size (from transform) to avoid gaps
                # Get pixel sizes from all tiles and use the median
                pixel_sizes_x = []
                pixel_sizes_y = []
                for tile_x, tile_y in sorted_tiles:
                    transform = tile_transforms[(tile_x, tile_y)]
                    pixel_sizes_x.append(abs(transform[0]))  # pixel width
                    pixel_sizes_y.append(
                        abs(transform[4])
                    )  # pixel height (usually negative)

                pixel_size_x = np.median(pixel_sizes_x)
                pixel_size_y = np.median(pixel_sizes_y)

                # Create heatmap based on geographic extent
                total_width = int(np.ceil((all_max_x - all_min_x) / pixel_size_x))
                total_height = int(np.ceil((all_max_y - all_min_y) / pixel_size_y))
                heatmap = np.full((total_height, total_width), np.nan)

                # Fill heatmap using exact transform coordinates
                for (tx, ty, row, col), pred in prediction_map.items():
                    if tx == "grid":
                        continue
                    if (tx, ty) not in tile_transforms:
                        continue

                    transform = tile_transforms[(tx, ty)]

                    # Use transform to get exact pixel center coordinates
                    # Transform (row, col) to (x, y) coordinates
                    pixel_x, pixel_y = transform * (col + 0.5, row + 0.5)

                    # Convert to heatmap indices
                    global_col = int((pixel_x - all_min_x) / pixel_size_x)
                    global_row = int((all_max_y - pixel_y) / pixel_size_y)

                    if 0 <= global_row < total_height and 0 <= global_col < total_width:
                        heatmap[global_row, global_col] = pred
            else:
                # Fallback: use bounds-based approach with median pixel size
                all_min_x = min(b[0] for b in tile_bounds.values())
                all_min_y = min(b[1] for b in tile_bounds.values())
                all_max_x = max(b[2] for b in tile_bounds.values())
                all_max_y = max(b[3] for b in tile_bounds.values())

                # Calculate pixel sizes from all tiles and use median
                pixel_sizes_x = []
                pixel_sizes_y = []
                for tile_x, tile_y in sorted_tiles:
                    bounds = tile_bounds[(tile_x, tile_y)]
                    tile_h, tile_w = tile_dims[(tile_x, tile_y)]
                    pixel_sizes_x.append((bounds[2] - bounds[0]) / tile_w)
                    pixel_sizes_y.append((bounds[3] - bounds[1]) / tile_h)

                pixel_size_x = np.median(pixel_sizes_x)
                pixel_size_y = np.median(pixel_sizes_y)

                # Create heatmap based on geographic extent
                total_width = int(np.ceil((all_max_x - all_min_x) / pixel_size_x))
                total_height = int(np.ceil((all_max_y - all_min_y) / pixel_size_y))
                heatmap = np.full((total_height, total_width), np.nan)

                # Fill heatmap using geographic coordinates
                for (tx, ty, row, col), pred in prediction_map.items():
                    if tx == "grid":
                        continue
                    if (tx, ty) not in tile_bounds:
                        continue

                    bounds = tile_bounds[(tx, ty)]
                    tile_h, tile_w = tile_dims[(tx, ty)]

                    # Use tile-specific pixel size for more accuracy
                    tile_pixel_size_x = (bounds[2] - bounds[0]) / tile_w
                    tile_pixel_size_y = (bounds[3] - bounds[1]) / tile_h

                    # Calculate global position based on geographic coordinates
                    # Get pixel's geographic position (center of pixel)
                    pixel_x = bounds[0] + (col + 0.5) * tile_pixel_size_x
                    pixel_y = (
                        bounds[3] - (row + 0.5) * tile_pixel_size_y
                    )  # Y is top-to-bottom

                    # Convert to heatmap indices using median pixel size
                    global_col = int((pixel_x - all_min_x) / pixel_size_x)
                    global_row = int((all_max_y - pixel_y) / pixel_size_y)

                    if 0 <= global_row < total_height and 0 <= global_col < total_width:
                        heatmap[global_row, global_col] = pred
        else:
            # Fallback to index-based positioning
            tile_x_coords = sorted(set(int(tx) for tx, _ in sorted_tiles))
            tile_y_coords = sorted(set(int(ty) for _, ty in sorted_tiles))

            # Create overall grid
            total_height = len(tile_y_coords) * tile_height
            total_width = len(tile_x_coords) * tile_width
            heatmap = np.full((total_height, total_width), np.nan)

            # Map tile coordinates to grid positions
            tile_x_to_idx = {tx: i for i, tx in enumerate(tile_x_coords)}
            tile_y_to_idx = {ty: i for i, ty in enumerate(tile_y_coords)}

            # Fill heatmap
            for (tx, ty, row, col), pred in prediction_map.items():
                if tx == "grid":
                    continue
                tx_idx = tile_x_to_idx.get(int(tx))
                ty_idx = tile_y_to_idx.get(int(ty))
                if tx_idx is not None and ty_idx is not None:
                    global_row = ty_idx * tile_height + row
                    global_col = tx_idx * tile_width + col
                    if 0 <= global_row < total_height and 0 <= global_col < total_width:
                        heatmap[global_row, global_col] = pred

    return heatmap


def create_heatmap(
    prediction_map,
    tiles,
    tile_height,
    tile_width,
    output_path,
    municipality_code,
    tiffpath=None,
    year=None,
    title=None,
    *,
    target_unit: str,
    aggregation: str,
):
    """Create heatmap visualization from spatial predictions using geographic coordinates."""
    if not prediction_map:
        print("  ⚠️  Warning: No predictions to visualize")
        return None

    # Get heatmap array
    heatmap = _create_heatmap_array_internal(
        prediction_map,
        tiles,
        tile_height,
        tile_width,
        municipality_code,
        tiffpath,
        year,
    )

    if heatmap is None:
        return None

    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use a smooth sequential colormap (low=beige, high=green)
    cmap = LinearSegmentedColormap.from_list("beige_green", ["#f4f1e6", "#2e7d32"])
    cmap.set_bad(color="lightgray", alpha=0.3)  # Color for NaN (no data)

    valid_predictions = heatmap[np.isfinite(heatmap)]
    if len(valid_predictions) > 0:
        vmin = float(np.nanpercentile(valid_predictions, 2))
        vmax = float(np.nanpercentile(valid_predictions, 98))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin = float(np.nanmin(valid_predictions))
            vmax = float(np.nanmax(valid_predictions))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = 0.0, 1.0

    im = ax.imshow(
        heatmap, cmap=cmap, interpolation="nearest", aspect="auto", vmin=vmin, vmax=vmax
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"Yield Prediction ({target_unit})", rotation=270, labelpad=20)

    municipality_name = get_municipality_name(tiffpath, municipality_code)
    if title is None:
        title = f"Yield Prediction Heatmap — {municipality_name}"

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Pixel Column")
    ax.set_ylabel("Pixel Row")

    # Add statistics text
    valid_predictions = heatmap[~np.isnan(heatmap)]
    if len(valid_predictions) > 0:
        stats_text = _yield_stats_text(valid_predictions, target_unit, aggregation)
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved heatmap to {output_path}")

    return heatmap  # Return heatmap array for combined visualization


def create_ndvi_heatmap(
    ndvi_map,
    tiles,
    tile_height,
    tile_width,
    output_path,
    municipality_code,
    tiffpath=None,
    year=None,
    title_suffix="",
):
    """Create NDVI heatmap visualization (same spatial layout as prediction heatmap)."""
    if not ndvi_map:
        print("  ⚠️  Warning: No NDVI data to visualize")
        return None

    heatmap = _create_heatmap_array_internal(
        ndvi_map,
        tiles,
        tile_height,
        tile_width,
        municipality_code,
        tiffpath,
        year,
    )

    if heatmap is None:
        return None

    fig, ax = plt.subplots(figsize=(12, 10))
    # Use the same smooth sequential colormap as predictions (low=beige, high=green)
    cmap = LinearSegmentedColormap.from_list("beige_green", ["#f4f1e6", "#2e7d32"])
    cmap.set_bad(color="lightgray", alpha=0.3)

    valid_ndvi = heatmap[np.isfinite(heatmap)]
    if len(valid_ndvi) > 0:
        vmin = float(np.nanpercentile(valid_ndvi, 2))
        vmax = float(np.nanpercentile(valid_ndvi, 98))
        vmin = max(vmin, -1.0)
        vmax = min(vmax, 1.0)
        if vmin >= vmax:
            vmin, vmax = -1.0, 1.0
    else:
        vmin, vmax = -1.0, 1.0

    norm = Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(
        heatmap, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto"
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("NDVI", rotation=270, labelpad=20)

    municipality_name = get_municipality_name(tiffpath, municipality_code)

    title = f"NDVI Heatmap — {municipality_name}"
    if title_suffix:
        title += f" - {title_suffix}"
    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Pixel Column")
    ax.set_ylabel("Pixel Row")

    valid_ndvi = heatmap[~np.isnan(heatmap)]
    if len(valid_ndvi) > 0:
        stats_text = (
            f"Min: {valid_ndvi.min():.3f}\n"
            f"Max: {valid_ndvi.max():.3f}\n"
            f"Mean: {valid_ndvi.mean():.3f}\n"
            f"Pixels: {len(valid_ndvi):,}"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved NDVI heatmap to {output_path}")
    return heatmap


BEIGE_GREEN_CMAP = LinearSegmentedColormap.from_list(
    "beige_green", ["#f4f1e6", "#2e7d32"]
)
BEIGE_GREEN_CMAP.set_bad(color="#d9d9d9", alpha=1.0)

REPORT_PANEL_CMAP = LinearSegmentedColormap.from_list(
    "beige_green_report", ["#f4f1e6", "#2e7d32"]
)
REPORT_PANEL_CMAP.set_bad(color="#ffffff", alpha=1.0)


def _concat_finite(*arrays: np.ndarray | None) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for arr in arrays:
        if arr is None:
            continue
        vals = np.asarray(arr, dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size:
            chunks.append(vals)
    if not chunks:
        return np.array([], dtype=float)
    return np.concatenate(chunks)


def _shared_yield_limits(*arrays: np.ndarray | None) -> tuple[float, float]:
    vals = _concat_finite(*arrays)
    if vals.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(vals, 2))
    vmax = float(np.nanpercentile(vals, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        return 0.0, 1.0
    return vmin, vmax


def _shared_ndvi_limits(*arrays: np.ndarray | None) -> tuple[float, float]:
    vals = _concat_finite(*arrays)
    if vals.size == 0:
        return -1.0, 1.0
    vmin = float(np.nanpercentile(vals, 2))
    vmax = float(np.nanpercentile(vals, 98))
    vmin = max(vmin, -1.0)
    vmax = min(vmax, 1.0)
    if vmin >= vmax:
        return -1.0, 1.0
    return vmin, vmax


def _shared_finite_bounds(
    *arrays: np.ndarray | None,
    pad: int = 2,
) -> tuple[slice, slice] | None:
    """Bounding box (rows, cols) covering finite pixels in any of the arrays."""
    combined: np.ndarray | None = None
    for arr in arrays:
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float)
        if a.ndim != 2 or a.size == 0:
            continue
        finite = np.isfinite(a)
        if not finite.any():
            continue
        combined = finite if combined is None else (combined | finite)
    if combined is None:
        return None
    rows, cols = np.where(combined)
    h, w = combined.shape
    r0 = max(0, int(rows.min()) - pad)
    r1 = min(h, int(rows.max()) + 1 + pad)
    c0 = max(0, int(cols.min()) - pad)
    c1 = min(w, int(cols.max()) + 1 + pad)
    return slice(r0, r1), slice(c0, c1)


def _crop_heatmap(heatmap: np.ndarray, bounds: tuple[slice, slice]) -> np.ndarray:
    rs, cs = bounds
    return np.asarray(heatmap, dtype=float)[rs, cs]


def _draw_heatmap_panel(
    ax,
    heatmap: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    cmap=REPORT_PANEL_CMAP,
) -> None:
    ax.set_facecolor("white")
    ax.imshow(
        heatmap,
        cmap=cmap,
        interpolation="nearest",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_axis_off()


def create_intramunicipal_report_panel(
    yield_sem: np.ndarray,
    yield_com: np.ndarray,
    ndvi_mean: np.ndarray,
    ndvi_peak: np.ndarray,
    output_path: Path,
    *,
    municipality_name: str,
    title: str | None = None,
    dpi: int = 160,
) -> None:
    """2×2 report panel: shared tons scale (top row), shared NDVI scale (bottom row)."""
    crop = _shared_finite_bounds(yield_sem, yield_com, ndvi_mean, ndvi_peak)
    if crop is not None:
        yield_sem = _crop_heatmap(yield_sem, crop)
        yield_com = _crop_heatmap(yield_com, crop)
        ndvi_mean = _crop_heatmap(ndvi_mean, crop)
        ndvi_peak = _crop_heatmap(ndvi_peak, crop)

    yield_vmin, yield_vmax = _shared_yield_limits(yield_sem, yield_com)
    ndvi_vmin, ndvi_vmax = _shared_ndvi_limits(ndvi_mean, ndvi_peak)

    rows: list[
        tuple[str, str, str, np.ndarray, np.ndarray, float, float, str]
    ] = [
        (
            "Sem dados de chuva",
            "Com dados de chuva",
            "Previsão (t)",
            yield_sem,
            yield_com,
            yield_vmin,
            yield_vmax,
            "Produção prevista (t)",
        ),
        (
            "NDVI médio (ano colheita)",
            "Pico de NDVI na safra",
            "NDVI",
            ndvi_mean,
            ndvi_peak,
            ndvi_vmin,
            ndvi_vmax,
            "NDVI",
        ),
    ]

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Segoe UI",
            "Helvetica Neue",
            "Arial",
            "DejaVu Sans",
            "sans-serif",
        ],
        "figure.titleweight": "650",
    }

    with plt.rc_context(rc):
        fig = plt.figure(figsize=(10.2, 9.2), facecolor="white")
        outer = gridspec.GridSpec(
            2,
            1,
            figure=fig,
            left=0.04,
            right=0.96,
            top=0.90,
            bottom=0.05,
            hspace=0.28,
        )
        suptitle = title or municipality_name
        fig.suptitle(suptitle, fontsize=14, y=0.97)

        for row_idx, (
            left_hdr,
            right_hdr,
            row_label,
            left_arr,
            right_arr,
            vmin,
            vmax,
            cbar_label,
        ) in enumerate(rows):
            panel_hdrs = (left_hdr, right_hdr)
            inner = gridspec.GridSpecFromSubplotSpec(
                2,
                2,
                subplot_spec=outer[row_idx],
                height_ratios=[1.0, 0.08],
                width_ratios=[1.0, 1.0],
                hspace=0.08,
                wspace=0.04,
            )
            norm = Normalize(vmin=vmin, vmax=vmax)
            sm = ScalarMappable(norm=norm, cmap=REPORT_PANEL_CMAP)
            sm.set_array([])

            for col_idx, arr in enumerate((left_arr, right_arr)):
                ax = fig.add_subplot(inner[0, col_idx])
                _draw_heatmap_panel(ax, arr, vmin=vmin, vmax=vmax)
                ax.text(
                    0.5,
                    1.02,
                    panel_hdrs[col_idx],
                    transform=ax.transAxes,
                    fontsize=10,
                    fontweight="600",
                    ha="center",
                    va="bottom",
                )
                if col_idx == 0:
                    ax.text(
                        0.02,
                        0.98,
                        row_label,
                        transform=ax.transAxes,
                        fontsize=9.5,
                        fontweight="600",
                        va="top",
                        ha="left",
                        color="#1a1a1a",
                    )

            cax = fig.add_subplot(inner[1, :])
            cax.set_facecolor("white")
            cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
            cax.set_xlabel(cbar_label, fontsize=9, labelpad=2)
            cax.tick_params(labelsize=8, pad=1)

    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✓ Saved intramunicipal report panel to {out}")
    print(
        f"    yield scale: {yield_vmin:.2f}–{yield_vmax:.2f} t; "
        f"NDVI scale: {ndvi_vmin:.3f}–{ndvi_vmax:.3f}"
    )


def load_stnet_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    sequencelength: int,
    feature_layout: str,
) -> tuple[STNetRegression, int]:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    state_dict = checkpoint["model_state"]
    input_dim = stnet_regression_input_dim_from_state_dict(state_dict)
    layout = normalize_feature_layout(feature_layout)
    expected_dim = feature_layout_input_dim(layout)
    if input_dim != expected_dim:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has input_dim={input_dim} but "
            f"feature_layout {layout!r} implies {expected_dim}."
        )
    ck_fl = checkpoint.get("feature_layout")
    if ck_fl is not None and normalize_feature_layout(ck_fl) != layout:
        raise ValueError(
            f"Checkpoint feature_layout={ck_fl!r} does not match config {layout!r}."
        )

    run_config = load_latest_run_config(checkpoint_path)
    model_kw = resolve_model_kwargs(run_config, checkpoint)
    model = STNetRegression(
        input_dim=input_dim,
        num_outputs=1,
        max_seq_len=sequencelength,
        **model_kw,
    ).to(device)
    if hasattr(model, "_orig_mod"):
        model._orig_mod.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, input_dim


def map_bundle_to_arrays(
    bundle: dict,
    municipality_code: str,
    *,
    tiffpath,
    year,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Yield, NDVI mean, NDVI peak arrays from a reconstruction bundle."""
    common = dict(
        tiles=bundle["tiles"],
        tile_height=bundle["tile_height"],
        tile_width=bundle["tile_width"],
        municipality_code=municipality_code,
        tiffpath=tiffpath,
        year=year,
    )
    yield_arr = create_heatmap_array(bundle["prediction_map"], **common)
    ndvi_mean = create_heatmap_array(bundle["ndvi_map"], **common)
    ndvi_peak = create_heatmap_array(bundle["ndvi_map_median"], **common)
    if yield_arr is None or ndvi_mean is None or ndvi_peak is None:
        raise RuntimeError(f"Could not build heatmap arrays for {municipality_code}")
    return yield_arr, ndvi_mean, ndvi_peak


def run_municipality_inference(
    model: STNetRegression,
    input_dim: int,
    municipality_code: str,
    args,
    device: torch.device,
) -> dict | None:
    (
        prediction_map,
        ndvi_map,
        ndvi_map_median,
        ndvi_map_per_month,
        tiles,
        tile_height,
        tile_width,
    ) = reconstruct_spatial_predictions(
        model,
        municipality_code,
        args.datapath,
        args.tiffpath,
        args.year,
        args.sequencelength,
        args.rc,
        args.interp,
        args.chunk_size,
        device,
        seed=args.seed,
        year_range=args.year_range,
        reference_date=args.reference_date,
        input_dim=input_dim,
    )
    if prediction_map is None:
        return None
    return {
        "prediction_map": prediction_map,
        "ndvi_map": ndvi_map,
        "ndvi_map_median": ndvi_map_median,
        "ndvi_map_per_month": ndvi_map_per_month,
        "tiles": tiles,
        "tile_height": tile_height,
        "tile_width": tile_width,
    }


def run_report_panel(args) -> None:
    if args.compare_checkpoint is None:
        raise SystemExit("--report-panel requires --compare-checkpoint")
    if len(args.municipality_code) != 1:
        raise SystemExit("--report-panel requires exactly one --municipality-code")

    municipality_code = args.municipality_code[0]
    device = torch.device(args.device)

    if args.output_dir is None:
        run_dir = run_dir_from_path(args.checkpoint)
        args.output_dir = intramunicipal_heatmap_dir(run_dir, create=True)
        if args.figures_subdir:
            args.output_dir = args.output_dir / args.figures_subdir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    file_stem = municipality_output_stem(
        municipality_code, tiff_root=args.tiffpath
    )
    display_name = get_municipality_name(args.tiffpath, municipality_code)
    panel_output = args.panel_output or (args.output_dir / f"{file_stem}_panel.png")

    print(f"Loading sem-chuva checkpoint: {args.checkpoint}")
    model_sem, dim_sem = load_stnet_from_checkpoint(
        args.checkpoint, device, args.sequencelength, args.feature_layout
    )
    print(f"  input_dim={dim_sem}")

    print(f"Loading com-chuva checkpoint: {args.compare_checkpoint}")
    model_com, dim_com = load_stnet_from_checkpoint(
        args.compare_checkpoint, device, args.sequencelength, args.feature_layout
    )
    print(f"  input_dim={dim_com}")

    print(f"\nInference (sem chuva) for {display_name} ({municipality_code})...")
    bundle_sem = run_municipality_inference(
        model_sem, dim_sem, municipality_code, args, device
    )
    if bundle_sem is None:
        raise SystemExit(f"Failed sem-chuva inference for {municipality_code}")

    print(f"Inference (com chuva) for {display_name} ({municipality_code})...")
    bundle_com = run_municipality_inference(
        model_com, dim_com, municipality_code, args, device
    )
    if bundle_com is None:
        raise SystemExit(f"Failed com-chuva inference for {municipality_code}")

    common_kw = dict(
        municipality_code=municipality_code,
        tiffpath=args.tiffpath,
        year=args.year,
    )
    yield_sem, ndvi_mean, ndvi_peak = map_bundle_to_arrays(bundle_sem, **common_kw)
    yield_com, _, _ = map_bundle_to_arrays(bundle_com, **common_kw)

    harvest = args.year_range or str(args.year)
    title = f"{display_name} — safra {harvest.split('-')[-1] if '-' in str(harvest) else harvest}"

    create_intramunicipal_report_panel(
        yield_sem,
        yield_com,
        ndvi_mean,
        ndvi_peak,
        panel_output,
        municipality_name=display_name,
        title=title,
        dpi=args.panel_dpi,
    )


def save_histogram(values, output_path, title, xlabel, bins=50):
    """Save a histogram of the given values (ignores NaN)."""
    values = np.asarray(values).flatten()
    values = values[np.isfinite(values)]
    if len(values) == 0:
        print(f"  ⚠️  Warning: No valid values for histogram {output_path}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins, color="steelblue", edgecolor="white", alpha=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.axvline(
        values.mean(),
        color="darkred",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {values.mean():.3f}",
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved histogram to {output_path}")


def create_combined_heatmap(
    heatmaps_data,
    output_path,
    municipality_codes,
    tiffpath=None,
    *,
    target_unit: str,
    aggregation: str,
):
    """Create a combined heatmap visualization with multiple municipalities side by side."""
    num_municipalities = len(heatmaps_data)

    # Calculate global min/max for consistent color scaling
    all_valid_values = []
    for heatmap, _, _, _ in heatmaps_data:
        valid = heatmap[~np.isnan(heatmap)]
        if len(valid) > 0:
            all_valid_values.extend(valid)

    if len(all_valid_values) == 0:
        print("  ⚠️  Warning: No valid predictions to visualize")
        return

    vmin = min(all_valid_values)
    vmax = max(all_valid_values)

    # Create figure with subplots
    fig, axes = plt.subplots(
        1, num_municipalities, figsize=(12 * num_municipalities, 10)
    )

    # Handle single municipality case (axes is not a list)
    if num_municipalities == 1:
        axes = [axes]

    # Use a colormap (yellow to red for yield)
    cmap = plt.cm.YlOrRd
    cmap.set_bad(color="lightgray", alpha=0.3)  # Color for NaN (no data)

    # Get municipality names
    municipality_names = []
    for municipality_code in municipality_codes:
        if tiffpath:
            name = get_municipality_name(tiffpath, municipality_code)
            municipality_names.append(name)
        else:
            municipality_names.append(municipality_code)

    for idx, (heatmap, tiles, municipality_code, _ndvi_map) in enumerate(heatmaps_data):
        ax = axes[idx]

        # Display heatmap with shared color scale
        im = ax.imshow(
            heatmap,
            cmap=cmap,
            interpolation="nearest",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )

        # Use municipality name if available
        if idx < len(municipality_names):
            title = municipality_names[idx]
        else:
            title = municipality_code

        ax.set_title(
            title,
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Pixel Column")
        ax.set_ylabel("Pixel Row")

        # Add statistics text
        valid_predictions = heatmap[~np.isnan(heatmap)]
        if len(valid_predictions) > 0:
            stats_text = _yield_stats_text(
                valid_predictions, target_unit, aggregation
            )
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                fontsize=10,
            )

    # Add single shared colorbar
    # Create title with names
    if tiffpath and len(municipality_names) == len(municipality_codes):
        title_parts = municipality_names
    else:
        title_parts = municipality_codes

    fig.suptitle(
        f"Yield Prediction Heatmaps - {', '.join(title_parts)}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Adjust layout first to make room for colorbar
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])  # Leave space for suptitle and colorbar

    # Add colorbar to the right of all subplots
    # Position it relative to the figure, not individual axes
    # Get the position of the rightmost axis
    rightmost_ax = axes[-1]
    pos = rightmost_ax.get_position()

    # Create colorbar axes to the right of all subplots
    cbar_ax = fig.add_axes(
        [0.93, pos.y0, 0.02, pos.height]
    )  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    cbar.set_label(f"Yield Prediction ({target_unit})", rotation=270, labelpad=20)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved combined heatmap to {output_path}")


def save_yield_heatmap_only(
    municipality_code,
    prediction_map,
    tiles,
    tile_height,
    tile_width,
    output_dir: Path,
    *,
    tiffpath,
    year,
    suffix: str = "",
    title_suffix: str = "",
    target_unit: str,
    aggregation: str,
) -> bool:
    """Save a single yield prediction heatmap (used for incomplete-series k=1..6)."""
    if prediction_map is None:
        return False
    municipality_name = get_municipality_name(tiffpath, municipality_code)
    file_stem = municipality_output_stem(municipality_code, tiff_root=tiffpath)
    output_path = output_dir / f"{file_stem}_heatmap{suffix}.png"
    plot_title = f"Yield Prediction Heatmap — {municipality_name}"
    if title_suffix:
        plot_title = f"{plot_title} ({title_suffix})"
    print(f"Creating yield heatmap for {municipality_name} ({municipality_code})...")
    create_heatmap(
        prediction_map,
        tiles,
        tile_height,
        tile_width,
        output_path,
        municipality_code,
        tiffpath=tiffpath,
        year=year,
        title=plot_title,
        target_unit=target_unit,
        aggregation=aggregation,
    )
    return True


def save_individual_municipality_figures(
    municipality_code: str,
    prediction_map,
    ndvi_map,
    ndvi_map_median,
    ndvi_map_per_month,
    tiles,
    tile_height,
    tile_width,
    output_dir: Path,
    *,
    tiffpath,
    year,
    target_unit: str,
    aggregation: str,
) -> bool:
    """Write yield / NDVI heatmaps and histograms for one municipality (no model inference)."""
    if prediction_map is None:
        return False

    municipality_name = get_municipality_name(tiffpath, municipality_code)
    file_stem = municipality_output_stem(municipality_code, tiff_root=tiffpath)

    output_path = output_dir / f"{file_stem}_heatmap.png"
    print(f"Creating heatmap visualization for {municipality_name} ({municipality_code})...")
    create_heatmap(
        prediction_map,
        tiles,
        tile_height,
        tile_width,
        output_path,
        municipality_code,
        tiffpath=tiffpath,
        year=year,
        target_unit=target_unit,
        aggregation=aggregation,
    )

    ndvi_mean_suffix = "Mean (harvest year, after month 3)"
    ndvi_output = output_dir / f"{file_stem}_ndvi_heatmap.png"
    print(f"Creating NDVI heatmap (mean) for {municipality_name}...")
    create_ndvi_heatmap(
        ndvi_map,
        tiles,
        tile_height,
        tile_width,
        ndvi_output,
        municipality_code,
        tiffpath=tiffpath,
        year=year,
        title_suffix=ndvi_mean_suffix,
    )

    pred_values = [v for v in prediction_map.values() if np.isfinite(v)]
    if pred_values:
        save_histogram(
            pred_values,
            output_dir / f"{file_stem}_predictions_histogram.png",
            f"Yield predictions - {municipality_name}",
            f"Prediction ({target_unit})",
        )

    ndvi_map = ndvi_map or {}
    ndvi_map_median = ndvi_map_median or {}
    ndvi_map_per_month = ndvi_map_per_month or {}

    ndvi_values = [v for v in ndvi_map.values() if np.isfinite(v)]
    if ndvi_values:
        save_histogram(
            ndvi_values,
            output_dir / f"{file_stem}_ndvi_histogram.png",
            f"NDVI (mean) - {municipality_name}",
            "NDVI",
        )

    ndvi_peak_output = output_dir / f"{file_stem}_ndvi_peak_heatmap.png"
    print(f"Creating NDVI heatmap (peak = max over time) for {municipality_name}...")
    create_ndvi_heatmap(
        ndvi_map_median,
        tiles,
        tile_height,
        tile_width,
        ndvi_peak_output,
        municipality_code,
        tiffpath=tiffpath,
        year=year,
        title_suffix="Peak (max over time)",
    )
    ndvi_peak_values = [v for v in ndvi_map_median.values() if np.isfinite(v)]
    if ndvi_peak_values:
        save_histogram(
            ndvi_peak_values,
            output_dir / f"{file_stem}_ndvi_peak_histogram.png",
            f"NDVI (peak) - {municipality_name}",
            "NDVI",
        )
    print(
        f"\n✓ Done! Heatmaps, NDVI (mean + peak), and histograms saved to {output_dir}"
    )
    return True


def main():
    args, run_config = parse_args()
    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=False
    )
    _, _, args.aggregation, args.target_unit = resolve_inference_target(
        run_config, checkpoint
    )
    print(
        f"Run config: {run_config['config_path']} (session {run_config['session_index']})"
    )
    print(f"Target unit: {args.target_unit} (pixel aggregation: {args.aggregation})")

    if args.incomplete_series:
        if args.num_periods:
            raise SystemExit("Use either --incomplete-series or --num-periods, not both.")
        args.num_periods = list(range(1, 7))

    if args.report_panel:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        run_report_panel(args)
        return

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loading checkpoint from {args.checkpoint}...")
    state_dict = checkpoint["model_state"]
    input_dim = stnet_regression_input_dim_from_state_dict(state_dict)
    print(f"STNet input_dim from checkpoint: {input_dim}")
    feature_layout = normalize_feature_layout(args.feature_layout)
    expected_dim = feature_layout_input_dim(feature_layout)
    if input_dim != expected_dim:
        raise ValueError(
            f"Checkpoint has input_dim={input_dim} but feature_layout {feature_layout!r} "
            f"implies {expected_dim}."
        )
    ck_fl = checkpoint.get("feature_layout")
    if ck_fl is not None and normalize_feature_layout(ck_fl) != feature_layout:
        raise ValueError(
            f"Checkpoint feature_layout={ck_fl!r} does not match config {feature_layout!r}."
        )

    # Create model
    print("Creating model...")
    device = torch.device(args.device)
    model_kw = resolve_model_kwargs(run_config, checkpoint)
    model = STNetRegression(
        input_dim=input_dim,
        num_outputs=1,
        max_seq_len=args.sequencelength,
        **model_kw,
    ).to(device)

    # Load model weights
    if hasattr(model, "_orig_mod"):
        model._orig_mod.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    print("Model loaded successfully")

    # Default figures output next to the checkpoint run
    if args.output_dir is None:
        run_dir = run_dir_from_path(args.checkpoint)
        args.output_dir = intramunicipal_heatmap_dir(run_dir, create=True)
        if args.figures_subdir:
            args.output_dir = args.output_dir / args.figures_subdir
    if args.num_periods:
        args.output_dir = args.output_dir / "incomplete_series"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to {args.output_dir}")

    municipality_codes = args.municipality_code
    period_list = args.num_periods if args.num_periods else [None]
    incomplete_mode = args.num_periods is not None

    heatmaps_data = []

    for municipality_code in municipality_codes:
        for num_periods in period_list:
            k_label = (
                f" (first {num_periods} season month{'s' if num_periods != 1 else ''})"
                if num_periods is not None
                else ""
            )
            print(
                f"\nGenerating pixel-level predictions for municipality "
                f"{municipality_code}{k_label}..."
            )
            (
                prediction_map,
                ndvi_map,
                ndvi_map_median,
                ndvi_map_per_month,
                tiles,
                tile_height,
                tile_width,
            ) = reconstruct_spatial_predictions(
                model,
                municipality_code,
                args.datapath,
                args.tiffpath,
                args.year,
                args.sequencelength,
                args.rc,
                args.interp,
                args.chunk_size,
                device,
                seed=args.seed,
                        year_range=args.year_range,
                reference_date=args.reference_date,
                input_dim=input_dim,
                num_periods=num_periods,
            )

            if prediction_map is None:
                print(f"  ❌ Failed to generate predictions for {municipality_code}{k_label}")
                continue

            if getattr(args, "bias_correct", False):
                from bias_correction import correct_prediction_map_to_pam
                import pandas as pd

                harvest_year = args.year
                if harvest_year is None and args.year_range and "-" in str(args.year_range):
                    harvest_year = int(str(args.year_range).split("-")[-1])
                pam = None
                try:
                    ydf = pd.read_csv(args.yield_csv)
                    ydf["municipality_code"] = ydf["municipality_code"].astype(str)
                    sub = ydf[ydf["municipality_code"] == str(municipality_code)]
                    if harvest_year is not None and "year" in sub.columns:
                        sub = sub[sub["year"].astype(int) == int(harvest_year)]
                    if len(sub) and "yield_t_ha" in sub.columns:
                        pam = float(sub["yield_t_ha"].iloc[0])
                except Exception as exc:
                    print(f"  ⚠️  bias-correct skipped (PAM lookup failed): {exc}")
                if pam is not None:
                    before = float(
                        np.nanmean(list(prediction_map.values()))
                    )
                    prediction_map = correct_prediction_map_to_pam(
                        prediction_map, pam, mode=args.bias_correct_mode
                    )
                    after = float(np.nanmean(list(prediction_map.values())))
                    print(
                        f"  bias-correct ({args.bias_correct_mode}): "
                        f"mean {before:.3f} → {after:.3f} (PAM={pam:.3f})"
                    )

            if incomplete_mode:
                suffix = f"_k{num_periods}"
                save_yield_heatmap_only(
                    municipality_code,
                    prediction_map,
                    tiles,
                    tile_height,
                    tile_width,
                    args.output_dir,
                    tiffpath=args.tiffpath,
                    year=args.year,
                    suffix=suffix,
                    title_suffix=(
                        f"first {num_periods} season month"
                        + ("s" if num_periods != 1 else "")
                    ),
                    target_unit=args.target_unit,
                    aggregation=args.aggregation,
                )
                continue

            if (
                args.daily_ndvi
        
                and tile_height is not None
                and tile_width is not None
                and args.year_range is not None
            ):
                daily_dir = resolve_daily_tiff_dir(
                    args.tiffpath, municipality_code, args.year_range
                )
                if daily_dir is not None:
                    dated = []
                    for p in sorted(daily_dir.glob("*.tif*")):
                        code, d = parse_daily_filename(p.name)
                        if code == municipality_code and d is not None:
                            dated.append((d, p))
                    dated.sort(key=lambda x: x[0])
                    daily_paths = [p for _, p in dated]
                    if daily_paths:
                        print(
                            f"  Saving {len(daily_paths)} daily NDVI heatmaps under "
                            f"{args.output_dir / args.daily_ndvi_subdir / municipality_code} ..."
                        )
                        save_daily_ndvi_heatmaps_from_tiffs(
                            municipality_code,
                            daily_paths,
                            args.output_dir,
                            tile_height,
                            tile_width,
                            subdir_name=args.daily_ndvi_subdir,
                        )

            print(f"  Creating heatmap array for {municipality_code}...")
            heatmap = create_heatmap_array(
                prediction_map,
                tiles,
                tile_height,
                tile_width,
                municipality_code,
                tiffpath=args.tiffpath,
                year=args.year,
            )

            if heatmap is None:
                continue

            heatmaps_data.append(
                {
                    "municipality_code": municipality_code,
                    "prediction_map": prediction_map,
                    "ndvi_map": ndvi_map,
                    "ndvi_map_median": ndvi_map_median,
                    "ndvi_map_per_month": ndvi_map_per_month,
                    "tiles": tiles,
                    "tile_height": tile_height,
                    "tile_width": tile_width,
                    "heatmap": heatmap,
                }
            )

    if incomplete_mode:
        print(f"\n✓ Done! Incomplete-series yield heatmaps saved to {args.output_dir}")
        return

    if len(heatmaps_data) == 0:
        print("\n❌ No heatmaps were successfully created")
        return

    if len(heatmaps_data) == 1:
        bundle = heatmaps_data[0]
        if not save_individual_municipality_figures(
            bundle["municipality_code"],
            bundle["prediction_map"],
            bundle["ndvi_map"],
            bundle["ndvi_map_median"],
            bundle["ndvi_map_per_month"],
            bundle["tiles"],
            bundle["tile_height"],
            bundle["tile_width"],
            args.output_dir,
            tiffpath=args.tiffpath,
            year=args.year,
            target_unit=args.target_unit,
            aggregation=args.aggregation,
        ):
            print("\n❌ Failed to create individual heatmap")
    else:
        output_path = (
            args.output_dir / f"{'_'.join(municipality_codes)}_combined_heatmap.png"
        )
        print(f"\nCreating combined heatmap visualization...")
        create_combined_heatmap(
            [
                (b["heatmap"], b["tiles"], b["municipality_code"], b["ndvi_map"])
                for b in heatmaps_data
            ],
            output_path,
            municipality_codes,
            tiffpath=args.tiffpath,
            target_unit=args.target_unit,
            aggregation=args.aggregation,
        )
        print(f"\n✓ Done! Combined heatmap saved to {output_path}")


if __name__ == "__main__":
    main()
