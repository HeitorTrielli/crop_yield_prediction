"""
Script to create pixel-level prediction heatmaps for municipalities.
Maps predictions back to spatial locations using original TIFF files.
"""

import argparse
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from torch.amp import autocast
from tqdm import tqdm

from datasets.datautils import getWeight
from models import STNetRegression
from utils import recursive_todevice


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create pixel-level prediction heatmaps for municipalities."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model_best.pth checkpoint file",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="files/npy",
        help="Path to dataset root directory containing municipality .npy files",
    )
    parser.add_argument(
        "--tiffpath",
        type=str,
        default="files/daily_tiff",
        help="Path to directory containing original TIFF files (for spatial mapping). For daily mode: .../daily_tiff/ (season/municipality subfolders).",
    )
    parser.add_argument(
        "--tiff-mode",
        type=str,
        default="daily",
        choices=["daily", "monthly_tiles"],
        help="TIFF layout mode. 'daily' expects {tiffpath}/{year-range}/{muni}/{muni}_YYYY_MM_DD.tiff (or legacy .../{muni}/{year-range}/). 'monthly_tiles' expects per-tile monthly TIFFs.",
    )
    parser.add_argument(
        "--year-range",
        type=str,
        default=None,
        help="Season folder name (e.g. 2020-2021). Required for tiff-mode=daily unless --tiffpath already points to {muni}/{year-range}.",
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
        default="heatmaps",
        help="Output directory for heatmap images",
    )
    parser.add_argument(
        "--sequencelength",
        type=int,
        default=45,
        help="Maximum length of time series data (default: 45). For daily mode, extra days are dropped (earliest by DOY).",
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
        default=27,
        help="Random seed for reproducibility (default: 27)",
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
        help="In tiff-mode=daily, also save one NDVI heatmap PNG per observation date (from daily TIFFs).",
    )
    parser.add_argument(
        "--daily-ndvi-subdir",
        type=str,
        default="daily_ndvi",
        help="Subfolder under --output-dir for per-day NDVI PNGs (default: daily_ndvi).",
    )
    args = parser.parse_args()

    args.datapath = Path(args.datapath)
    args.tiffpath = Path(args.tiffpath)
    args.checkpoint = Path(args.checkpoint)
    args.output_dir = Path(args.output_dir)
    if args.reference_date is not None:
        args.reference_date = datetime.strptime(args.reference_date, "%Y-%m-%d").date()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


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


def compute_daily_ndvi_stats_from_npy_row(
    X_row: np.ndarray,
    # Keep literal defaults here because this function is defined before RED_BAND_IDX/NIR_BAND_IDX
    red_idx: int = 2,
    nir_idx: int = 6,
):
    """Compute mean/peak (max) NDVI over valid daily timesteps from one .npy pixel row (T, 11).

    Uses raw reflectance (bands * 1e-4). Treats 0 and -9999 as missing.
    Returns (mean_ndvi, peak_ndvi, ndvi_per_season_month_dict) where peak is max NDVI on valid days.
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
    ndvi_valid = ndvi[valid]
    mean_ndvi = float(np.nanmean(ndvi_valid)) if len(ndvi_valid) else np.nan
    peak_ndvi = float(np.nanmax(ndvi_valid)) if len(ndvi_valid) else np.nan

    # Per-season-month mean NDVI
    per_month = {}
    for m in range(1, 7):
        per_month[m] = np.nan
    return mean_ndvi, peak_ndvi, per_month


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
        # Legacy: .../<resolution>/<muni>/<year-range>/
        if year_range is not None:
            candidate = tiffpath / municipality_code / year_range
            if candidate.is_dir():
                return candidate
        # Fallback: .../<muni>/ (no season in path)
        candidate = tiffpath / municipality_code
        if candidate.is_dir():
            return candidate
    return None


def transform_pixel(x, sequencelength, rc, interp, seed=None, deterministic_head=False):
    """Transform pixel data: normalize, pad/sample to sequencelength, extract DOY.
    If deterministic_head is True and x has more than sequencelength rows, keeps the earliest sequencelength rows.
    """
    if seed is not None:
        np.random.seed(seed)

    doy = x[:, -1].astype(np.int32)
    x = x[:, :10] * 1e-4

    mean = np.array(
        [[0.147, 0.169, 0.186, 0.221, 0.273, 0.297, 0.308, 0.316, 0.256, 0.188]]
    )
    std = np.array([0.227, 0.219, 0.222, 0.22, 0.2, 0.193, 0.192, 0.182, 0.123, 0.106])

    weight = getWeight(x)
    x = (x - mean) / std

    if interp:
        doy_pad = np.linspace(0, 366, sequencelength).astype("int")
        x_pad = np.array([np.interp(doy_pad, doy, x[:, i]) for i in range(10)]).T
        weight_pad = getWeight(x_pad * std + mean)
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
    """Extract municipality name from directory name (e.g., '4100103_Abatiá' -> 'Abatiá')."""
    for dir_path in tiff_dir.iterdir():
        if dir_path.is_dir() and dir_path.name.startswith(municipality_code + "_"):
            # Extract name after the code and underscore
            name = dir_path.name[len(municipality_code) + 1 :]
            return name
    # Fallback: return code if name not found
    return municipality_code


def get_tile_structure(tiff_dir, municipality_code, year):
    """Get tile structure and dimensions from TIFF files."""
    # Find directory that starts with municipality_code (e.g., "4100103_Abatiá")
    muni_dir = None
    for dir_path in tiff_dir.iterdir():
        if dir_path.is_dir() and dir_path.name.startswith(municipality_code + "_"):
            muni_dir = dir_path
            break

    if muni_dir is None:
        # Try exact match as fallback
        muni_dir = tiff_dir / municipality_code
        if not muni_dir.exists():
            return None, None, None

    tiff_files = list(muni_dir.glob("*.tif"))
    if not tiff_files:
        return None, None, None

    # Group by tile
    tiles = {}
    for tiff_file in tiff_files:
        _, file_year, month, tile_x, tile_y = parse_filename(tiff_file.name)
        if file_year != year:
            continue
        tile_key = (tile_x, tile_y)
        if tile_key not in tiles:
            tiles[tile_key] = {}
        tiles[tile_key][month] = tiff_file

    if not tiles:
        return None, None, None

    # Get dimensions from first tile
    first_tile = list(tiles.values())[0]
    first_file = list(first_tile.values())[0]
    with rasterio.open(first_file) as src:
        tile_height, tile_width = src.height, src.width

    return tiles, tile_height, tile_width


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
    tiff_mode="monthly_tiles",
    year_range=None,
    reference_date=None,
):
    """
    Reconstruct spatial layout of predictions by processing pixels in same order as preprocessing.
    Uses TIFF files to get exact spatial structure and matches pixels to .npy data.
    Returns: prediction_map dict with (tile_x, tile_y, row, col) -> prediction, or None if mapping fails
    """
    if tiff_mode == "daily":
        if year_range is None:
            print("  ⚠️  Warning: --year-range is required for tiff-mode=daily (unless --tiffpath points directly to {muni}/{year-range})")
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

        # Load .npy file
        muni_npy_file = datapath / municipality_code / f"{municipality_code}.npy"
        if not muni_npy_file.exists():
            print(f"  ⚠️  Warning: {muni_npy_file} does not exist")
            return None, None, None, None, None, None, None

        try:
            municipality_data = np.load(muni_npy_file, mmap_mode="r")
        except Exception as e:
            print(f"  ⚠️  Warning: Could not load {muni_npy_file}: {e}")
            return None, None, None, None, None, None, None

        if len(municipality_data) == 0:
            print(f"  ⚠️  Warning: {municipality_code} has 0 pixels")
            return None, None, None, None, None, None, None

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
                            mean_ndvi, peak_ndvi, _ = compute_daily_ndvi_stats_from_npy_row(X)
                        except Exception:
                            mean_ndvi, peak_ndvi = np.nan, np.nan
                        ndvi_map[("grid", row, col)] = mean_ndvi
                        # Reuse the same return slot as the legacy "median" map, but in daily mode it is peak (max) NDVI.
                        ndvi_map_median[("grid", row, col)] = peak_ndvi

                        X_tuple = transform_pixel(
                            X, sequencelength, rc, interp, seed=seed, deterministic_head=True
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

    # monthly_tiles mode (legacy)
    # Get tile structure and load TIFF files
    tiles, tile_height, tile_width = get_tile_structure(tiffpath, municipality_code, year)

    if tiles is None:
        print(f"  ⚠️  Warning: No TIFF files found for municipality {municipality_code}")
        return None, None, None, None, None, None, None

    # Load .npy file
    muni_npy_file = datapath / municipality_code / f"{municipality_code}.npy"
    if not muni_npy_file.exists():
        print(f"  ⚠️  Warning: {muni_npy_file} does not exist")
        return None, None, None, None, None, None, None

    try:
        municipality_data = np.load(muni_npy_file, mmap_mode="r")
    except Exception as e:
        print(f"  ⚠️  Warning: Could not load {muni_npy_file}: {e}")
        return None, None, None, None, None, None, None

    if len(municipality_data) == 0:
        print(f"  ⚠️  Warning: {municipality_code} has 0 pixels")
        return None, None, None, None, None, None, None

    # Group TIFF files by month
    # Find directory that starts with municipality_code (e.g., "4100103_Abatiá")
    tiff_dir = None
    for dir_path in tiffpath.iterdir():
        if dir_path.is_dir() and dir_path.name.startswith(municipality_code + "_"):
            tiff_dir = dir_path
            break

    if tiff_dir is None:
        # Try exact match as fallback
        tiff_dir = tiffpath / municipality_code
        if not tiff_dir.exists():
            print(
                f"  ⚠️  Warning: No TIFF directory found for municipality {municipality_code}"
            )
            return None, None, None, None, None, None, None, None

    tiff_files = list(tiff_dir.glob("*.tif"))
    files_by_month = defaultdict(dict)
    tiles = {}  # Rebuild tiles structure from files

    for tiff_file in tiff_files:
        _, file_year, month, tile_x, tile_y = parse_filename(tiff_file.name)
        if file_year != year:
            continue
        files_by_month[month][(tile_x, tile_y)] = tiff_file
        # Build tiles structure
        tile_key = (tile_x, tile_y)
        if tile_key not in tiles:
            tiles[tile_key] = {}
        tiles[tile_key][month] = tiff_file

    if not tiles:
        print(
            f"  ⚠️  Warning: No tiles found for municipality {municipality_code} in year {year}"
        )
        return None, None, None, None, None, None, None

    expected_months = sorted(files_by_month.keys())
    month_to_doy = {1: 1, 2: 32, 3: 60, 4: 91, 5: 121, 6: 152}

    model.eval()
    prediction_map = {}
    ndvi_map = {}  # Same keys as prediction_map; value = mean NDVI for that pixel
    ndvi_map_per_month = defaultdict(
        dict
    )  # month -> {(tile_x, tile_y, row, col): ndvi}
    npy_pixel_idx = 0  # Index into .npy file (only valid pixels)

    # Process tiles in sorted order (same as preprocessing)
    sorted_tiles = sorted(tiles.keys())

    print(
        f"  Processing {len(sorted_tiles)} tiles, {len(municipality_data)} valid pixels in .npy file"
    )

    with torch.no_grad():
        for tile_x, tile_y in tqdm(sorted_tiles, desc="  Tiles", unit="tile"):
            # Load TIFF data for this tile and get dimensions
            tiff_data = {}
            nodata_values = {}
            tile_height = None
            tile_width = None

            # First, get dimensions from first available TIFF file for this tile
            for month in expected_months:
                if (tile_x, tile_y) in files_by_month[month]:
                    tiff_file = files_by_month[month][(tile_x, tile_y)]
                    try:
                        with rasterio.open(tiff_file) as src:
                            tile_height, tile_width = src.height, src.width
                            break
                    except Exception:
                        continue

            # If no TIFF file found, skip this tile
            if tile_height is None or tile_width is None:
                print(
                    f"  ⚠️  Warning: Could not determine dimensions for tile ({tile_x}, {tile_y}), skipping"
                )
                continue

            # Now load all months for this tile
            for month in expected_months:
                if (tile_x, tile_y) in files_by_month[month]:
                    tiff_file = files_by_month[month][(tile_x, tile_y)]
                    try:
                        with rasterio.open(tiff_file) as src:
                            pixel_data = src.read()[:10].astype(np.float32)
                            # Verify dimensions match
                            if (
                                pixel_data.shape[1] != tile_height
                                or pixel_data.shape[2] != tile_width
                            ):
                                print(
                                    f"  ⚠️  Warning: Tile ({tile_x}, {tile_y}) month {month} has different dimensions, using zeros"
                                )
                                tiff_data[month] = np.zeros(
                                    (10, tile_height, tile_width), dtype=np.float32
                                )
                            else:
                                tiff_data[month] = pixel_data
                            nodata_values[month] = src.nodata
                    except Exception as e:
                        print(f"  ⚠️  Warning: Could not load {tiff_file.name}: {e}")
                        tiff_data[month] = np.zeros(
                            (10, tile_height, tile_width), dtype=np.float32
                        )
                        nodata_values[month] = None
                else:
                    tiff_data[month] = np.zeros(
                        (10, tile_height, tile_width), dtype=np.float32
                    )
                    nodata_values[month] = None

            # Process pixels row by row, col by col (same order as preprocessing)
            current_chunk = []
            chunk_indices = []  # Track which pixels belong to this chunk

            for row in range(tile_height):
                for col in range(tile_width):
                    # Reconstruct pixel timeseries from TIFF (same as preprocessing)
                    timeseries = []
                    doys = []

                    for month in expected_months:
                        pixel_data = tiff_data[month]  # [10, height, width]
                        pixel_values = pixel_data[:, row, col]  # [10]

                        # Check for nodata (same as preprocessing)
                        if nodata_values[month] is not None and np.all(
                            pixel_values == nodata_values[month]
                        ):
                            pixel_values = np.zeros(10, dtype=np.float32)

                        timeseries.append(pixel_values)
                        doys.append(month_to_doy.get(month, month * 30))

                    timeseries = np.array(timeseries, dtype=np.float32)
                    doys = np.array(doys, dtype=np.uint8)

                    # Apply same filtering logic as preprocessing
                    if should_include_pixel(timeseries):
                        # Only store NDVI for soybean pixels (same as prediction coverage)
                        pixel_key = (tile_x, tile_y, row, col)
                        ndvi_val = compute_ndvi_from_timeseries(timeseries)
                        ndvi_map[pixel_key] = ndvi_val
                        for month_idx, month in enumerate(expected_months):
                            ndvi_month = compute_ndvi_single_band(timeseries[month_idx])
                            ndvi_map_per_month[month][pixel_key] = ndvi_month
                        # This pixel should be in the .npy file
                        if npy_pixel_idx >= len(municipality_data):
                            print(
                                f"  ⚠️  Warning: More valid pixels than expected in .npy file"
                            )
                            break

                        # Use data from .npy file (it's already processed)
                        X = municipality_data[npy_pixel_idx].copy()
                        X_tuple = transform_pixel(
                            X, sequencelength, rc, interp, seed=seed
                        )
                        current_chunk.append(X_tuple)
                        chunk_indices.append((tile_x, tile_y, row, col))
                        npy_pixel_idx += 1
                    else:
                        # This pixel was filtered out - mark as NaN in heatmap
                        prediction_map[(tile_x, tile_y, row, col)] = np.nan

                    # Process chunk when it reaches chunk_size
                    # Note: Need at least 2 pixels to avoid BatchNorm issues (single pixel causes 1D tensor)
                    # Process if we have chunk_size, or if we have at least 2 pixels and we're at the end of the tile
                    is_last_pixel_in_tile = (
                        row == tile_height - 1 and col == tile_width - 1
                    )
                    if len(current_chunk) >= chunk_size or (
                        len(current_chunk) >= 2 and is_last_pixel_in_tile
                    ):
                        # Process chunk
                        chunk_x = torch.stack([p[0] for p in current_chunk])
                        chunk_mask = torch.stack([p[1] for p in current_chunk])
                        chunk_doy = torch.stack([p[2] for p in current_chunk])
                        chunk_weight = torch.stack([p[3] for p in current_chunk])

                        municipality_X_chunk = (
                            chunk_x,
                            chunk_mask,
                            chunk_doy,
                            chunk_weight,
                        )
                        municipality_X_chunk = recursive_todevice(
                            municipality_X_chunk, device
                        )

                        with (
                            autocast("cuda", dtype=torch.bfloat16)
                            if device == "cuda"
                            else torch.no_grad()
                        ):
                            chunk_predictions = model(municipality_X_chunk)

                            # Handle single-pixel case (model returns 1D tensor)
                            if chunk_predictions.dim() == 1:
                                chunk_predictions = chunk_predictions.unsqueeze(0)

                        # Store predictions with spatial info
                        for i, (tx, ty, r, c) in enumerate(chunk_indices):
                            pred = chunk_predictions[i].item()
                            prediction_map[(tx, ty, r, c)] = pred

                        current_chunk = []
                        chunk_indices = []

                if npy_pixel_idx >= len(municipality_data):
                    break

            # Process remaining pixels in this tile
            # Only process if we have at least 2 pixels (to avoid BatchNorm 1D input issue)
            if current_chunk and len(current_chunk) >= 2:
                chunk_x = torch.stack([p[0] for p in current_chunk])
                chunk_mask = torch.stack([p[1] for p in current_chunk])
                chunk_doy = torch.stack([p[2] for p in current_chunk])
                chunk_weight = torch.stack([p[3] for p in current_chunk])

                municipality_X_chunk = (
                    chunk_x,
                    chunk_mask,
                    chunk_doy,
                    chunk_weight,
                )
                municipality_X_chunk = recursive_todevice(municipality_X_chunk, device)

                with (
                    autocast("cuda", dtype=torch.bfloat16)
                    if device == "cuda"
                    else torch.no_grad()
                ):
                    chunk_predictions = model(municipality_X_chunk)

                    # Handle single-pixel case (model returns 1D tensor)
                    if chunk_predictions.dim() == 1:
                        chunk_predictions = chunk_predictions.unsqueeze(0)

                for i, (tx, ty, r, c) in enumerate(chunk_indices):
                    pred = chunk_predictions[i].item()
                    prediction_map[(tx, ty, r, c)] = pred
            elif current_chunk and len(current_chunk) == 1:
                # Handle single pixel separately - duplicate it to avoid BatchNorm issue
                single_pixel = current_chunk[0]
                single_indices = chunk_indices[0]

                # Duplicate the pixel to create a batch of 2
                chunk_x = torch.stack([single_pixel[0], single_pixel[0]])
                chunk_mask = torch.stack([single_pixel[1], single_pixel[1]])
                chunk_doy = torch.stack([single_pixel[2], single_pixel[2]])
                chunk_weight = torch.stack([single_pixel[3], single_pixel[3]])

                municipality_X_chunk = (
                    chunk_x,
                    chunk_mask,
                    chunk_doy,
                    chunk_weight,
                )
                municipality_X_chunk = recursive_todevice(municipality_X_chunk, device)

                with (
                    autocast("cuda", dtype=torch.bfloat16)
                    if device == "cuda"
                    else torch.no_grad()
                ):
                    chunk_predictions = model(municipality_X_chunk)
                    # Take only the first prediction (the duplicate is just for batch processing)
                    pred = chunk_predictions[0].item()
                    tx, ty, r, c = single_indices
                    prediction_map[(tx, ty, r, c)] = pred

    # Median NDVI over the six months (same keys as ndvi_map)
    ndvi_map_median = {}
    for pixel_key in ndvi_map:
        vals = [
            ndvi_map_per_month[month][pixel_key]
            for month in ndvi_map_per_month
            if pixel_key in ndvi_map_per_month[month]
        ]
        if vals:
            med = np.nanmedian(vals)
            ndvi_map_median[pixel_key] = float(med) if np.isfinite(med) else np.nan

    return (
        prediction_map,
        ndvi_map,
        ndvi_map_median,
        dict(ndvi_map_per_month),
        tiles,
        tile_height,
        tile_width,
    )


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
    cbar.set_label("Yield Prediction (tons)", rotation=270, labelpad=20)

    # Get municipality name from TIFF directory
    if tiffpath:
        municipality_name = get_municipality_name(tiffpath, municipality_code)
    else:
        municipality_name = municipality_code

    ax.set_title(
        f"Yield Prediction Heatmap - {municipality_name} ({municipality_code})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Pixel Column")
    ax.set_ylabel("Pixel Row")

    # Add statistics text
    valid_predictions = heatmap[~np.isnan(heatmap)]
    if len(valid_predictions) > 0:
        stats_text = (
            f"Min: {valid_predictions.min():.2f} tons\n"
            f"Max: {valid_predictions.max():.2f} tons\n"
            f"Mean: {valid_predictions.mean():.2f} tons\n"
            f"Total: {valid_predictions.sum():.2f} tons\n"
            f"Pixels: {len(valid_predictions):,}"
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

    if tiffpath:
        municipality_name = get_municipality_name(tiffpath, municipality_code)
    else:
        municipality_name = municipality_code

    title = f"NDVI Heatmap - {municipality_name} ({municipality_code})"
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
            municipality_names.append(f"{name} ({municipality_code})")
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
            stats_text = (
                f"Min: {valid_predictions.min():.2f} tons\n"
                f"Max: {valid_predictions.max():.2f} tons\n"
                f"Mean: {valid_predictions.mean():.2f} tons\n"
                f"Total: {valid_predictions.sum():.2f} tons\n"
                f"Pixels: {len(valid_predictions):,}"
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
    cbar.set_label("Yield Prediction (tons)", rotation=270, labelpad=20)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved combined heatmap to {output_path}")


def main():
    args = parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=False
    )

    # Create model
    print("Creating model...")
    device = torch.device(args.device)
    model = STNetRegression(
        input_dim=10,
        num_outputs=1,
        max_seq_len=args.sequencelength,
    ).to(device)

    # Load model weights
    state_dict = checkpoint["model_state"]
    if hasattr(model, "_orig_mod"):
        model._orig_mod.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    print("Model loaded successfully")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    municipality_codes = args.municipality_code

    # Process each municipality
    heatmaps_data = []

    for municipality_code in municipality_codes:
        # Generate spatial predictions
        print(
            f"\nGenerating pixel-level predictions for municipality {municipality_code}..."
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
            tiff_mode=args.tiff_mode,
            year_range=args.year_range,
            reference_date=args.reference_date,
        )

        if prediction_map is None:
            print(f"  ❌ Failed to generate predictions for {municipality_code}")
            continue

        if (
            args.daily_ndvi
            and args.tiff_mode == "daily"
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

        # Create heatmap array (without saving)
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

        if heatmap is not None:
            heatmaps_data.append((heatmap, tiles, municipality_code, ndvi_map))

    if len(heatmaps_data) == 0:
        print("\n❌ No heatmaps were successfully created")
        return

    # Create visualization(s)
    if len(heatmaps_data) == 1:
        # Single municipality - recreate prediction_map and create individual heatmap
        municipality_code = municipality_codes[0]
        print(f"\nRecreating predictions for individual heatmap...")
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
            tiff_mode=args.tiff_mode,
            year_range=args.year_range,
            reference_date=args.reference_date,
        )

        if prediction_map is not None:
            municipality_name = (
                get_municipality_name(args.tiffpath, municipality_code)
                if args.tiffpath
                else municipality_code
            )

            output_path = args.output_dir / f"{municipality_code}_heatmap.png"
            print(f"Creating heatmap visualization...")
            create_heatmap(
                prediction_map,
                tiles,
                tile_height,
                tile_width,
                output_path,
                municipality_code,
                tiffpath=args.tiffpath,
                year=args.year,
            )

            # NDVI heatmap (mean over months)
            ndvi_output = args.output_dir / f"{municipality_code}_ndvi_heatmap.png"
            print(f"Creating NDVI heatmap (mean)...")
            create_ndvi_heatmap(
                ndvi_map,
                tiles,
                tile_height,
                tile_width,
                ndvi_output,
                municipality_code,
                tiffpath=args.tiffpath,
                year=args.year,
            )

            # Histogram of yield predictions
            pred_values = [v for v in prediction_map.values() if np.isfinite(v)]
            if pred_values:
                save_histogram(
                    pred_values,
                    args.output_dir / f"{municipality_code}_predictions_histogram.png",
                    f"Yield predictions - {municipality_name}",
                    "Prediction (tons)",
                )

            # NDVI mean: over time in daily mode; over months in monthly_tiles mode
            ndvi_map = ndvi_map or {}
            ndvi_map_median = ndvi_map_median or {}
            ndvi_map_per_month = ndvi_map_per_month or {}

            ndvi_values = [v for v in ndvi_map.values() if np.isfinite(v)]
            if ndvi_values:
                save_histogram(
                    ndvi_values,
                    args.output_dir / f"{municipality_code}_ndvi_histogram.png",
                    f"NDVI (mean) - {municipality_name}",
                    "NDVI",
                )

            if args.tiff_mode == "daily":
                ndvi_peak_output = (
                    args.output_dir / f"{municipality_code}_ndvi_peak_heatmap.png"
                )
                print(f"Creating NDVI heatmap (peak = max over time)...")
                create_ndvi_heatmap(
                    ndvi_map_median,
                    tiles,
                    tile_height,
                    tile_width,
                    ndvi_peak_output,
                    municipality_code,
                    tiffpath=args.tiffpath,
                    year=args.year,
                    title_suffix="Peak (max over time)",
                )
                ndvi_peak_values = [v for v in ndvi_map_median.values() if np.isfinite(v)]
                if ndvi_peak_values:
                    save_histogram(
                        ndvi_peak_values,
                        args.output_dir
                        / f"{municipality_code}_ndvi_peak_histogram.png",
                        f"NDVI (peak) - {municipality_name}",
                        "NDVI",
                    )
            else:
                # NDVI heatmap and histogram (median over months)
                ndvi_median_output = (
                    args.output_dir / f"{municipality_code}_ndvi_median_heatmap.png"
                )
                print(f"Creating NDVI heatmap (median)...")
                create_ndvi_heatmap(
                    ndvi_map_median,
                    tiles,
                    tile_height,
                    tile_width,
                    ndvi_median_output,
                    municipality_code,
                    tiffpath=args.tiffpath,
                    year=args.year,
                    title_suffix="Median (6 months)",
                )
                ndvi_median_values = [
                    v for v in ndvi_map_median.values() if np.isfinite(v)
                ]
                if ndvi_median_values:
                    save_histogram(
                        ndvi_median_values,
                        args.output_dir
                        / f"{municipality_code}_ndvi_median_histogram.png",
                        f"NDVI (median, 6 months) - {municipality_name}",
                        "NDVI",
                    )

            # Per-month NDVI heatmaps and histograms (one per TIFF month)
            month_names = {
                1: "Jan",
                2: "Feb",
                3: "Mar",
                4: "Apr",
                5: "May",
                6: "Jun",
                7: "Jul",
                8: "Aug",
                9: "Sep",
                10: "Oct",
                11: "Nov",
                12: "Dec",
            }
            for month in sorted(ndvi_map_per_month.keys()):
                month_label = month_names.get(month, f"Month {month}")
                month_suffix = f"Month {month:02d} ({month_label})"
                print(f"  Creating NDVI heatmap for {month_suffix}...")
                create_ndvi_heatmap(
                    ndvi_map_per_month[month],
                    tiles,
                    tile_height,
                    tile_width,
                    args.output_dir
                    / f"{municipality_code}_ndvi_heatmap_month_{month:02d}.png",
                    municipality_code,
                    tiffpath=args.tiffpath,
                    year=args.year,
                    title_suffix=month_suffix,
                )
                m_vals = [
                    v for v in ndvi_map_per_month[month].values() if np.isfinite(v)
                ]
                if m_vals:
                    save_histogram(
                        m_vals,
                        args.output_dir
                        / f"{municipality_code}_ndvi_histogram_month_{month:02d}.png",
                        f"NDVI - {municipality_name} ({month_suffix})",
                        "NDVI",
                    )

            if args.tiff_mode == "daily":
                print(
                    f"\n✓ Done! Heatmaps, NDVI (mean + peak), and histograms saved to {args.output_dir}"
                )
            else:
                print(
                    f"\n✓ Done! Heatmaps, NDVI (mean + median + per month), and histograms saved to {args.output_dir}"
                )
        else:
            print(f"\n❌ Failed to create individual heatmap")
    else:
        # Multiple municipalities - create combined heatmap
        output_path = (
            args.output_dir / f"{'_'.join(municipality_codes)}_combined_heatmap.png"
        )
        print(f"\nCreating combined heatmap visualization...")
        create_combined_heatmap(
            heatmaps_data,
            output_path,
            municipality_codes,
            tiffpath=args.tiffpath,
        )
        print(f"\n✓ Done! Combined heatmap saved to {output_path}")


if __name__ == "__main__":
    main()
