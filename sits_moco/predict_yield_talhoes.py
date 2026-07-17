#!/usr/bin/env python3
"""
Predict yield at talhão (plot) level by aggregating pixel predictions inside each plot polygon
(sum for total production, mean for productivity / t/ha).

Uses the same STNet checkpoint and .npy inputs as predict_yield.py, but assigns pixels to
talhões via daily municipal TIFF georeferencing (same row-major order as preprocess_daily_to_npy).

Typical usage (all talhões in benchmark shapes, Seed6007):

  python scripts/run_benchmark_talhoes_predictions.py

Or call directly (no --municipalities = all talhões in the GPKG/GeoJSON):

  python predict_yield_talhoes.py --checkpoint results/Total_20_22_23_Seed6007/training/model_best.pth --datapath files/npy --tiffpath files/daily_tiff --year-range 2020-2021 --sequencelength 45

Polygons: benchmark/talhoes_baseline.geojson (or .gpkg). Output CSV defaults to
{run_dir}/predictions/talhoes_forecasts.csv with columns forecast (t), baseline_tons, error (t),
and optional forecast_t_ha / baseline_t_ha for productivity comparison.

Only pixels inside talhão polygons are sent through the model. Valid-pixel alignment with the
.npy uses a fast direct-TIFF mask with optional cache at {muni}/{muni}.keep_mask.npy.
"""

from __future__ import annotations

import argparse
import importlib.util
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio import features
from rasterio.warp import Resampling, reproject
from torch.amp import autocast
from tqdm import tqdm

from datasets.pixel_transform import PixelTransform
from models import STNetRegression
from run_paths import (
    apply_run_config_to_args,
    load_latest_run_config,
    predictions_dir,
    resolve_checkpoint_path,
    run_dir_from_path,
)
from utils_aggregated import (
    regression_metrics,
    resolve_inference_target,
    resolve_model_kwargs,
    stnet_regression_input_dim_from_state_dict,
)


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return [recursive_todevice(c, device) for c in x]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_REPO = Path(__file__).resolve().parent
_feature_layout = _load_module(
    "feature_layout", _REPO / "datasets" / "feature_layout.py"
)
_datautils = _load_module("datautils", _REPO / "datasets" / "datautils.py")
feature_layout_choices = _feature_layout.feature_layout_choices
feature_layout_input_dim = _feature_layout.feature_layout_input_dim
normalize_feature_layout = _feature_layout.normalize_feature_layout
getWeight = _datautils.getWeight

NO_DATA_VALUE = -9999
NUM_SPECTRAL_BANDS = 10


def _read_bands_on_reference_grid(
    path: Path,
    ref_transform: object,
    ref_crs: object,
    height: int,
    width: int,
    nbands: int = NUM_SPECTRAL_BANDS,
) -> np.ndarray:
    """Warp daily TIFF bands onto the reference grid (matches preprocess_daily_to_npy)."""
    out = np.full((nbands, height, width), np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        src_crs = src.crs if src.crs is not None else ref_crs
        n = min(src.count, nbands)
        nodata = src.nodata if src.nodata is not None else NO_DATA_VALUE
        for b in range(n):
            reproject(
                rasterio.band(src, b + 1),
                out[b],
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.nearest,
                dst_nodata=np.nan,
            )
        for b in range(n):
            band = out[b]
            band[band == nodata] = np.nan
            band[band == 0] = np.nan
            band[band == NO_DATA_VALUE] = np.nan
    return out


def _reference_grid_from_first_tiff(daily_tiffs: list[Path]):
    """(transform, crs, height, width) from the first daily TIFF."""
    with rasterio.open(daily_tiffs[0]) as src0:
        if src0.crs is None:
            return None, None, None, None
        return src0.transform, src0.crs, src0.height, src0.width


def _accumulate_valid_pixel_mask_from_day(
    data: np.ndarray,
    *,
    any_data: np.ndarray,
    days_with_data: np.ndarray,
) -> None:
    day_has = np.any(np.isfinite(data) & (data != 0), axis=0)
    any_data |= day_has
    days_with_data += day_has.astype(np.uint16)


def _read_day_bands_for_mask(
    path: Path,
    ref_transform,
    ref_crs,
    height: int,
    width: int,
    *,
    force_reproject: bool = False,
) -> np.ndarray | None:
    """Read spectral bands for one day; warp onto the reference grid when shapes differ."""
    if not force_reproject:
        try:
            with rasterio.open(path) as src:
                if src.height == height and src.width == width:
                    nodata = src.nodata if src.nodata is not None else NO_DATA_VALUE
                    data = src.read()[:NUM_SPECTRAL_BANDS].astype(np.float32)
                    data[data == nodata] = np.nan
                    data[data == NO_DATA_VALUE] = np.nan
                    return data
        except OSError:
            pass
    try:
        return _read_bands_on_reference_grid(
            path, ref_transform, ref_crs, height, width
        )
    except OSError:
        return None


def _compute_daily_valid_pixel_mask(
    daily_tiffs: list[Path],
    *,
    force_reproject: bool = False,
) -> np.ndarray | None:
    """Valid-pixel mask; warps days that do not share the first TIFF grid."""
    if not daily_tiffs:
        return None
    ref_transform, ref_crs, height, width = _reference_grid_from_first_tiff(daily_tiffs)
    if ref_transform is None:
        return None
    any_data = np.zeros((height, width), dtype=bool)
    days_with_data = np.zeros((height, width), dtype=np.uint16)
    for p in daily_tiffs:
        data = _read_day_bands_for_mask(
            p,
            ref_transform,
            ref_crs,
            height,
            width,
            force_reproject=force_reproject,
        )
        if data is None:
            continue
        _accumulate_valid_pixel_mask_from_day(
            data, any_data=any_data, days_with_data=days_with_data
        )
    return any_data & (days_with_data > 2)


def compute_daily_valid_pixel_mask_fast(daily_tiffs: list[Path]) -> np.ndarray | None:
    """Valid-pixel mask: direct read when grids match, else warp that day."""
    return _compute_daily_valid_pixel_mask(daily_tiffs, force_reproject=False)


def compute_daily_valid_pixel_mask_reproject(
    daily_tiffs: list[Path],
) -> np.ndarray | None:
    """Valid-pixel mask with every day warped onto the first TIFF grid."""
    return _compute_daily_valid_pixel_mask(daily_tiffs, force_reproject=True)


def keep_mask_cache_path(npy_path: Path) -> Path:
    return npy_path.parent / f"{npy_path.stem}.keep_mask.npy"


def resolve_keep_mask(
    daily_tiffs: list[Path],
    npy_path: Path,
    *,
    reproject: bool,
    use_cache: bool,
) -> np.ndarray | None:
    """Load cached keep_mask or compute it; verify row count matches .npy."""
    if not daily_tiffs:
        return None

    cache = keep_mask_cache_path(npy_path)
    if use_cache and cache.is_file():
        keep_mask = np.load(cache)
        try:
            npy_rows = len(np.load(npy_path, mmap_mode="r"))
        except OSError:
            npy_rows = None
        if npy_rows is not None and int(keep_mask.sum()) == npy_rows:
            return keep_mask

    if reproject:
        keep_mask = compute_daily_valid_pixel_mask_reproject(daily_tiffs)
    else:
        keep_mask = compute_daily_valid_pixel_mask_fast(daily_tiffs)

    if keep_mask is None:
        return None

    try:
        npy_rows = len(np.load(npy_path, mmap_mode="r"))
    except OSError:
        npy_rows = None
    if npy_rows is not None and int(keep_mask.sum()) != npy_rows:
        raise ValueError(
            f"keep_mask True count ({int(keep_mask.sum())}) != .npy rows ({npy_rows}) "
            f"for {npy_path}. Try --reproject-tiffs or delete {cache}."
        )

    if use_cache:
        np.save(cache, keep_mask)
    return keep_mask


def build_talhao_union_mask(
    geometries,
    transform,
    out_shape: tuple[int, int],
) -> np.ndarray:
    """True where pixel centers fall inside any talhão polygon."""
    if not len(geometries):
        return np.zeros(out_shape, dtype=bool)
    geoms = [g for g in geometries if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(out_shape, dtype=bool)
    return features.geometry_mask(
        geoms,
        out_shape=out_shape,
        transform=transform,
        invert=True,
    )


def parse_daily_filename(filename: str):
    """4109401_2020_10_04.tiff -> (code, date)."""
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 4:
        return None, None
    code = parts[0]
    try:
        return code, date(int(parts[1]), int(parts[2]), int(parts[3]))
    except (TypeError, ValueError):
        return None, None


def resolve_daily_tiff_dir(
    tiffpath: Path, municipality_code: str, year_range: str | None
) -> Path | None:
    if not tiffpath.is_dir():
        return None
    if year_range is not None:
        candidate = tiffpath / year_range / municipality_code
        if candidate.is_dir():
            return candidate
        candidate = tiffpath / municipality_code / year_range
        if candidate.is_dir():
            return candidate
    candidate = tiffpath / municipality_code
    return candidate if candidate.is_dir() else None


def transform_pixel(
    x,
    pixel_transform: PixelTransform,
):
    """Delegate to PixelTransform (same as training / predict_yield on .npy)."""
    raw = np.asarray(x, dtype=np.float32)
    x, mask, doy, weight = pixel_transform.transform_chunk(raw[np.newaxis, :])
    return x[0], mask[0], doy[0], weight[0]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Generate yield forecasts per talhão (sum of pixels inside plot polygon)."
    )
    p.add_argument("--checkpoint", type=Path, required=True, help="model_best.pth")
    p.add_argument(
        "--datapath",
        type=Path,
        default=None,
        help="NPY root (default: training/config.json)",
    )
    p.add_argument(
        "--tiffpath",
        type=Path,
        default=Path("files/daily_tiff"),
        help="Daily TIFF root (see create_pixel_heatmap / preprocessing/clip_tiffs_to_shapefiles layout)",
    )
    p.add_argument(
        "--year-range",
        type=str,
        required=True,
        metavar="YYYY-YYYY",
        help="Season folder, e.g. 2020-2021 (must match .npy and daily TIFFs)",
    )
    p.add_argument(
        "--talhoes-gpkg",
        type=Path,
        default=None,
        help=(
            "Talhão polygons: .gpkg or .geojson (default: benchmark/talhoes_baseline.gpkg "
            "if present, else benchmark/talhoes_baseline.geojson)"
        ),
    )
    p.add_argument(
        "--talhoes-layer",
        type=str,
        default="talhoes_baseline",
        help="Layer name inside a GeoPackage (ignored for .geojson)",
    )
    p.add_argument(
        "--baseline-csv",
        type=Path,
        default=here / "benchmark" / "produtividade_AgroIA_baseline.csv",
        help="AgroIA baseline CSV (max prod per harvest cycle for metrics)",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: {run_dir}/predictions/talhoes_forecasts.csv)",
    )
    p.add_argument(
        "-seq",
        "--sequencelength",
        type=int,
        default=None,
        help="Max time-series length (default: training/config.json)",
    )
    p.add_argument(
        "--chunk-size", type=int, default=2000, help="Pixels per forward pass"
    )
    p.add_argument("-d", "--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--feature-layout",
        type=str,
        default=None,
        metavar="NAME",
        help="Override feature layout from training/config.json",
    )
    p.add_argument(
        "--municipalities",
        type=str,
        nargs="*",
        default=None,
        help="Optional IBGE filter; default = every talhão in --talhoes-gpkg / baseline GeoJSON",
    )
    p.add_argument(
        "--reproject-tiffs",
        action="store_true",
        help="Warp each daily TIFF when building the valid-pixel mask (slow; use if fast mask misaligns).",
    )
    p.add_argument(
        "--no-mask-cache",
        action="store_true",
        help="Do not read/write {muni}.keep_mask.npy next to the municipal .npy",
    )
    args = p.parse_args()

    if args.talhoes_gpkg is None:
        gpkg = here / "benchmark" / "talhoes_baseline.gpkg"
        geojson = here / "benchmark" / "talhoes_baseline.geojson"
        args.talhoes_gpkg = gpkg if gpkg.is_file() else geojson

    args.checkpoint = resolve_checkpoint_path(args.checkpoint)
    run_config = load_latest_run_config(args.checkpoint)
    apply_run_config_to_args(args, run_config)
    if args.feature_layout is not None:
        normalize_feature_layout(args.feature_layout)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args, run_config


def resolve_muni_npy(
    datapath: Path, municipality_code: str, year_range: str
) -> Path | None:
    """Find municipality .npy under datapath (flat or year-range subfolder)."""
    candidates = [
        datapath / municipality_code / f"{municipality_code}.npy",
        datapath / year_range / municipality_code / f"{municipality_code}.npy",
        datapath / f"{municipality_code}.npy",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def load_talhoes(talhoes_path: Path, layer: str) -> gpd.GeoDataFrame:
    """Load talhão polygons from GeoPackage or GeoJSON (benchmark shapes)."""
    if not talhoes_path.is_file():
        raise FileNotFoundError(f"Talhões file not found: {talhoes_path}")

    if talhoes_path.suffix.lower() in (".geojson", ".json"):
        gdf = gpd.read_file(talhoes_path)
    else:
        gdf = gpd.read_file(talhoes_path, layer=layer)

    for col in ("talhao_id", "geo_cod", "geometry"):
        if col not in gdf.columns:
            raise ValueError(f"{talhoes_path.name} missing column {col!r}")

    if "talhao_key" not in gdf.columns:
        gdf["talhao_key"] = (
            gdf["talhao_id"].astype(str) + "_" + gdf["geo_cod"].astype(str)
        )

    return gdf


def harvest_year_from_year_range(year_range: str) -> int:
    """Harvest year = second year in a season folder label (e.g. 2020-2021 → 2021)."""
    parts = year_range.strip().split("-")
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[1])
    raise ValueError(
        f"Could not parse harvest year from --year-range={year_range!r} (expected like 2020-2021)"
    )


def season_bounds_for_harvest(harvest_year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Oct 1 (planting year) through Sep 30 (harvest year), matching YYYY-YYYY folders."""
    return (
        pd.Timestamp(date(harvest_year - 1, 10, 1)),
        pd.Timestamp(date(harvest_year, 9, 30)),
    )


def load_baseline_agg(baseline_csv: Path, year_range: str) -> pd.DataFrame:
    """Max AgroIA prod per talhão within the harvest cycle for year_range."""
    df = pd.read_csv(baseline_csv)
    if "fid" in df.columns and "talhao_id" not in df.columns:
        df = df.rename(columns={"fid": "talhao_id"})
    df["talhao_key"] = df["talhao_id"].astype(str) + "_" + df["geo_cod"].astype(str)
    df["day"] = pd.to_datetime(df["day"])

    harvest_year = harvest_year_from_year_range(year_range)
    start, end = season_bounds_for_harvest(harvest_year)
    df = df[(df["day"] >= start) & (df["day"] <= end)]
    if df.empty:
        raise ValueError(
            f"No baseline rows for harvest {harvest_year} ({start.date()} – {end.date()})"
        )

    agg: dict = {"prod": ["max", "count"]}
    if "lai" in df.columns:
        agg["lai"] = ["max"]
    out = df.groupby(["talhao_id", "geo_cod", "talhao_key"], as_index=False).agg(agg)
    out.columns = [
        c if not isinstance(c, tuple) else "_".join(str(x) for x in c).strip("_")
        for c in out.columns
    ]
    rename = {
        "prod_max": "baseline_prod_max",
        "prod_count": "baseline_n_obs",
        "lai_max": "baseline_lai_max",
    }
    out["harvest_year"] = harvest_year
    return out.rename(columns={k: v for k, v in rename.items() if k in out.columns})


def _run_model_chunk(model, chunk, device) -> np.ndarray:
    chunk_x = torch.stack([p[0] for p in chunk])
    chunk_mask = torch.stack([p[1] for p in chunk])
    chunk_doy = torch.stack([p[2] for p in chunk])
    chunk_weight = torch.stack([p[3] for p in chunk])
    batch = recursive_todevice((chunk_x, chunk_mask, chunk_doy, chunk_weight), device)
    ctx = (
        autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.no_grad()
    )
    with ctx:
        preds = model(batch)
    if preds.dim() == 1:
        preds = preds.unsqueeze(0)
    # autocast may yield bfloat16; NumPy cannot represent that dtype
    return preds.detach().float().cpu().numpy().astype(np.float64).ravel()


def predict_pixel_grid_daily(
    model,
    municipality_code: str,
    npy_path: Path,
    daily_tiffs: list[Path],
    *,
    pixel_transform: PixelTransform,
    chunk_size: int,
    device: torch.device,
    talhao_mask: np.ndarray | None = None,
    talhao_only: bool = True,
    reproject_tiffs: bool = False,
    use_mask_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, object, object] | None:
    """
    Returns (pred_grid, keep_mask, transform, crs).
    pred_grid shape (H, W): per-pixel prediction; NaN outside inferred pixels.

    When talhao_only=True (default), the model runs only on keep_mask pixels that
    also fall inside talhao_mask. .npy row order is preserved by advancing the index
    for every keep_mask pixel.
    """
    keep_mask = resolve_keep_mask(
        daily_tiffs,
        npy_path,
        reproject=reproject_tiffs,
        use_cache=use_mask_cache,
    )
    if keep_mask is None:
        return None

    ref_transform, ref_crs, height, width = _reference_grid_from_first_tiff(daily_tiffs)
    if ref_transform is None:
        return None

    if talhao_only and talhao_mask is not None:
        infer_mask = keep_mask & talhao_mask
        n_infer = int(infer_mask.sum())
        n_keep = int(keep_mask.sum())
        print(
            f"  {municipality_code}: infer {n_infer:,} / {n_keep:,} valid pixels "
            f"({100.0 * n_infer / max(n_keep, 1):.1f}% of municipality)"
        )
    else:
        infer_mask = keep_mask

    try:
        municipality_data = np.load(npy_path, mmap_mode="r")
    except OSError:
        return None
    if len(municipality_data) == 0:
        return None

    transform = ref_transform
    crs = ref_crs

    pred_grid = np.full((height, width), np.nan, dtype=np.float64)
    model.eval()
    npy_pixel_idx = 0
    current_chunk: list = []
    chunk_rc: list[tuple[int, int]] = []

    def flush_chunk() -> None:
        nonlocal current_chunk, chunk_rc
        if not current_chunk:
            return
        batch = current_chunk
        if len(batch) == 1:
            batch = batch + batch
            take = slice(0, 1)
        else:
            take = slice(None)
        preds = _run_model_chunk(model, batch, device)[take]
        for (r, c), p in zip(chunk_rc, preds):
            pred_grid[r, c] = float(p)
        current_chunk = []
        chunk_rc = []

    with torch.no_grad():
        for row in range(height):
            for col in range(width):
                if not keep_mask[row, col]:
                    continue
                if npy_pixel_idx >= len(municipality_data):
                    break
                if infer_mask[row, col]:
                    X = municipality_data[npy_pixel_idx]
                    tup = transform_pixel(X, pixel_transform)
                    current_chunk.append(tup)
                    chunk_rc.append((row, col))
                    if len(current_chunk) >= chunk_size:
                        flush_chunk()
                npy_pixel_idx += 1
            if npy_pixel_idx >= len(municipality_data):
                break
        flush_chunk()

    return pred_grid, keep_mask, transform, crs


def forecast_for_talhao(
    geometry,
    pred_grid: np.ndarray,
    keep_mask: np.ndarray,
    transform,
    aggregation: str = "sum",
) -> tuple[float, int]:
    """Aggregate pixel predictions inside the talhão polygon (sum or mean)."""
    inside = features.geometry_mask(
        [geometry],
        out_shape=pred_grid.shape,
        transform=transform,
        invert=True,
    )
    valid = inside & keep_mask
    vals = pred_grid[valid]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), 0
    if aggregation == "mean":
        return float(vals.mean()), int(vals.size)
    return float(vals.sum()), int(vals.size)


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    sequencelength: int,
    feature_layout: str,
):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state"]
    input_dim = stnet_regression_input_dim_from_state_dict(state_dict)
    resolved_layout = normalize_feature_layout(feature_layout)
    expected = feature_layout_input_dim(resolved_layout)
    if input_dim != expected:
        raise ValueError(
            f"Checkpoint input_dim={input_dim} vs feature_layout {resolved_layout!r} ({expected})"
        )
    ck_fl = checkpoint.get("feature_layout")
    if ck_fl is not None and normalize_feature_layout(ck_fl) != resolved_layout:
        raise ValueError(
            f"Checkpoint feature_layout={ck_fl!r} does not match config {resolved_layout!r}"
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
    _, _, aggregation, _ = resolve_inference_target(run_config, checkpoint)
    return model, input_dim, resolved_layout, aggregation


def run_talhao_predictions(
    *,
    checkpoint: Path | None,
    datapath: Path,
    tiffpath: Path,
    year_range: str,
    talhoes_path: Path,
    talhoes_layer: str = "talhoes_baseline",
    baseline_csv: Path | None = None,
    output_csv: Path | None = None,
    sequencelength: int,
    chunk_size: int = 2000,
    device: torch.device | str | None = None,
    seed: int = 42,
    feature_layout: str | None = None,
    municipalities: list[str] | None = None,
    reproject_tiffs: bool = False,
    use_mask_cache: bool = True,
    rc: bool = False,
    interp: bool = False,
    model=None,
    aggregation: str | None = None,
    run_config: dict | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Predict yield per talhão for one harvest season and optionally write a CSV.

    When ``model`` is provided, ``checkpoint`` is only used for run metadata
    (target unit, aggregation) if ``aggregation`` is omitted.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    if checkpoint is not None:
        checkpoint = resolve_checkpoint_path(checkpoint)
    if run_config is None and checkpoint is not None:
        run_config = load_latest_run_config(checkpoint)

    np.random.seed(seed)
    torch.manual_seed(seed)

    if verbose:
        print(f"Loading talhões from {talhoes_path} (layer={talhoes_layer})...")
    talhoes = load_talhoes(talhoes_path, talhoes_layer)
    if verbose:
        print(f"  {len(talhoes)} talhões")

    baseline_agg = None
    if baseline_csv is not None and baseline_csv.is_file():
        harvest_year = harvest_year_from_year_range(year_range)
        if verbose:
            print(
                f"Loading baseline max prod for harvest {harvest_year} "
                f"from {baseline_csv}..."
            )
        baseline_agg = load_baseline_agg(baseline_csv, year_range)

    if municipalities:
        codes = {str(c) for c in municipalities}
        talhoes = talhoes[talhoes["geo_cod"].astype(str).isin(codes)]
        if verbose:
            print(f"  filtered to {len(talhoes)} talhões in municipalities {sorted(codes)}")

    if model is None:
        if checkpoint is None:
            raise ValueError("Either model or checkpoint must be provided")
        if feature_layout is None:
            raise ValueError("feature_layout is required when model is not provided")
        model, input_dim, feature_layout, aggregation = load_model(
            checkpoint, device, sequencelength, feature_layout
        )
    else:
        if feature_layout is None:
            raise ValueError("feature_layout is required when passing a pre-loaded model")
        feature_layout = normalize_feature_layout(feature_layout)
        if aggregation is None:
            if run_config is None or checkpoint is None:
                raise ValueError(
                    "aggregation or (checkpoint + run_config) required with pre-loaded model"
                )
            ck = torch.load(checkpoint, map_location=device, weights_only=False)
            _, _, aggregation, _ = resolve_inference_target(run_config, ck)
        input_dim = stnet_regression_input_dim_from_state_dict(
            model.state_dict()
            if not hasattr(model, "_orig_mod")
            else model._orig_mod.state_dict()
        )

    if run_config is not None and checkpoint is not None:
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        target, _, _, target_unit = resolve_inference_target(run_config, ck)
    else:
        target, target_unit = "production", "t"

    if verbose:
        print(
            f"Model ready (input_dim={input_dim}, pixel aggregation={aggregation}, "
            f"target={target}, unit={target_unit})"
        )
        print("Mode: talhão-only (infer pixels inside plot polygons only)")

    pixel_transform = PixelTransform(
        sequencelength,
        feature_layout,
        randomchoice=rc,
        interp=interp,
        seed=seed,
    )

    results: list[dict] = []
    failed: list[str] = []

    muni_groups = talhoes.groupby(talhoes["geo_cod"].astype(str))
    for geo_cod, group in tqdm(
        muni_groups, desc="Talhões", unit="muni", disable=not verbose
    ):
        npy_path = resolve_muni_npy(datapath, geo_cod, year_range)
        if npy_path is None:
            for _, row in group.iterrows():
                failed.append(str(row["talhao_key"]))
            continue

        daily_dir = resolve_daily_tiff_dir(tiffpath, geo_cod, year_range)
        if daily_dir is None:
            for _, row in group.iterrows():
                failed.append(str(row["talhao_key"]))
            continue

        dated = []
        for p in sorted(daily_dir.glob("*.tif*")):
            code, d = parse_daily_filename(p.name)
            if code == geo_cod and d is not None:
                dated.append((d, p))
        dated.sort(key=lambda x: x[0])
        daily_tiffs = [p for _, p in dated]
        if not daily_tiffs:
            for _, row in group.iterrows():
                failed.append(str(row["talhao_key"]))
            continue

        ref_transform, ref_crs, height, width = _reference_grid_from_first_tiff(
            daily_tiffs
        )
        if ref_transform is None:
            for _, row in group.iterrows():
                failed.append(str(row["talhao_key"]))
            continue

        sub = group.to_crs(ref_crs) if group.crs != ref_crs else group
        talhao_union = build_talhao_union_mask(
            sub.geometry.tolist(),
            ref_transform,
            (height, width),
        )

        grid_out = predict_pixel_grid_daily(
            model,
            geo_cod,
            npy_path,
            daily_tiffs,
            pixel_transform=pixel_transform,
            chunk_size=chunk_size,
            device=device,
            talhao_mask=talhao_union,
            talhao_only=True,
            reproject_tiffs=reproject_tiffs,
            use_mask_cache=use_mask_cache,
        )
        if grid_out is None:
            for _, row in group.iterrows():
                failed.append(str(row["talhao_key"]))
            continue

        pred_grid, keep_mask, transform, crs = grid_out

        for _, row in sub.iterrows():
            forecast, n_pixels = forecast_for_talhao(
                row.geometry, pred_grid, keep_mask, transform, aggregation
            )
            rec = {
                "talhao_key": row["talhao_key"],
                "talhao_id": int(row["talhao_id"]),
                "geo_cod": int(row["geo_cod"]),
                "nome_municipio": row.get("nome_municipio"),
                "fid_mask": row.get("fid_mask"),
                "forecast": forecast,
                "n_pixels": n_pixels,
                "area_ha": row.get("area_ha"),
            }
            results.append(rec)

    if not results:
        if verbose:
            print("No forecasts generated.")
            if failed:
                print(f"Failed talhões: {len(failed)}")
        return pd.DataFrame()

    out = pd.DataFrame(results)
    if baseline_agg is not None:
        merge_cols = ["talhao_id", "geo_cod", "talhao_key"]
        extra = [c for c in baseline_agg.columns if c not in out.columns]
        out = out.merge(baseline_agg[merge_cols + extra], on=merge_cols, how="left")

    if "baseline_prod_max" in out.columns:
        area = out["area_ha"]
        valid_area = area.notna() & (area > 0)
        has_fc = out["forecast"].notna()
        out["baseline_t_ha"] = out["baseline_prod_max"] / 1000.0
        if aggregation == "mean":
            out["forecast_t_ha"] = out["forecast"]
            out["baseline_tons"] = np.where(
                valid_area & out["baseline_prod_max"].notna(),
                out["baseline_t_ha"] * area,
                np.nan,
            )
            out["error_t_ha"] = out["forecast_t_ha"] - out["baseline_t_ha"]
            out["error"] = np.where(
                valid_area & has_fc & out["baseline_tons"].notna(),
                out["forecast_t_ha"] * area - out["baseline_tons"],
                np.nan,
            )
        else:
            out["baseline_tons"] = np.where(
                valid_area & out["baseline_prod_max"].notna(),
                out["baseline_prod_max"] * area / 1000.0,
                np.nan,
            )
            out["forecast_t_ha"] = np.where(
                valid_area & has_fc,
                out["forecast"] / area,
                np.nan,
            )
            out["error"] = out["forecast"] - out["baseline_tons"]
            out["error_t_ha"] = out["forecast_t_ha"] - out["baseline_t_ha"]
        out["abs_error"] = out["error"].abs()
        out["abs_error_t_ha"] = out["error_t_ha"].abs()

    out = out.sort_values(["geo_cod", "talhao_id"])
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_csv, index=False)
        if verbose:
            print(f"\nSaved {len(out)} forecasts to {output_csv}")

    if verbose:
        valid = out["forecast"].notna()
        if valid.any():
            print(
                f"  forecast range: {out.loc[valid, 'forecast'].min():.2f} – "
                f"{out.loc[valid, 'forecast'].max():.2f}"
            )

        if "baseline_prod_max" in out.columns:
            m_ha = (
                out["forecast_t_ha"].notna()
                & out["baseline_t_ha"].notna()
                & (out["n_pixels"] > 0)
            )
            if m_ha.any():
                sub_ha = out.loc[m_ha]
                metrics_ha = regression_metrics(
                    sub_ha["forecast_t_ha"].values,
                    sub_ha["baseline_t_ha"].values,
                )
                print(
                    f"\nMetrics vs AgroIA baseline ({m_ha.sum()} talhões, produtividade t/ha):"
                )
                print(f"  RMSE: {metrics_ha['rmse']:.2f}")
                print(f"  MAE:  {metrics_ha['mae']:.2f}")
                print(f"  R²:   {metrics_ha['r2']:.4f}")

            if aggregation != "mean":
                m = (
                    out["forecast"].notna()
                    & out["baseline_tons"].notna()
                    & (out["n_pixels"] > 0)
                )
                if m.any():
                    sub = out.loc[m]
                    metrics_t = regression_metrics(
                        sub["forecast"].values,
                        sub["baseline_tons"].values,
                    )
                    print(
                        f"\nMetrics vs AgroIA baseline ({m.sum()} talhões, produção total t):"
                    )
                    print(f"  RMSE: {metrics_t['rmse']:.2f}")
                    print(f"  MAE:  {metrics_t['mae']:.2f}")
                    print(f"  R²:   {metrics_t['r2']:.4f}")

        if failed:
            print(f"\nFailed/skipped {len(failed)} talhões (missing .npy or TIFFs)")

    return out


def main() -> None:
    args, run_config = parse_args()
    print(
        f"Run config: {run_config['config_path']} (session {run_config['session_index']})"
    )
    if args.output_csv is None:
        args.output_csv = (
            predictions_dir(run_dir_from_path(args.checkpoint), create=True)
            / "talhoes_forecasts.csv"
        )
    run_talhao_predictions(
        checkpoint=args.checkpoint,
        datapath=args.datapath,
        tiffpath=args.tiffpath,
        year_range=args.year_range,
        talhoes_path=args.talhoes_gpkg,
        talhoes_layer=args.talhoes_layer,
        baseline_csv=args.baseline_csv,
        output_csv=args.output_csv,
        sequencelength=args.sequencelength,
        chunk_size=args.chunk_size,
        device=args.device,
        seed=args.seed,
        feature_layout=args.feature_layout,
        municipalities=args.municipalities,
        reproject_tiffs=args.reproject_tiffs,
        use_mask_cache=not args.no_mask_cache,
        rc=args.rc,
        interp=args.interp,
        run_config=run_config,
    )


if __name__ == "__main__":
    main()
