#!/usr/bin/env python3
"""
Predict yield at talhão (plot) level by summing pixel predictions inside each plot polygon.

Uses the same STNet checkpoint and .npy inputs as predict_yield.py, but assigns pixels to
talhões via daily municipal TIFF georeferencing (same row-major order as preprocess_daily_to_npy).

Typical usage (all talhões in benchmark shapes, Seed6007):

  python scripts/run_benchmark_talhoes_predictions.py

Or call directly (no --municipalities = all talhões in the GPKG/GeoJSON):

  python predict_yield_talhoes.py --checkpoint results/Yield_STNetRegression_Pad_Hy_2020_2022_2023_Seed6007/model_best.pth --datapath files/npy --tiffpath files/daily_tiff --year-range 2020-2021 --sequencelength 45

Polygons: benchmark/talhoes_baseline.geojson (or .gpkg). Output CSV defaults to
{run_dir}/predictions/talhoes_forecasts.csv with columns forecast, n_pixels, baseline_prod_max, error.
"""

from __future__ import annotations

import argparse
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

import importlib.util

from datasets.pixel_transform import PixelTransform
from models import STNetRegression
from run_paths import predictions_dir, run_dir_from_path
from utils_aggregated import regression_metrics, stnet_regression_input_dim_from_state_dict


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


def compute_daily_valid_pixel_mask(daily_tiffs: list[Path]):
    """(keep_mask, height, width) — same logic as preprocess_daily_to_npy / create_pixel_heatmap."""
    if not daily_tiffs:
        return None, None, None
    with rasterio.open(daily_tiffs[0]) as src0:
        height, width = src0.height, src0.width
        ref_transform = src0.transform
        ref_crs = src0.crs
        if ref_crs is None:
            return None, None, None
    any_data = np.zeros((height, width), dtype=bool)
    days_with_data = np.zeros((height, width), dtype=np.uint16)
    for p in daily_tiffs:
        try:
            data = _read_bands_on_reference_grid(
                p, ref_transform, ref_crs, height, width
            )
        except OSError:
            continue
        day_has = np.any(np.isfinite(data) & (data != 0), axis=0)
        any_data |= day_has
        days_with_data += day_has.astype(np.uint16)
    return any_data & (days_with_data > 2), height, width


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
    return pixel_transform.transform_chunk(raw[np.newaxis, :])[0]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Generate yield forecasts per talhão (sum of pixels inside plot polygon)."
    )
    p.add_argument("--checkpoint", type=Path, required=True, help="model_best.pth")
    p.add_argument(
        "--datapath",
        type=Path,
        default=Path("files/npy"),
        help="Root with {municipality}/{municipality}.npy (or {year-range}/{muni}/...)",
    )
    p.add_argument(
        "--tiffpath",
        type=Path,
        default=Path("files/daily_tiff"),
        help="Daily TIFF root (see create_pixel_heatmap / clip_tiffs_to_shapefiles layout)",
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
        "--sequencelength",
        type=int,
        default=45,
        help="Max time-series length (default: 45)",
    )
    p.add_argument("--rc", action="store_true", help="Random choice subsampling")
    p.add_argument("--interp", action="store_true", help="Interpolate time series to sequencelength")
    p.add_argument("--chunk-size", type=int, default=2000, help="Pixels per forward pass")
    p.add_argument("-d", "--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=27)
    p.add_argument(
        "--feature-layout",
        type=str,
        default="auto",
        metavar="NAME",
        help=", ".join(feature_layout_choices()) + ', or "auto"',
    )
    p.add_argument(
        "--municipalities",
        type=str,
        nargs="*",
        default=None,
        help="Optional IBGE filter; default = every talhão in --talhoes-gpkg / baseline GeoJSON",
    )
    args = p.parse_args()

    if args.talhoes_gpkg is None:
        gpkg = here / "benchmark" / "talhoes_baseline.gpkg"
        geojson = here / "benchmark" / "talhoes_baseline.geojson"
        args.talhoes_gpkg = gpkg if gpkg.is_file() else geojson

    if args.feature_layout != "auto":
        normalize_feature_layout(args.feature_layout)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def resolve_muni_npy(datapath: Path, municipality_code: str, year_range: str) -> Path | None:
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
    ctx = autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad()
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
) -> tuple[np.ndarray, np.ndarray, object, object] | None:
    """
    Returns (pred_grid, keep_mask, transform, crs).
    pred_grid shape (H, W): sum-ready per-pixel prediction; NaN outside valid pixels.
    """
    keep_mask, height, width = compute_daily_valid_pixel_mask(daily_tiffs)
    if keep_mask is None:
        return None

    try:
        municipality_data = np.load(npy_path, mmap_mode="r")
    except OSError:
        return None
    if len(municipality_data) == 0:
        return None

    with rasterio.open(daily_tiffs[0]) as src0:
        transform = src0.transform
        crs = src0.crs

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
                X = municipality_data[npy_pixel_idx]
                tup = transform_pixel(X, pixel_transform)
                current_chunk.append(tup)
                chunk_rc.append((row, col))
                npy_pixel_idx += 1
                if len(current_chunk) >= chunk_size:
                    flush_chunk()
            if npy_pixel_idx >= len(municipality_data):
                break
        flush_chunk()

    return pred_grid, keep_mask, transform, crs


def forecast_for_talhao(
    geometry,
    pred_grid: np.ndarray,
    keep_mask: np.ndarray,
    transform,
) -> tuple[float, int]:
    """Sum pixel predictions whose centers fall inside the talhão polygon."""
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
    return float(vals.sum()), int(vals.size)


def load_model(checkpoint_path: Path, device: torch.device, sequencelength: int, feature_layout: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state"]
    input_dim = stnet_regression_input_dim_from_state_dict(state_dict)
    if feature_layout != "auto":
        expected = feature_layout_input_dim(feature_layout)
        if input_dim != expected:
            raise ValueError(
                f"Checkpoint input_dim={input_dim} vs --feature-layout {feature_layout!r} ({expected})"
            )
    model = STNetRegression(
        input_dim=input_dim,
        num_outputs=1,
        max_seq_len=sequencelength,
    ).to(device)
    if hasattr(model, "_orig_mod"):
        model._orig_mod.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    ck_fl = checkpoint.get("feature_layout")
    if feature_layout == "auto":
        if ck_fl is not None:
            resolved_layout = normalize_feature_layout(ck_fl)
        elif input_dim == 12:
            resolved_layout = "spectral_xavier"
        else:
            resolved_layout = "spectral"
    else:
        resolved_layout = normalize_feature_layout(feature_layout)
    return model, input_dim, resolved_layout


def main() -> None:
    args = parse_args()
    if args.output_csv is None:
        args.output_csv = (
            predictions_dir(run_dir_from_path(args.checkpoint), create=True)
            / "talhoes_forecasts.csv"
        )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print(f"Loading talhões from {args.talhoes_gpkg} (layer={args.talhoes_layer})...")
    talhoes = load_talhoes(args.talhoes_gpkg, args.talhoes_layer)
    print(f"  {len(talhoes)} talhões")

    baseline_agg = None
    if args.baseline_csv.is_file():
        harvest_year = harvest_year_from_year_range(args.year_range)
        print(
            f"Loading baseline max prod for harvest {harvest_year} "
            f"from {args.baseline_csv}..."
        )
        baseline_agg = load_baseline_agg(args.baseline_csv, args.year_range)

    if args.municipalities:
        codes = {str(c) for c in args.municipalities}
        talhoes = talhoes[talhoes["geo_cod"].astype(str).isin(codes)]
        print(f"  filtered to {len(talhoes)} talhões in municipalities {sorted(codes)}")

    model, input_dim, feature_layout = load_model(
        args.checkpoint, device, args.sequencelength, args.feature_layout
    )
    print(f"Model loaded (input_dim={input_dim})")
    pixel_transform = PixelTransform(
        args.sequencelength,
        feature_layout,
        randomchoice=args.rc,
        interp=args.interp,
        seed=args.seed,
    )

    results: list[dict] = []
    failed: list[str] = []

    muni_groups = talhoes.groupby(talhoes["geo_cod"].astype(str))
    for geo_cod, group in tqdm(muni_groups, desc="Municipalities", unit="muni"):
        npy_path = resolve_muni_npy(args.datapath, geo_cod, args.year_range)
        if npy_path is None:
            for _, row in group.iterrows():
                failed.append(str(row["talhao_key"]))
            continue

        daily_dir = resolve_daily_tiff_dir(args.tiffpath, geo_cod, args.year_range)
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

        grid_out = predict_pixel_grid_daily(
            model,
            geo_cod,
            npy_path,
            daily_tiffs,
            pixel_transform=pixel_transform,
            chunk_size=args.chunk_size,
            device=device,
        )
        if grid_out is None:
            for _, row in group.iterrows():
                failed.append(str(row["talhao_key"]))
            continue

        pred_grid, keep_mask, transform, crs = grid_out
        sub = group.to_crs(crs) if group.crs != crs else group

        for _, row in sub.iterrows():
            forecast, n_pixels = forecast_for_talhao(
                row.geometry, pred_grid, keep_mask, transform
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
        print("No forecasts generated.")
        if failed:
            print(f"Failed talhões: {len(failed)}")
        return

    out = pd.DataFrame(results)
    if baseline_agg is not None:
        merge_cols = ["talhao_id", "geo_cod", "talhao_key"]
        extra = [c for c in baseline_agg.columns if c not in out.columns]
        out = out.merge(baseline_agg[merge_cols + extra], on=merge_cols, how="left")

    if "baseline_prod_max" in out.columns:
        out["error"] = out["forecast"] - out["baseline_prod_max"]
        out["abs_error"] = out["error"].abs()

    out = out.sort_values(["geo_cod", "talhao_id"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"\nSaved {len(out)} forecasts to {args.output_csv}")
    valid = out["forecast"].notna()
    print(
        f"  forecast range: {out.loc[valid, 'forecast'].min():.2f} – "
        f"{out.loc[valid, 'forecast'].max():.2f}"
    )

    if "baseline_prod_max" in out.columns:
        m = out["forecast"].notna() & out["baseline_prod_max"].notna()
        if m.any():
            metrics = regression_metrics(
                out.loc[m, "forecast"].values,
                out.loc[m, "baseline_prod_max"].values,
            )
            print(f"\nMetrics vs baseline_prod_max ({m.sum()} talhões):")
            print(f"  RMSE: {metrics['rmse']:.2f}")
            print(f"  MAE:  {metrics['mae']:.2f}")
            print(f"  R²:   {metrics['r2']:.4f}")

    if failed:
        print(f"\nFailed/skipped {len(failed)} talhões (missing .npy or TIFFs)")


if __name__ == "__main__":
    main()
