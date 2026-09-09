#!/usr/bin/env python3
"""
Clip MapBiomas Solo Collection 3 (beta) soil layers to a region shapefile.

Streams national GeoTIFFs from MapBiomas public GCS via GDAL /vsicurl (no full
Brazil download). Writes local LZW GeoTIFFs under ``files/mapbiomas_solo/`` by
default.

Products (Collection 3):
  - Texture (static): clay / silt / sand % for 10 cm layers; we average
    000_010 + 010_020 + 020_030 → 0–30 cm.
  - SOC stock (annual): carbon 0–30 cm in t/ha for requested years.

Example:
  python data_download/download_mapbiomas_solo.py \\
    --shapefile files/parana_state/state_41.shp \\
    --output-dir files/mapbiomas_solo \\
    --soc-years 2019 2020 2021 2022 2023 2024
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.warp import reproject

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data_download.bdc_zonal import fix_proj_gdal_data_env  # noqa: E402

fix_proj_gdal_data_env()

GCS_ROOT = (
    "https://storage.googleapis.com/shared-development-storage/"
    "COLLECTIONS/BRASIL/SOLO/COLLECTION3"
)
TEXTURE_DEPTHS_0_30 = ("000_010cm", "010_020cm", "020_030cm")
TEXTURE_PROPERTIES = ("clay", "silt", "sand")
DEFAULT_SOC_YEARS = (2019, 2020, 2021, 2022, 2023, 2024)
NODATA = -9999.0


def texture_url(prop: str, depth: str) -> str:
    return (
        f"{GCS_ROOT}/mbsoil_c03_{prop}_fraction_v1/"
        f"mbsoil03-{prop}_fraction_{depth}_v1.tif"
    )


def carbon_url(year: int) -> str:
    return f"{GCS_ROOT}/mbsoil_c03_carbon_v1/mbsoil03-carbon_{int(year)}_v1.tif"


def _vsicurl(href: str) -> str:
    if href.startswith("/vsi"):
        return href
    return f"/vsicurl/{href}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Clip MapBiomas Solo Collection 3 clay/silt/sand (0–30 cm mean) "
            "and annual SOC stock to a shapefile via /vsicurl."
        )
    )
    p.add_argument(
        "--shapefile",
        type=Path,
        default=Path("files/parana_state/state_41.shp"),
        help="Region polygon (default: Paraná state_41.shp)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("files/mapbiomas_solo"),
        help="Output directory for clipped GeoTIFFs",
    )
    p.add_argument(
        "--soc-years",
        type=int,
        nargs="*",
        default=list(DEFAULT_SOC_YEARS),
        help="SOC calendar years to clip (default: 2019–2024)",
    )
    p.add_argument(
        "--skip-texture",
        action="store_true",
        help="Skip clay/silt/sand 0–30 cm products",
    )
    p.add_argument(
        "--skip-soc",
        action="store_true",
        help="Skip annual SOC products",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-write outputs even if they already exist",
    )
    p.add_argument(
        "--region-tag",
        type=str,
        default=None,
        help="Filename prefix (default: inferred from shapefile stem)",
    )
    return p.parse_args()


def _load_geometry(shapefile: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shapefile)
    if gdf.empty:
        raise SystemExit(f"Empty shapefile: {shapefile}")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4674")
    # dissolve multipolygons / multiple features into one clip mask
    if hasattr(gdf, "union_all"):
        geom = gdf.union_all()
    else:
        geom = gdf.unary_union
    return gpd.GeoDataFrame(geometry=[geom], crs=gdf.crs)


def _clip_url_to_array(
    url: str,
    region: gpd.GeoDataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return float32 band + profile for the clipped window.

    Outside-polygon pixels become NaN (MapBiomas Solo texture is often uint8
    with no nodata tag, so we must not fill with -9999 into the source dtype).
    """
    with rasterio.open(_vsicurl(url)) as src:
        gdf = region.to_crs(src.crs)
        data, transform = mask(
            src,
            list(gdf.geometry),
            crop=True,
            filled=False,
            all_touched=False,
        )
        band = data[0]
        arr = band.astype(np.float32)
        # masked array from rasterio.mask(filled=False)
        if np.ma.isMaskedArray(band):
            arr = np.ma.filled(band.astype(np.float32), np.nan)
        src_nodata = src.nodata
        if src_nodata is not None:
            arr = np.where(arr == float(src_nodata), np.nan, arr)
        profile = src.profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "height": int(arr.shape[0]),
                "width": int(arr.shape[1]),
                "count": 1,
                "dtype": "float32",
                "transform": transform,
                "crs": src.crs,
                "nodata": NODATA,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
            }
        )
        for key in ("photometric", "nbits"):
            profile.pop(key, None)
        meta = {
            "crs": str(src.crs),
            "src_dtype": str(src.dtypes[0]),
            "src_nodata": src_nodata,
            "url": url,
        }
    return arr, {**profile, "_meta": meta}


def _write_geotiff(
    path: Path,
    arr: np.ndarray,
    profile: dict[str, Any],
    description: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.where(np.isfinite(arr), arr, NODATA).astype(np.float32)
    write_profile = {k: v for k, v in profile.items() if not k.startswith("_")}
    write_profile.update(
        {
            "count": 1,
            "dtype": "float32",
            "nodata": NODATA,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
        }
    )
    with rasterio.open(path, "w", **write_profile) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, description)


def _reproject_match(
    src_arr: np.ndarray,
    src_profile: dict[str, Any],
    dst_profile: dict[str, Any],
) -> np.ndarray:
    """Warp ``src_arr`` onto ``dst_profile`` grid (nearest / bilinear)."""
    dst = np.full(
        (dst_profile["height"], dst_profile["width"]), np.nan, dtype=np.float32
    )
    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        src_nodata=np.nan,
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    return dst


def clip_texture_0_30(
    prop: str,
    region: gpd.GeoDataFrame,
    out_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    if out_path.exists() and not overwrite:
        print(f"  skip (exists): {out_path.name}")
        return {"path": str(out_path), "skipped": True}

    print(f"  clipping {prop} depths {', '.join(TEXTURE_DEPTHS_0_30)} ...")
    layers: list[np.ndarray] = []
    ref_profile: dict[str, Any] | None = None
    urls: list[str] = []
    for depth in TEXTURE_DEPTHS_0_30:
        url = texture_url(prop, depth)
        urls.append(url)
        print(f"    {depth}")
        arr, profile = _clip_url_to_array(url, region)
        if ref_profile is None:
            ref_profile = profile
            layers.append(arr)
        else:
            if (
                arr.shape != layers[0].shape
                or profile["transform"] != ref_profile["transform"]
            ):
                arr = _reproject_match(arr, profile, ref_profile)
            layers.append(arr)

    assert ref_profile is not None
    stack = np.stack(layers, axis=0)
    with np.errstate(all="ignore"):
        mean = np.nanmean(stack, axis=0).astype(np.float32)
    _write_geotiff(out_path, mean, ref_profile, f"{prop}_fraction_0_30cm_pct")
    valid = int(np.isfinite(mean).sum())
    with np.errstate(all="ignore"):
        vmean = float(np.nanmean(mean)) if valid else None
        vmin = float(np.nanmin(mean)) if valid else None
        vmax = float(np.nanmax(mean)) if valid else None
    print(
        f"  wrote {out_path.name}  shape={mean.shape}  "
        f"valid={valid:,}  mean={vmean if vmean is not None else float('nan'):.2f}"
    )
    return {
        "path": str(out_path),
        "property": prop,
        "depths": list(TEXTURE_DEPTHS_0_30),
        "urls": urls,
        "shape": list(mean.shape),
        "valid_pixels": valid,
        "value_mean": vmean,
        "value_min": vmin,
        "value_max": vmax,
    }


def clip_soc_year(
    year: int,
    region: gpd.GeoDataFrame,
    out_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    if out_path.exists() and not overwrite:
        print(f"  skip (exists): {out_path.name}")
        return {"path": str(out_path), "year": year, "skipped": True}

    url = carbon_url(year)
    print(f"  clipping SOC {year} ...")
    arr, profile = _clip_url_to_array(url, region)
    _write_geotiff(out_path, arr, profile, f"soc_stock_0_30cm_t_ha_{year}")
    valid = int(np.isfinite(arr).sum())
    with np.errstate(all="ignore"):
        vmean = float(np.nanmean(arr)) if valid else None
        vmin = float(np.nanmin(arr)) if valid else None
        vmax = float(np.nanmax(arr)) if valid else None
    print(
        f"  wrote {out_path.name}  shape={arr.shape}  "
        f"valid={valid:,}  mean={vmean if vmean is not None else float('nan'):.2f} t/ha"
    )
    return {
        "path": str(out_path),
        "year": year,
        "url": url,
        "shape": list(arr.shape),
        "valid_pixels": valid,
        "value_mean": vmean,
        "value_min": vmin,
        "value_max": vmax,
    }


def write_multiband_stack(
    band_paths: list[tuple[str, Path]],
    out_path: Path,
    overwrite: bool,
) -> dict[str, Any] | None:
    """Stack existing single-band TIFFs (reproject to first band's grid)."""
    if out_path.exists() and not overwrite:
        print(f"  skip stack (exists): {out_path.name}")
        return {"path": str(out_path), "skipped": True}
    if not band_paths:
        return None

    arrays: list[np.ndarray] = []
    names: list[str] = []
    ref_profile: dict[str, Any] | None = None
    for name, path in band_paths:
        if not path.is_file():
            print(f"  warn: missing band for stack: {path}")
            continue
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                arr = np.where(arr == nodata, np.nan, arr)
            profile = src.profile.copy()
            profile["crs"] = src.crs
        if ref_profile is None:
            ref_profile = profile
            arrays.append(arr)
        else:
            if (
                arr.shape != arrays[0].shape
                or profile["transform"] != ref_profile["transform"]
            ):
                arr = _reproject_match(arr, profile, ref_profile)
            arrays.append(arr)
        names.append(name)

    if ref_profile is None or not arrays:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_profile = {k: v for k, v in ref_profile.items() if k != "crs"}
    write_profile.update(
        {
            "driver": "GTiff",
            "count": len(arrays),
            "dtype": "float32",
            "nodata": NODATA,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
        }
    )
    with rasterio.open(out_path, "w", **write_profile) as dst:
        for i, (name, arr) in enumerate(zip(names, arrays), start=1):
            dst.write(np.where(np.isfinite(arr), arr, NODATA).astype(np.float32), i)
            dst.set_band_description(i, name)
    print(f"  wrote stack {out_path.name}  bands={names}")
    return {"path": str(out_path), "bands": names}


def main() -> None:
    args = parse_args()
    shapefile = args.shapefile.resolve()
    if not shapefile.is_file():
        raise SystemExit(f"Shapefile not found: {shapefile}")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.region_tag or shapefile.stem.replace("state_", "pr_")
    if tag == "pr_41" or tag.endswith("_41"):
        tag = "pr"

    print(f"Region: {shapefile}")
    print(f"Output: {out_dir}")
    print(f"Tag:    {tag}")

    region = _load_geometry(shapefile)
    print(f"CRS: {region.crs}  bounds~{region.total_bounds}")

    manifest: dict[str, Any] = {
        "source": "MapBiomas Solo Collection 3 (beta)",
        "gcs_root": GCS_ROOT,
        "shapefile": str(shapefile),
        "region_tag": tag,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "texture_depths_cm": list(TEXTURE_DEPTHS_0_30),
        "nodata": NODATA,
        "products": {},
    }

    texture_paths: list[tuple[str, Path]] = []
    if not args.skip_texture:
        print("\n=== Texture 0–30 cm (mean of 10 cm layers) ===")
        for prop in TEXTURE_PROPERTIES:
            path = out_dir / f"{tag}_{prop}_0_30cm.tif"
            info = clip_texture_0_30(prop, region, path, args.overwrite)
            manifest["products"][f"{prop}_0_30cm"] = info
            texture_paths.append((f"{prop}_0_30cm_pct", path))

        stack_path = out_dir / f"{tag}_texture_0_30cm.tif"
        stack_info = write_multiband_stack(texture_paths, stack_path, args.overwrite)
        if stack_info is not None:
            manifest["products"]["texture_stack_0_30cm"] = stack_info

    if not args.skip_soc:
        years = sorted(set(int(y) for y in (args.soc_years or [])))
        print(f"\n=== SOC stock 0–30 cm (t/ha), years={years} ===")
        for year in years:
            path = out_dir / f"{tag}_soc_0_30cm_{year}.tif"
            info = clip_soc_year(year, region, path, args.overwrite)
            manifest["products"][f"soc_0_30cm_{year}"] = info

    manifest_path = out_dir / f"{tag}_mapbiomas_solo_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")
    print("Done.")


if __name__ == "__main__":
    main()
