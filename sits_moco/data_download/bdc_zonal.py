"""
Local zonal means from Brazil Data Cube Sentinel-2 L2A COGs.

Same idea as Green-AI-Lab-GAIA / scrocha `bdc_downloader.py` (pystac-client +
rasterio), but we never download whole granules: windowed HTTP reads of COGs,
SCL cloud mask, MapBiomas soy, then numpy mean.

CSV columns match the GEE zonal exporter. Daily soy-masked GeoTIFFs (same
layout as download_soy_gee_drive.py after merge) are written by
download_soy_bdc_daily_tiff.py. Cloud Score+ is not on BDC; clear pixels use
Sen2Cor SCL instead (default classes 4 and 5).
"""

from __future__ import annotations

import os
import re
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.windows
from rasterio.mask import mask
from rasterio.warp import Resampling, reproject
from shapely.geometry import mapping, shape

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.TIF")


def silence_bdc_numpy_warnings() -> None:
    """Worker processes re-import this module; also call as ProcessPoolExecutor initializer."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*All-NaN.*")
    np.seterr(all="ignore")


silence_bdc_numpy_warnings()

BDC_STAC_URL = "https://data.inpe.br/bdc/stac/v1"
S2_L2A_COLLECTION = "S2_L2A-1"
SOY_MAPBIOMAS_CODE = 39
NO_DATA_VALUE = -9999.0

BAND_ASSETS = {
    "B2": "B02",
    "B3": "B03",
    "B4": "B04",
    "B5": "B05",
    "B6": "B06",
    "B7": "B07",
    "B8": "B08",
    "B8A": "B8A",
    "B11": "B11",
    "B12": "B12",
}

MAPBIOMAS_COVERAGE_URL = (
    "https://storage.googleapis.com/mapbiomas-public/initiatives/brasil/"
    "collection_10/lulc/coverage/brazil_coverage_{year}.tif"
)

DEFAULT_SCL_CLEAR = frozenset({4, 5})

# GEE COPERNICUS/S2_SR_HARMONIZED: after PB 04.00 (2022-01-25) ESA shifted spectral
# DNs by +1000. Harmonized collections subtract that offset so values match older scenes.
GEE_SR_HARMONIZE_OFFSET = 1000.0
GEE_SR_HARMONIZE_BASELINE = 4.0
GEE_SR_HARMONIZE_DATE = "2022-01-25"


def _vsicurl(href: str) -> str:
    if href.startswith("/vsi"):
        return href
    return f"/vsicurl/{href}"


def _as_shapely(geometry):
    from shapely.geometry.base import BaseGeometry

    if isinstance(geometry, BaseGeometry):
        return geometry
    if hasattr(geometry, "__geo_interface__"):
        return shape(geometry.__geo_interface__)
    return shape(geometry)


def get_stac_client(url: str = BDC_STAC_URL):
    from pystac_client import Client

    return Client.open(url)


def search_s2_items(
    geometry,
    start_date: str,
    end_date: str,
    cloud_pct: float,
    collection: str = S2_L2A_COLLECTION,
    client=None,
) -> list:
    """STAC items intersecting ``geometry`` in [start_date, end_date] (inclusive)."""
    if client is None:
        client = get_stac_client()
    geo = mapping(_as_shapely(geometry))
    search = client.search(
        collections=[collection],
        intersects=geo,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": float(cloud_pct)}},
    )
    return list(search.items())


def _item_date(item) -> str:
    raw = item.properties.get("datetime") or item.properties.get("start_datetime")
    if not raw:
        raise ValueError(f"Item {item.id} has no datetime")
    dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


def _item_doy(date_str: str) -> int:
    return int(datetime.strptime(date_str, "%Y-%m-%d").timetuple().tm_yday)


def _parse_processing_baseline(raw: object) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    m = re.match(r"^(\d+)(?:[._](\d+))?$", text)
    if m:
        major, minor = int(m.group(1)), m.group(2)
        try:
            return float(f"{major}.{minor}") if minor is not None else float(major)
        except ValueError:
            return None
    try:
        return float(text.replace("_", "."))
    except ValueError:
        return None


def processing_baseline_from_item(item) -> float | None:
    """ESA processing baseline (e.g. 4.00, 5.09) from STAC properties or granule id."""
    props = getattr(item, "properties", None) or {}
    for key in (
        "s2:processing_baseline",
        "PROCESSING_BASELINE",
        "processing_baseline",
        "processing:version",
    ):
        parsed = _parse_processing_baseline(props.get(key))
        # ESA S2 baselines are ~02.04–05.xx; skip catalog versions like "1.0".
        if parsed is not None and parsed >= 2.0:
            return parsed
    ident = str(getattr(item, "id", "") or props.get("s2:product_uri") or "")
    m = re.search(r"_N(\d{2})(\d{2})(?:_|\.|$)", ident)
    if m:
        return float(f"{int(m.group(1))}.{m.group(2)}")
    return None


def gee_sr_needs_offset(baseline: float | None, date_str: str | None) -> bool:
    """True when GEE S2_SR_HARMONIZED would subtract 1000 from spectral DNs."""
    if baseline is not None:
        return baseline >= GEE_SR_HARMONIZE_BASELINE
    if date_str:
        return date_str >= GEE_SR_HARMONIZE_DATE
    return False


def gee_sr_harmonize_dn(
    values: np.ndarray,
    *,
    baseline: float | None = None,
    date_str: str | None = None,
    enabled: bool = True,
) -> np.ndarray:
    """
    Match GEE ``COPERNICUS/S2_SR_HARMONIZED``: spectral DNs minus 1000 when
    PROCESSING_BASELINE >= 04.00 (else sensing date >= 2022-01-25). SCL/DOY unchanged.
    Negatives are kept (ESA's reason for the offset).
    """
    if not enabled or not gee_sr_needs_offset(baseline, date_str):
        return values
    out = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(out)
    out = out.copy()
    out[finite] = out[finite] - np.float32(GEE_SR_HARMONIZE_OFFSET)
    return out


def mapbiomas_url(year: int, template: str | None = None) -> str:
    return (template or MAPBIOMAS_COVERAGE_URL).format(year=int(year))


def _open_path(path: str):
    if path.startswith("http"):
        return _vsicurl(path)
    return path


def soy_pixel_count(
    geometry,
    mapbiomas_path: str,
    soy_code: int = SOY_MAPBIOMAS_CODE,
) -> int:
    """MapBiomas soy pixel count in the municipality (native 30 m grid)."""
    geom = _as_shapely(geometry)
    gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
    with rasterio.open(_open_path(mapbiomas_path)) as src:
        gdf = gdf.to_crs(src.crs)
        out, _ = mask(src, list(gdf.geometry), crop=True, filled=True, nodata=0)
        return int(np.count_nonzero(out[0] == int(soy_code)))


def soy_pixel_lonlat(
    geometry,
    mapbiomas_path: str,
    soy_code: int = SOY_MAPBIOMAS_CODE,
) -> tuple[np.ndarray, np.ndarray]:
    """WGS84 centroids of MapBiomas soy pixels inside ``geometry``."""
    import pyproj

    geom = _as_shapely(geometry)
    gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
    with rasterio.open(_open_path(mapbiomas_path)) as src:
        gdf = gdf.to_crs(src.crs)
        out, transform = mask(src, list(gdf.geometry), crop=True, filled=True, nodata=0)
        rows, cols = np.where(out[0] == int(soy_code))
        if rows.size == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        crs = src.crs
    if crs is None or str(crs).upper() in ("EPSG:4326", "OGC:CRS84"):
        return xs, ys
    t = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = t.transform(xs, ys)
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def _window_geoms(src: rasterio.DatasetReader, geometry) -> list:
    gdf = gpd.GeoDataFrame(geometry=[_as_shapely(geometry)], crs="EPSG:4326")
    gdf = gdf.to_crs(src.crs)
    return list(gdf.geometry)


def _masked_band(src: rasterio.DatasetReader, geoms: list) -> tuple[np.ndarray, Any]:
    out, transform = mask(
        src, geoms, crop=True, filled=True, nodata=src.nodata if src.nodata is not None else 0
    )
    arr = out[0].astype(np.float32)
    nd = src.nodata
    if nd is not None:
        arr[arr == nd] = np.nan
    arr[arr == 0] = np.nan
    return arr, transform


def _soy_on_s2_grid(
    mapbiomas_path: str,
    dst_shape: tuple[int, int],
    dst_transform,
    dst_crs,
    soy_code: int,
) -> np.ndarray:
    dest = np.zeros(dst_shape, dtype=np.uint8)
    with rasterio.open(_open_path(mapbiomas_path)) as mb:
        reproject(
            source=rasterio.band(mb, 1),
            destination=dest,
            src_transform=mb.transform,
            src_crs=mb.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
    return dest == int(soy_code)


def zonal_mean_for_items(
    items: list,
    geometry,
    mapbiomas_path: str,
    soy_pixel_count_n: int,
    soy_code: int = SOY_MAPBIOMAS_CODE,
    scl_clear: frozenset[int] = DEFAULT_SCL_CLEAR,
    harmonize: bool = True,
) -> list[dict[str, Any]]:
    """
    One row per calendar date: mean of clear soy pixels (DN 0–10000, GEE-like).

    Overlapping tiles the same day: concatenate valid pixels (tile-edge overlap
    may double-count; rare for soy).
    """
    by_date: dict[str, list] = defaultdict(list)
    for item in items:
        by_date[_item_date(item)].append(item)

    rows: list[dict[str, Any]] = []
    for date_str in sorted(by_date):
        band_vals: dict[str, list[np.ndarray]] = {k: [] for k in BAND_ASSETS}
        n_clear = 0
        n_soy_in_s2 = 0
        for item in by_date[date_str]:
            ref_asset = item.assets.get("B04") or item.assets.get("B02")
            scl_asset = item.assets.get("SCL")
            if ref_asset is None or scl_asset is None:
                continue
            try:
                with rasterio.open(_vsicurl(ref_asset.href)) as src:
                    geoms = _window_geoms(src, geometry)
                    _ref, transform = _masked_band(src, geoms)
                    crs = src.crs
                    shape_hw = _ref.shape
                clear = np.zeros(shape_hw, dtype=bool)
                with rasterio.open(_vsicurl(scl_asset.href)) as ssrc:
                    sgeoms = _window_geoms(ssrc, geometry)
                    scl, stf = mask(
                        ssrc,
                        sgeoms,
                        crop=True,
                        filled=True,
                        nodata=ssrc.nodata if ssrc.nodata is not None else 0,
                    )
                    scl_on = np.zeros(shape_hw, dtype=np.uint8)
                    reproject(
                        source=scl[0],
                        destination=scl_on,
                        src_transform=stf,
                        src_crs=ssrc.crs,
                        dst_transform=transform,
                        dst_crs=crs,
                        resampling=Resampling.nearest,
                    )
                    clear = np.isin(scl_on, list(scl_clear))
                soy = _soy_on_s2_grid(
                    mapbiomas_path, shape_hw, transform, crs, soy_code
                )
                valid = soy & clear
                n_soy_in_s2 += int(np.count_nonzero(soy))
                n_clear += int(np.count_nonzero(valid))
                if not np.any(valid):
                    continue
                for csv_name, asset_name in BAND_ASSETS.items():
                    asset = item.assets.get(asset_name)
                    if asset is None:
                        continue
                    with rasterio.open(_vsicurl(asset.href)) as bsrc:
                        g2 = _window_geoms(bsrc, geometry)
                        arr, btf = _masked_band(bsrc, g2)
                        bcrs = bsrc.crs
                    if arr.shape != valid.shape:
                        aligned = np.full(valid.shape, np.nan, dtype=np.float32)
                        reproject(
                            source=arr,
                            destination=aligned,
                            src_transform=btf,
                            src_crs=bcrs,
                            dst_transform=transform,
                            dst_crs=crs,
                            resampling=Resampling.bilinear,
                        )
                        arr = aligned
                    pix = arr[valid]
                    pix = pix[np.isfinite(pix)]
                    if pix.size:
                        pix = gee_sr_harmonize_dn(
                            pix,
                            baseline=processing_baseline_from_item(item),
                            date_str=date_str,
                            enabled=harmonize,
                        )
                        band_vals[csv_name].append(pix)
            except Exception:
                continue

        if n_clear <= 0 or not any(band_vals[k] for k in BAND_ASSETS):
            continue
        row: dict[str, Any] = {
            "date": date_str,
            "doy": float(_item_doy(date_str)),
            "valid_pixel_count": n_clear,
            "soy_pixel_count": soy_pixel_count_n,
            "clear_fraction": (
                float(n_clear) / float(soy_pixel_count_n) if soy_pixel_count_n else 0.0
            ),
            "mean_cloud_score_soy": (
                float(n_clear) / float(n_soy_in_s2) if n_soy_in_s2 else NO_DATA_VALUE
            ),
        }
        for csv_name in BAND_ASSETS:
            chunks = band_vals[csv_name]
            if not chunks:
                row[csv_name] = NO_DATA_VALUE
                continue
            row[csv_name] = float(np.mean(np.concatenate(chunks)))
        rows.append(row)
    return rows


GEE_SPECTRAL_BANDS = list(BAND_ASSETS.keys())
DAILY_TIFF_BANDS = [*GEE_SPECTRAL_BANDS, "doy"]


def item_asset_hrefs(item) -> dict[str, str]:
    """SCL + spectral asset hrefs for a STAC item (skip if SCL or a band is missing)."""
    scl = item.assets.get("SCL")
    if scl is None:
        return {}
    hrefs = {"SCL": scl.href}
    for asset_name in BAND_ASSETS.values():
        asset = item.assets.get(asset_name)
        if asset is None:
            return {}
        hrefs[asset_name] = asset.href
    return hrefs


def group_items_by_date(items: list) -> dict[str, list[dict[str, Any]]]:
    """Pickle-friendly records: date -> [{id, bbox, hrefs}, ...]."""
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        hrefs = item_asset_hrefs(item)
        if not hrefs:
            continue
        bbox = list(item.bbox) if getattr(item, "bbox", None) else None
        by_date[_item_date(item)].append(
            {
                "id": item.id,
                "bbox": bbox,
                "hrefs": hrefs,
                "baseline": processing_baseline_from_item(item),
                "date": _item_date(item),
            }
        )
    return dict(by_date)


def write_mapbiomas_soy_mask(
    geometry,
    mapbiomas_path: str,
    out_path,
    soy_code: int = SOY_MAPBIOMAS_CODE,
) -> dict[str, Any]:
    """Crop MapBiomas to ``geometry`` and write a 0/1 soy mask (destination grid)."""
    from pathlib import Path

    geom = _as_shapely(geometry)
    gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(_open_path(mapbiomas_path)) as src:
        gdf = gdf.to_crs(src.crs)
        cropped, transform = mask(
            src, list(gdf.geometry), crop=True, filled=True, nodata=0
        )
        soy = (cropped[0] == int(soy_code)).astype(np.uint8)
        profile = src.profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "height": soy.shape[0],
                "width": soy.shape[1],
                "count": 1,
                "dtype": "uint8",
                "transform": transform,
                "nodata": 0,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
            }
        )
        for key in ("photometric", "nbits"):
            profile.pop(key, None)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(soy, 1)
        dst.set_band_description(1, "mapbiomas_soy")
    return {
        "path": str(out_path),
        "height": int(soy.shape[0]),
        "width": int(soy.shape[1]),
        "soy_pixels": int(np.count_nonzero(soy)),
    }


def _window_overlaps_bbox(crs, transform, window, bbox: list[float] | None) -> bool:
    if not bbox or len(bbox) < 4:
        return True
    from rasterio.warp import transform_bounds
    from shapely.geometry import box

    bounds = rasterio.windows.bounds(window, transform)
    w, s, e, n = transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=0)
    return box(w, s, e, n).intersects(
        box(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    )


def _median_axis0_skip_nan(cube: np.ndarray) -> np.ndarray:
    """Median over granules (axis 0). All-NaN pixels stay NaN; no RuntimeWarning."""
    masked = np.ma.masked_invalid(cube)
    med = np.ma.median(masked, axis=0)
    return np.asarray(np.ma.filled(med, np.nan), dtype=np.float32)


def _reproject_band(
    href: str,
    dst: np.ndarray,
    dst_transform,
    dst_crs,
    resampling: Resampling,
) -> None:
    with rasterio.open(_vsicurl(href)) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
            src_nodata=src.nodata,
            dst_nodata=np.nan if np.issubdtype(dst.dtype, np.floating) else 0,
        )


def write_daily_soy_tiff(
    date_str: str,
    item_records: list[dict[str, Any]],
    soy_mask_path: str,
    out_path,
    scl_clear: frozenset[int] = DEFAULT_SCL_CLEAR,
    block_size: int = 512,
    nodata: float = NO_DATA_VALUE,
    harmonize: bool = True,
) -> dict[str, Any]:
    """
    Mosaic BDC L2A COGs for one day onto the soy-mask grid.

    Bands match GEE soy TIFFs: B2–B12 plus calendar DOY. Cloudy / non-soy pixels
    are ``nodata`` (-9999). Overlapping granules use a per-pixel median.
    """
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doy = float(_item_doy(date_str))
    n_valid = 0
    with rasterio.open(soy_mask_path) as mask_src:
        height, width = mask_src.height, mask_src.width
        dest_transform = mask_src.transform
        dest_crs = mask_src.crs
        tiled = width >= 16 and height >= 16
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": len(DAILY_TIFF_BANDS),
            "height": height,
            "width": width,
            "crs": dest_crs,
            "transform": dest_transform,
            "nodata": nodata,
            "compress": "lzw",
            "BIGTIFF": "YES",
            "predictor": 2,
        }
        if tiled:
            profile.update(
                {
                    "tiled": True,
                    "blockxsize": max(16, (min(block_size, width) // 16) * 16),
                    "blockysize": max(16, (min(block_size, height) // 16) * 16),
                }
            )
        with rasterio.open(out_path, "w", **profile) as dst:
            for i, name in enumerate(DAILY_TIFF_BANDS, start=1):
                dst.set_band_description(i, name)
            step = max(16, int(block_size))
            for row_off in range(0, height, step):
                for col_off in range(0, width, step):
                    window = rasterio.windows.Window(
                        col_off,
                        row_off,
                        min(step, width - col_off),
                        min(step, height - row_off),
                    )
                    soy = mask_src.read(1, window=window) > 0
                    h, w = soy.shape
                    wtransform = rasterio.windows.transform(window, dest_transform)
                    if not np.any(soy):
                        fill = np.full(
                            (len(DAILY_TIFF_BANDS), h, w), nodata, dtype=np.float32
                        )
                        dst.write(fill, window=window)
                        continue
                    overlapping = [
                        rec
                        for rec in item_records
                        if _window_overlaps_bbox(
                            dest_crs, dest_transform, window, rec.get("bbox")
                        )
                    ]
                    stack: list[np.ndarray] = []
                    for rec in overlapping:
                        hrefs = rec["hrefs"]
                        scl = np.zeros((h, w), dtype=np.float32)
                        try:
                            _reproject_band(
                                hrefs["SCL"], scl, wtransform, dest_crs, Resampling.nearest
                            )
                        except Exception:
                            continue
                        clear = np.isin(
                            np.nan_to_num(scl, nan=0).astype(np.uint8), list(scl_clear)
                        )
                        valid = soy & clear
                        if not np.any(valid):
                            continue
                        bands = np.full(
                            (len(GEE_SPECTRAL_BANDS), h, w), np.nan, dtype=np.float32
                        )
                        ok = True
                        for i, csv_name in enumerate(GEE_SPECTRAL_BANDS):
                            asset = BAND_ASSETS[csv_name]
                            try:
                                _reproject_band(
                                    hrefs[asset],
                                    bands[i],
                                    wtransform,
                                    dest_crs,
                                    Resampling.bilinear,
                                )
                            except Exception:
                                ok = False
                                break
                        if not ok:
                            continue
                        bands = gee_sr_harmonize_dn(
                            bands,
                            baseline=rec.get("baseline"),
                            date_str=rec.get("date") or date_str,
                            enabled=harmonize,
                        )
                        bands[:, ~valid] = np.nan
                        stack.append(bands)
                    out = np.full(
                        (len(DAILY_TIFF_BANDS), h, w), nodata, dtype=np.float32
                    )
                    if stack:
                        cube = np.stack(stack, axis=0)
                        med = _median_axis0_skip_nan(cube)
                        good = np.isfinite(med[0])
                        n_valid += int(np.count_nonzero(good))
                        out[:-1, good] = med[:, good]
                        out[-1, good] = doy
                    dst.write(out, window=window)
    if n_valid <= 0:
        try:
            out_path.unlink(missing_ok=True)
        except Exception:
            pass
        return {
            "date": date_str,
            "status": "no_imagery",
            "note": "no clear soy pixels",
            "path": "",
        }
    return {
        "date": date_str,
        "status": "success",
        "note": f"{n_valid} clear soy pixels / {len(item_records)} granules",
        "path": str(out_path),
    }


def profile_daily_soy_windows(
    date_str: str,
    item_records: list[dict[str, Any]],
    soy_mask_path: str,
    scl_clear: frozenset[int] = DEFAULT_SCL_CLEAR,
    block_size: int = 512,
    max_soy_windows: int = 16,
    harmonize: bool = True,
) -> dict[str, Any]:
    """Time HTTP+reproject+median on the first ``max_soy_windows`` soy tiles (no GeoTIFF)."""
    import time

    t0 = time.perf_counter()
    n_soy_windows = 0
    n_valid = 0
    n_granule_reads = 0
    window_pixels = 0
    with rasterio.open(soy_mask_path) as mask_src:
        height, width = mask_src.height, mask_src.width
        dest_transform = mask_src.transform
        dest_crs = mask_src.crs
        step = max(16, int(block_size))
        done = False
        for row_off in range(0, height, step):
            if done:
                break
            for col_off in range(0, width, step):
                window = rasterio.windows.Window(
                    col_off,
                    row_off,
                    min(step, width - col_off),
                    min(step, height - row_off),
                )
                soy = mask_src.read(1, window=window) > 0
                if not np.any(soy):
                    continue
                h, w = soy.shape
                wtransform = rasterio.windows.transform(window, dest_transform)
                overlapping = [
                    rec
                    for rec in item_records
                    if _window_overlaps_bbox(
                        dest_crs, dest_transform, window, rec.get("bbox")
                    )
                ]
                stack: list[np.ndarray] = []
                for rec in overlapping:
                    hrefs = rec["hrefs"]
                    scl = np.zeros((h, w), dtype=np.float32)
                    try:
                        _reproject_band(
                            hrefs["SCL"], scl, wtransform, dest_crs, Resampling.nearest
                        )
                    except Exception:
                        continue
                    clear = np.isin(
                        np.nan_to_num(scl, nan=0).astype(np.uint8), list(scl_clear)
                    )
                    valid = soy & clear
                    if not np.any(valid):
                        continue
                    bands = np.full(
                        (len(GEE_SPECTRAL_BANDS), h, w), np.nan, dtype=np.float32
                    )
                    ok = True
                    for i, csv_name in enumerate(GEE_SPECTRAL_BANDS):
                        asset = BAND_ASSETS[csv_name]
                        try:
                            _reproject_band(
                                hrefs[asset],
                                bands[i],
                                wtransform,
                                dest_crs,
                                Resampling.bilinear,
                            )
                        except Exception:
                            ok = False
                            break
                    if not ok:
                        continue
                    n_granule_reads += 1
                    bands = gee_sr_harmonize_dn(
                        bands,
                        baseline=rec.get("baseline"),
                        date_str=rec.get("date") or date_str,
                        enabled=harmonize,
                    )
                    bands[:, ~valid] = np.nan
                    stack.append(bands)
                if stack:
                    cube = np.stack(stack, axis=0)
                    med = _median_axis0_skip_nan(cube)
                    n_valid += int(np.count_nonzero(np.isfinite(med[0])))
                n_soy_windows += 1
                window_pixels += int(h * w)
                if n_soy_windows >= int(max_soy_windows):
                    done = True
                    break
    elapsed = time.perf_counter() - t0
    mpix = window_pixels / 1e6 if window_pixels else 0.0
    return {
        "date": date_str,
        "status": "success" if n_soy_windows else "no_imagery",
        "seconds": round(elapsed, 2),
        "soy_windows": n_soy_windows,
        "window_mpix": round(mpix, 3),
        "sec_per_mpix": round(elapsed / mpix, 2) if mpix else None,
        "valid_pixels": n_valid,
        "granule_reads": n_granule_reads,
        "granules": len(item_records),
        "block_size": int(block_size),
    }
