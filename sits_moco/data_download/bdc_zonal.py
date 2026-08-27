"""
Local zonal means from Brazil Data Cube Sentinel-2 L2A COGs.

Same idea as Green-AI-Lab-GAIA / scrocha `bdc_downloader.py` (pystac-client +
rasterio), but we never download whole granules: windowed HTTP reads of COGs,
SCL cloud mask, MapBiomas soy, then numpy mean.

CSV columns match the GEE zonal exporter. Cloud Score+ is not on BDC; clear
pixels use Sen2Cor SCL instead (default classes 4 and 5).
"""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import Resampling, reproject
from shapely.geometry import mapping, shape

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff,.TIF")

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
