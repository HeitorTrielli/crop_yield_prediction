"""
Google Earth Engine helpers for soybean daily imagery: task count, date listing,
daily image build (S2 / S1 / Landsat + MapBiomas soy mask), and export to Drive.

Used by download_soy_gee_drive.py and download_s1_landsat_gee_drive.py.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import ee

# -----------------------------------------------------------------------------
# Constants (match GEE catalog and pipeline no-data convention)
# -----------------------------------------------------------------------------
S2_SR_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SCORE_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
S1_GRD_COLLECTION = "COPERNICUS/S1_GRD"
LANDSAT8_SR_COLLECTION = "LANDSAT/LC08/C02/T1_L2"
LANDSAT9_SR_COLLECTION = "LANDSAT/LC09/C02/T1_L2"
MAPBIOMAS_ASSET = (
    "projects/mapbiomas-public/assets/brazil/lulc/collection10/"
    "mapbiomas_brazil_collection10_integration_v2"
)
SOY_MAPBIOMAS_CODE = 39
BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S1_BANDS = ["VV", "VH"]
LANDSAT_BANDS = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
NO_DATA_VALUE = -9999
# Landsat Collection 2 Level-2 surface reflectance scale/offset (USGS).
LANDSAT_SR_SCALE = 0.0000275
LANDSAT_SR_OFFSET = -0.2


def get_active_gee_task_count() -> int:
    """Count RUNNING + READY export tasks."""
    try:
        task_list = ee.data.getTaskList()
        active = [t for t in task_list if t.get("state") in ("RUNNING", "READY")]
        return len(active)
    except Exception:
        return 0


def _dates_from_collection_timestamps(
    timestamps: list,
    start_date: str,
    end_date: str,
) -> list[str]:
    """Convert GEE system:time_start millis to sorted YYYY-MM-DD within [start, end]."""
    if not timestamps:
        return []
    dates_set = set()
    for ms in timestamps:
        d = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        dates_set.add(d.strftime("%Y-%m-%d"))
    start_d = datetime.strptime(start_date, "%Y-%m-%d")
    end_d = datetime.strptime(end_date, "%Y-%m-%d")
    return [
        d for d in sorted(dates_set) if start_d <= datetime.strptime(d, "%Y-%m-%d") <= end_d
    ]


def _apply_soy_mask_and_doy(
    img: ee.Image,
    ee_geometry: ee.Geometry,
    date_str: str,
    season_year: int,
) -> ee.Image:
    """Add DOY band and apply MapBiomas soy mask; return float image."""
    doy = ee.Date(date_str).getRelative("day", "year")
    img = img.addBands(ee.Image.constant(doy).rename("doy").toFloat())

    mapbiomas = ee.Image(MAPBIOMAS_ASSET)
    classification = mapbiomas.select(f"classification_{season_year}").clip(ee_geometry)
    soy_mask = classification.eq(SOY_MAPBIOMAS_CODE)
    img = img.updateMask(soy_mask).multiply(ee.Image.constant(1).updateMask(soy_mask))
    return img.toFloat()


def _filter_s1_collection(
    ee_geometry: ee.Geometry,
    start_date: str,
    end_exclusive: str,
    orbit_pass: Optional[str] = None,
) -> ee.ImageCollection:
    """IW dual-pol GRD collection filtered by bounds/date (optional ASC/DESC)."""
    s1 = (
        ee.ImageCollection(S1_GRD_COLLECTION)
        .filterBounds(ee_geometry)
        .filterDate(start_date, end_exclusive)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )
    if orbit_pass:
        s1 = s1.filter(ee.Filter.eq("orbitProperties_pass", orbit_pass.upper()))
    return s1.select(S1_BANDS)


def _filter_landsat_collection(
    ee_geometry: ee.Geometry,
    start_date: str,
    end_exclusive: str,
    cloud_pct: float,
) -> ee.ImageCollection:
    """Merged L8+L9 Collection 2 Tier-1 L2, filtered by cloud cover metadata."""
    cloud_filter = ee.Filter.lt("CLOUD_COVER", cloud_pct)
    l8 = (
        ee.ImageCollection(LANDSAT8_SR_COLLECTION)
        .filterBounds(ee_geometry)
        .filterDate(start_date, end_exclusive)
        .filter(cloud_filter)
    )
    l9 = (
        ee.ImageCollection(LANDSAT9_SR_COLLECTION)
        .filterBounds(ee_geometry)
        .filterDate(start_date, end_exclusive)
        .filter(cloud_filter)
    )
    return l8.merge(l9)


def _mask_and_scale_landsat(img: ee.Image) -> ee.Image:
    """Mask cloud/shadow via QA_PIXEL and apply C2 L2 SR scale/offset."""
    qa = img.select("QA_PIXEL")
    # Bits 3=cloud, 4=cloud shadow (also drop dilated cloud / fill).
    mask = (
        qa.bitwiseAnd(1 << 0)
        .eq(0)
        .And(qa.bitwiseAnd(1 << 1).eq(0))
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
    )
    optical = (
        img.select(LANDSAT_BANDS)
        .multiply(LANDSAT_SR_SCALE)
        .add(LANDSAT_SR_OFFSET)
    )
    return optical.updateMask(mask).copyProperties(img, ["system:time_start"])


def get_dates_with_s2_scenes(
    ee_geometry: ee.Geometry,
    start_date: str,
    end_date: str,
    cloud_pct: float,
) -> list[str]:
    """
    Return sorted list of YYYY-MM-DD dates that have at least one S2 scene
    (bounds + date range + metadata cloud filter). One GEE getInfo() call.
    """
    end_exclusive = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )
    s2 = (
        ee.ImageCollection(S2_SR_COLLECTION)
        .filterBounds(ee_geometry)
        .filterDate(start_date, end_exclusive)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
    )
    timestamps = s2.aggregate_array("system:time_start").getInfo()
    return _dates_from_collection_timestamps(timestamps, start_date, end_date)


def get_dates_with_s1_scenes(
    ee_geometry: ee.Geometry,
    start_date: str,
    end_date: str,
    orbit_pass: Optional[str] = None,
) -> list[str]:
    """
    Return sorted YYYY-MM-DD dates with at least one Sentinel-1 IW VV+VH scene.
    """
    end_exclusive = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )
    s1 = _filter_s1_collection(ee_geometry, start_date, end_exclusive, orbit_pass)
    timestamps = s1.aggregate_array("system:time_start").getInfo()
    return _dates_from_collection_timestamps(timestamps, start_date, end_date)


def get_dates_with_landsat_scenes(
    ee_geometry: ee.Geometry,
    start_date: str,
    end_date: str,
    cloud_pct: float,
) -> list[str]:
    """
    Return sorted YYYY-MM-DD dates with at least one Landsat 8/9 C2 L2 scene
    under the metadata cloud cover filter.
    """
    end_exclusive = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )
    landsat = _filter_landsat_collection(
        ee_geometry, start_date, end_exclusive, cloud_pct
    )
    timestamps = landsat.aggregate_array("system:time_start").getInfo()
    return _dates_from_collection_timestamps(timestamps, start_date, end_date)


def build_daily_image(
    ee_geometry: ee.Geometry,
    date_str: str,
    season_year: int,
    cloud_pct: float,
    cloud_score_threshold: float,
) -> Optional[ee.Image]:
    """
    Build a single-day Sentinel-2 image: pre-filter by cloud %, link Cloud Score+,
    mask clear pixels, apply MapBiomas soy mask. Returns None if no imagery.
    """
    d = datetime.strptime(date_str, "%Y-%m-%d")
    end_date = (d + timedelta(days=1)).strftime("%Y-%m-%d")

    s2 = (
        ee.ImageCollection(S2_SR_COLLECTION)
        .filterBounds(ee_geometry)
        .filterDate(date_str, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
    )
    cs_plus = ee.ImageCollection(CLOUD_SCORE_COLLECTION)

    s2_with_cs = s2.linkCollection(cs_plus, ["cs"])

    def mask_clouds(img):
        return img.updateMask(img.select("cs").gte(cloud_score_threshold))

    s2_masked = s2_with_cs.map(mask_clouds)

    count = s2_masked.size()
    n = count.getInfo()
    if n is None or n == 0:
        return None

    img = s2_masked.median().clip(ee_geometry)
    img = img.select(BANDS)
    return _apply_soy_mask_and_doy(img, ee_geometry, date_str, season_year)


def build_daily_s1_image(
    ee_geometry: ee.Geometry,
    date_str: str,
    season_year: int,
    orbit_pass: Optional[str] = None,
) -> Optional[ee.Image]:
    """
    Build a single-day Sentinel-1 IW GRD image (VV/VH median), MapBiomas soy mask.
    Returns None if no imagery. Values remain in dB as provided by S1_GRD.
    """
    d = datetime.strptime(date_str, "%Y-%m-%d")
    end_date = (d + timedelta(days=1)).strftime("%Y-%m-%d")

    s1 = _filter_s1_collection(ee_geometry, date_str, end_date, orbit_pass)
    n = s1.size().getInfo()
    if n is None or n == 0:
        return None

    img = s1.median().clip(ee_geometry).select(S1_BANDS)
    return _apply_soy_mask_and_doy(img, ee_geometry, date_str, season_year)


def build_daily_landsat_image(
    ee_geometry: ee.Geometry,
    date_str: str,
    season_year: int,
    cloud_pct: float,
) -> Optional[ee.Image]:
    """
    Build a single-day Landsat 8/9 C2 L2 image: QA cloud mask, SR scale/offset,
    median composite, MapBiomas soy mask. Returns None if no imagery.
    """
    d = datetime.strptime(date_str, "%Y-%m-%d")
    end_date = (d + timedelta(days=1)).strftime("%Y-%m-%d")

    landsat = _filter_landsat_collection(
        ee_geometry, date_str, end_date, cloud_pct
    ).map(_mask_and_scale_landsat)

    n = landsat.size().getInfo()
    if n is None or n == 0:
        return None

    img = landsat.median().clip(ee_geometry).select(LANDSAT_BANDS)
    return _apply_soy_mask_and_doy(img, ee_geometry, date_str, season_year)


def get_aligned_crs_transform(
    scale: int,
    crs: str = "EPSG:4326",
) -> list[float]:
    """
    Return a global crsTransform for ``crs`` at ``scale`` meters/pixel.

    Uses ``ee.Projection(crs).atScale(scale)`` so every export that shares
    (crs, scale) lands on the same pixel grid (S2 / S1 / Landsat).
    """
    proj_info = ee.Projection(crs).atScale(scale).getInfo()
    transform = proj_info.get("transform")
    if not transform or len(transform) != 6:
        raise RuntimeError(
            f"Could not read crsTransform from Projection.atScale({scale}): {proj_info}"
        )
    return list(transform)


def crs_transform_from_reference_tiff(path) -> tuple[str, list[float], tuple[int, int]]:
    """
    Read (crs, crsTransform, (width, height)) from an existing GeoTIFF
    (e.g. a Sentinel-2 export) so another sensor can match that grid exactly.
    """
    try:
        import rasterio
    except ImportError as e:
        raise ImportError(
            "rasterio is required for --s2-reference alignment"
        ) from e

    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Reference TIFF has no CRS: {path}")
        t = src.transform
        crs = src.crs.to_string()
        # GEE crsTransform: [xScale, xShear, xOrigin, yShear, yScale, yOrigin]
        crs_transform = [t.a, t.b, t.c, t.d, t.e, t.f]
        return crs, crs_transform, (src.width, src.height)


def start_export_task(
    image: ee.Image,
    description: str,
    folder_name: str,
    region: ee.Geometry,
    scale: int,
    crs: str = "EPSG:4326",
    crs_transform: Optional[list[float]] = None,
    dimensions: Optional[tuple[int, int]] = None,
) -> str:
    """
    Start Export.image.toDrive; return task id from task list.

    If ``crs_transform`` is set, pixels are forced onto that grid (scale is not
    passed to Export — the transform defines pixel size). Otherwise Export uses
    ``scale`` + ``crs`` + ``region`` (GEE-fitted grid; may differ across sensors).
    """
    image = image.unmask(NO_DATA_VALUE)
    task_config = {
        "image": image,
        "description": description,
        "folder": folder_name,
        "fileNamePrefix": description,
        "region": region,
        "crs": crs,
        "fileFormat": "GeoTIFF",
        "maxPixels": 1e13,
        "formatOptions": {"noData": NO_DATA_VALUE},
    }
    if crs_transform is not None:
        # Snap image samples onto the shared grid before export.
        image = image.reproject(crs=crs, crsTransform=crs_transform)
        task_config["image"] = image
        task_config["crsTransform"] = crs_transform
        if dimensions is not None:
            # GEE wants "widthxheight" string or a list; list [w, h] works in Python API.
            task_config["dimensions"] = [int(dimensions[0]), int(dimensions[1])]
    else:
        task_config["scale"] = scale
    task = ee.batch.Export.image.toDrive(**task_config)
    task.start()
    time.sleep(1)
    for t in ee.data.getTaskList():
        if t.get("state") not in ("RUNNING", "READY"):
            continue
        desc = t.get("description") or (t.get("config") or {}).get("description")
        if desc == description:
            return t["id"]
    raise RuntimeError(
        f"Started export '{description}' but task not found in task list"
    )
