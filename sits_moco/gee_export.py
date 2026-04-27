"""
Google Earth Engine helpers for soybean daily imagery: task count, date listing,
daily image build (S2 + Cloud Score+ + MapBiomas soy mask), and export to Drive.

Used by download_soy_gee_drive.py.
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
MAPBIOMAS_ASSET = (
    "projects/mapbiomas-public/assets/brazil/lulc/collection10/"
    "mapbiomas_brazil_collection10_integration_v2"
)
SOY_MAPBIOMAS_CODE = 39
BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
NO_DATA_VALUE = -9999


def get_active_gee_task_count() -> int:
    """Count RUNNING + READY export tasks."""
    try:
        task_list = ee.data.getTaskList()
        active = [t for t in task_list if t.get("state") in ("RUNNING", "READY")]
        return len(active)
    except Exception:
        return 0


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

    doy = ee.Date(date_str).getRelative("day", "year")
    img = img.addBands(ee.Image.constant(doy).rename("doy").toFloat())

    mapbiomas = ee.Image(MAPBIOMAS_ASSET)
    classification = mapbiomas.select(f"classification_{season_year}").clip(ee_geometry)
    soy_mask = classification.eq(SOY_MAPBIOMAS_CODE)
    img = img.updateMask(soy_mask).multiply(ee.Image.constant(1).updateMask(soy_mask))

    img = img.toFloat()
    return img


def start_export_task(
    image: ee.Image,
    description: str,
    folder_name: str,
    region: ee.Geometry,
    scale: int,
) -> str:
    """Start Export.image.toDrive; return task id from task list."""
    image = image.unmask(NO_DATA_VALUE)
    task_config = {
        "image": image,
        "description": description,
        "folder": folder_name,
        "fileNamePrefix": description,
        "region": region,
        "scale": scale,
        "crs": "EPSG:4326",
        "fileFormat": "GeoTIFF",
        "maxPixels": 1e13,
        "formatOptions": {"noData": NO_DATA_VALUE},
    }
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
