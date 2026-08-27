"""
Zonal (mean) statistics over MapBiomas soybean pixels in a municipality.

Builds daily Sentinel-2 composites with the same cloud filters as gee_export.py,
then reduces to one row per date: band means, valid-pixel count, cloud score on soy.

Chunked ``reduceRegions`` (many municipalities, one daily mosaic) is the default
export path. Per-municipality ``reduceRegion`` helpers remain for debugging.

Used by download_soy_gee_zonal_mean.py.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import ee

from gee_export import (
    BANDS,
    CLOUD_SCORE_COLLECTION,
    MAPBIOMAS_ASSET,
    NO_DATA_VALUE,
    S2_SR_COLLECTION,
    SOY_MAPBIOMAS_CODE,
    get_dates_with_s2_scenes,
)


def mapbiomas_soy_mask(
    ee_geometry: ee.Geometry,
    season_year: int,
) -> ee.Image:
    """Binary mask: 1 where MapBiomas class == soybean for ``season_year``."""
    mapbiomas = ee.Image(MAPBIOMAS_ASSET)
    classification = mapbiomas.select(f"classification_{season_year}").clip(ee_geometry)
    return classification.eq(SOY_MAPBIOMAS_CODE).rename("soy")


def mapbiomas_soy_image(season_year: int) -> ee.Image:
    """Unclipped MapBiomas soybean binary (1 = soy) for ``season_year``."""
    return (
        ee.Image(MAPBIOMAS_ASSET)
        .select(f"classification_{season_year}")
        .eq(SOY_MAPBIOMAS_CODE)
        .rename("soy")
    )


def mapbiomas_soy_area_ha(
    ee_geometry: ee.Geometry,
    season_year: int,
    scale: int = 30,
) -> dict[str, float | int]:
    """
    Soybean area (ha) and pixel count from MapBiomas within ``ee_geometry``.

    Uses ``scale`` m pixels (default 30, same as S2 export grid).
    """
    soy = mapbiomas_soy_mask(ee_geometry, season_year)
    stats = soy.rename("soy").reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=ee_geometry,
        scale=scale,
        maxPixels=int(1e13),
        tileScale=4,
    )
    info = stats.getInfo() or {}
    soy_pixels = int(info.get("soy", 0) or 0)
    area_ha = float(soy_pixels * scale * scale / 10000.0)
    return {
        "mapbiomas_soy_pixel_count": soy_pixels,
        "mapbiomas_soy_area_ha": area_ha,
    }


def _soy_count_from_props(props: dict) -> int:
    for key in ("soy", "sum", "soy_sum"):
        if key in props and props[key] is not None:
            return int(float(props[key]))
    return 0


def mapbiomas_soy_areas_for_collection(
    features: ee.FeatureCollection,
    season_year: int,
    scale: int = 30,
) -> list[dict[str, float | int | str]]:
    """
    Pixel count + area (ha) per feature via one ``reduceRegions`` getInfo.

    Features must have ``municipality_code``. Geometries are dropped in the
    response so the payload stays small.
    """
    soy = mapbiomas_soy_image(season_year)
    reduced = soy.reduceRegions(
        collection=features,
        reducer=ee.Reducer.sum(),
        scale=scale,
        tileScale=4,
    ).map(lambda f: ee.Feature(None, f.toDictionary()))
    info = reduced.getInfo() or {}
    rows: list[dict[str, float | int | str]] = []
    for feat in info.get("features") or []:
        props = feat.get("properties") or {}
        code = str(props.get("municipality_code", "")).strip()
        pixels = _soy_count_from_props(props)
        rows.append(
            {
                "municipality_code": code,
                "mapbiomas_soy_pixel_count": pixels,
                "mapbiomas_soy_area_ha": float(pixels * scale * scale / 10000.0),
            }
        )
    return rows


def wait_for_gee_task(task_id: str, poll_interval: int = 15) -> str:
    """Block until export ``task_id`` finishes. Returns terminal state."""
    while True:
        statuses = ee.data.getTaskStatus(task_id)
        status = statuses[0] if statuses else {}
        state = str(status.get("state") or "")
        if state in ("COMPLETED", "SUCCEEDED", "FAILED", "CANCELLED"):
            if state == "FAILED":
                err = status.get("error_message") or "unknown error"
                raise RuntimeError(f"GEE export failed ({task_id}): {err}")
            if state == "CANCELLED":
                raise RuntimeError(f"GEE export cancelled ({task_id})")
            return state
        time.sleep(max(5, int(poll_interval)))


def _ee_or_default(value, default) -> ee.ComputedObject:
    """GEE Dictionary.set rejects null; coalesce reduceRegion outputs."""
    return ee.List([value, default]).reduce(ee.Reducer.firstNonNull())


def _mean_count_reducer() -> ee.Reducer:
    return ee.Reducer.mean().combine(reducer2=ee.Reducer.count(), sharedInputs=True)


def daily_s2_stack(
    region: ee.Geometry,
    date: ee.Date,
    season_year: int,
    cloud_pct: float,
    cloud_score_threshold: float,
) -> ee.Image:
    """Daily S2 median + soy mask + DOY + Cloud Score+ on soy (same as pixel pipeline)."""
    end_date = date.advance(1, "day")
    doy = date.getRelative("day", "year")
    s2 = (
        ee.ImageCollection(S2_SR_COLLECTION)
        .filterBounds(region)
        .filterDate(date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
    )
    cs_plus = ee.ImageCollection(CLOUD_SCORE_COLLECTION)
    s2_with_cs = s2.linkCollection(cs_plus, ["cs"])
    soy = mapbiomas_soy_image(season_year)

    def mask_clear(img: ee.Image) -> ee.Image:
        return img.updateMask(img.select("cs").gte(cloud_score_threshold))

    s2_clear = s2_with_cs.map(mask_clear)
    composite = s2_clear.median().clip(region).select(BANDS).updateMask(soy)
    cs_on_soy = (
        s2_with_cs.median().select("cs").clip(region).updateMask(soy).rename("cs_soy")
    )
    doy_img = (
        ee.Image.constant(doy)
        .rename("doy")
        .toFloat()
        .updateMask(composite.select(BANDS[0]).mask())
    )
    return composite.addBands(doy_img).addBands(cs_on_soy)


def _format_zonal_feature(
    feat: ee.Feature,
    date_str: ee.ComputedObject,
    season_year: int,
) -> ee.Feature:
    """Turn reduceRegions mean/count properties into the on-disk CSV schema."""
    valid = ee.Number(_ee_or_default(feat.get(f"{BANDS[0]}_count"), 0))
    soy_pixel_count = ee.Number(_ee_or_default(feat.get("soy_pixel_count"), 0))
    clear_fraction = ee.Number(
        ee.Algorithms.If(
            soy_pixel_count.gt(0),
            valid.divide(soy_pixel_count),
            0,
        )
    )
    props = ee.Dictionary(
        {
            "date": date_str,
            "municipality_code": feat.get("municipality_code"),
            "season_year": season_year,
            "valid_pixel_count": valid,
            "clear_fraction": clear_fraction,
            "soy_pixel_count": soy_pixel_count,
            "mean_cloud_score_soy": _ee_or_default(
                feat.get("cs_soy_mean"), NO_DATA_VALUE
            ),
            "doy": _ee_or_default(feat.get("doy_mean"), NO_DATA_VALUE),
        }
    )
    for b in BANDS:
        props = props.set(b, _ee_or_default(feat.get(f"{b}_mean"), NO_DATA_VALUE))
    return ee.Feature(None, props)


def s2_coverage_date_list(
    region: ee.Geometry,
    start_date: str,
    end_date: str,
    cloud_pct: float,
) -> ee.List:
    """Unique YYYY-MM-dd strings with an S2 scene under the metadata cloud filter."""
    end_exclusive = (
        datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")
    s2 = (
        ee.ImageCollection(S2_SR_COLLECTION)
        .filterBounds(region)
        .filterDate(start_date, end_exclusive)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .map(
            lambda img: img.set(
                "date",
                ee.Date(img.get("system:time_start")).format("YYYY-MM-dd"),
            )
        )
    )
    return s2.distinct("date").aggregate_array("date")


def chunk_zonal_feature_collection(
    features: ee.FeatureCollection,
    start_date: str,
    end_date: str,
    season_year: int,
    cloud_pct: float,
    cloud_score_threshold: float,
    scale: int,
) -> ee.FeatureCollection:
    """
    One FeatureCollection: (municipality × date) rows via reduceRegions.

    Features must have ``municipality_code`` and ``soy_pixel_count``. Dates are
    taken from S2 scenes over the chunk union (no client-side getInfo).
    """
    region = features.geometry()
    date_list = s2_coverage_date_list(region, start_date, end_date, cloud_pct)

    def map_one_date(date_str: ee.ComputedObject) -> ee.FeatureCollection:
        date_str = ee.String(date_str)
        stack = daily_s2_stack(
            region,
            ee.Date(date_str),
            season_year,
            cloud_pct,
            cloud_score_threshold,
        )
        reduced = stack.reduceRegions(
            collection=features,
            reducer=_mean_count_reducer(),
            scale=scale,
            tileScale=4,
            maxPixelsPerRegion=int(1e13),
        )
        return reduced.map(
            lambda f: _format_zonal_feature(f, date_str, season_year)
        )

    def accumulate(date_str: ee.ComputedObject, acc: ee.ComputedObject) -> ee.FeatureCollection:
        return ee.FeatureCollection(acc).merge(map_one_date(date_str))

    merged = ee.FeatureCollection(
        date_list.iterate(accumulate, ee.FeatureCollection([]))
    )
    return merged.filter(ee.Filter.gt("valid_pixel_count", 0))


def daily_zonal_feature_ee(
    ee_geometry: ee.Geometry,
    date: ee.Date,
    municipality_code: str,
    season_year: int,
    cloud_pct: float,
    cloud_score_threshold: float,
    scale: int,
    soy_pixel_count: int,
) -> ee.Feature:
    """One table row: mean reflectance + quality for a single date (server-side)."""
    stack = daily_s2_stack(
        ee_geometry, date, season_year, cloud_pct, cloud_score_threshold
    )
    stats = stack.reduceRegion(
        reducer=_mean_count_reducer(),
        geometry=ee_geometry,
        scale=scale,
        maxPixels=int(1e13),
        tileScale=4,
    )
    feat = (
        ee.Feature(None, stats)
        .set("municipality_code", municipality_code)
        .set("soy_pixel_count", soy_pixel_count)
    )
    return _format_zonal_feature(feat, date.format("YYYY-MM-dd"), season_year)


def season_zonal_feature_collection(
    ee_geometry: ee.Geometry,
    dates: list[str],
    municipality_code: str,
    season_year: int,
    cloud_pct: float,
    cloud_score_threshold: float,
    scale: int,
    soy_pixel_count: int,
) -> ee.FeatureCollection:
    """FeatureCollection with one feature per date (server-side map)."""
    millis = [
        int(
            datetime.strptime(d, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp()
            * 1000
        )
        for d in dates
    ]
    date_list = ee.List(millis)

    def _map_ms(ms: ee.Number) -> ee.Feature:
        d = ee.Date(ms)
        return daily_zonal_feature_ee(
            ee_geometry,
            d,
            municipality_code,
            season_year,
            cloud_pct,
            cloud_score_threshold,
            scale,
            soy_pixel_count,
        )

    # Drop dates with S2 coverage but no clear soy pixels after Cloud Score+.
    return ee.FeatureCollection(date_list.map(_map_ms)).filter(
        ee.Filter.gt("valid_pixel_count", 0)
    )


def start_table_export_task(
    collection: ee.FeatureCollection,
    description: str,
    folder_name: str,
    file_prefix: str,
    selectors: list[str] | None = None,
) -> str:
    """Export FeatureCollection to Drive as CSV; return task id."""
    kwargs: dict = {
        "collection": collection,
        "description": description,
        "folder": folder_name,
        "fileNamePrefix": file_prefix,
        "fileFormat": "CSV",
    }
    if selectors:
        kwargs["selectors"] = selectors
    task = ee.batch.Export.table.toDrive(**kwargs)
    task.start()
    tid = getattr(task, "id", None)
    if not tid:
        status = task.status() or {}
        tid = status.get("id")
    if not tid:
        raise RuntimeError(
            f"Started table export '{description}' but task id is missing"
        )
    return str(tid)


def dates_with_s2_coverage(
    ee_geometry: ee.Geometry,
    start_date: str,
    end_date: str,
    cloud_pct: float,
) -> list[str]:
    """Dates with at least one S2 scene under metadata cloud filter."""
    return get_dates_with_s2_scenes(ee_geometry, start_date, end_date, cloud_pct)
