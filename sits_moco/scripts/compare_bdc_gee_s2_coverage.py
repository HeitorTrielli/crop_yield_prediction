#!/usr/bin/env python3
"""
Catalog coverage: BDC S2_L2A-1 vs GEE COPERNICUS/S2_SR_HARMONIZED.

Uses the same scene-level filters as the downloaders:
  geometry  = shapefile union (not a loose bbox)
  dates     = soy season Oct (Y-1)–Mar Y
  cloud     = BDC eo:cloud_cover < N  and  GEE CLOUDY_PIXEL_PERCENTAGE < N

Compares unique dates (what daily TIFFs mosaic on) and (date, MGRS tile)
granules. Pixel masks still differ (Sen2Cor SCL vs Cloud Score+) and are
not part of this catalog check.

Usage (WSL, sits_moco venv, repo root):
  python scripts/compare_bdc_gee_s2_coverage.py \\
      --shapefile files/rs_state/rs.shp --season-years 2020-2024
  python scripts/compare_bdc_gee_s2_coverage.py \\
      --shapefile files/rs_state/rs.shp --season-years 2023 --list-missing
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ee
import geopandas as gpd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_DL = _ROOT / "data_download"
if str(_DL) not in sys.path:
    sys.path.insert(0, str(_DL))

from bdc_zonal import (  # noqa: E402
    S2_L2A_COLLECTION,
    _item_date,
    get_stac_client,
    item_asset_hrefs,
    search_s2_items,
)
from gee_export import (  # noqa: E402
    S2_SR_COLLECTION,
    _dates_from_collection_timestamps,
    initialize_gee,
)

_TILE_RE = re.compile(r"T(\d{2}[A-Z]{3})")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_season_years(spec: str) -> list[int]:
    years: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if hi < lo:
                lo, hi = hi, lo
            years.extend(range(lo, hi + 1))
        else:
            years.append(int(part))
    out: list[int] = []
    seen: set[int] = set()
    for y in years:
        if y not in seen:
            seen.add(y)
            out.append(y)
    return out


def season_dates(season_year: int) -> tuple[str, str]:
    start = datetime(season_year - 1, 10, 1)
    end = datetime(season_year, 3, 31)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def read_geometry(shapefile: Path):
    gdf = gpd.read_file(shapefile)
    if gdf.empty:
        raise SystemExit(f"Empty shapefile: {shapefile}")
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf.geometry.union_all()


def mgrs_from_bdc_item(item) -> str | None:
    props = item.properties or {}
    for key in ("s2:mgrs_tile", "mgrs:tile", "grid:code"):
        raw = props.get(key)
        if raw:
            return str(raw).lstrip("T").upper()
    zone = props.get("mgrs:utm_zone")
    band = props.get("mgrs:latitude_band")
    square = props.get("mgrs:grid_square")
    if zone is not None and band and square:
        return f"{int(zone):02d}{band}{square}".upper()
    match = _TILE_RE.search(item.id or "")
    return match.group(1).upper() if match else None


def _cloud_stats(values: list[float]) -> str:
    if not values:
        return "n=0"
    return (
        f"n={len(values)} min={min(values):.2f} median="
        f"{statistics.median(values):.2f} max={max(values):.2f}"
    )


def bdc_season(geometry, start: str, end: str, cloud_pct: float | None, client) -> dict:
    if cloud_pct is None:
        from shapely.geometry import mapping
        from bdc_zonal import _as_shapely

        search = client.search(
            collections=[S2_L2A_COLLECTION],
            intersects=mapping(_as_shapely(geometry)),
            datetime=f"{start}/{end}",
        )
        items = list(search.items())
    else:
        items = search_s2_items(
            geometry, start, end, cloud_pct, collection=S2_L2A_COLLECTION, client=client
        )

    dates: set[str] = set()
    usable_dates: set[str] = set()
    tiles: set[tuple[str, str]] = set()
    clouds: list[float] = []
    for item in items:
        date_str = _item_date(item)
        dates.add(date_str)
        if item_asset_hrefs(item):
            usable_dates.add(date_str)
        tile = mgrs_from_bdc_item(item)
        if tile:
            tiles.add((date_str, tile))
        cc = (item.properties or {}).get("eo:cloud_cover")
        if cc is not None:
            try:
                clouds.append(float(cc))
            except (TypeError, ValueError):
                pass
    return {
        "granules": len(items),
        "dates": dates,
        "usable_dates": usable_dates,
        "tiles": tiles,
        "clouds": clouds,
    }


def gee_season(ee_geometry, start: str, end: str, cloud_pct: float | None) -> dict:
    end_exclusive = (
        datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")
    col = (
        ee.ImageCollection(S2_SR_COLLECTION)
        .filterBounds(ee_geometry)
        .filterDate(start, end_exclusive)
    )
    if cloud_pct is not None:
        col = col.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))

    info = ee.Dictionary(
        {
            "n": col.size(),
            "t": col.aggregate_array("system:time_start"),
            "mgrs": col.aggregate_array("MGRS_TILE"),
            "cloud": col.aggregate_array("CLOUDY_PIXEL_PERCENTAGE"),
        }
    ).getInfo()

    timestamps = info.get("t") or []
    dates = set(_dates_from_collection_timestamps(timestamps, start, end))
    tiles: set[tuple[str, str]] = set()
    mgrs = info.get("mgrs") or []
    for ms, tile in zip(timestamps, mgrs):
        d = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
        if tile:
            tiles.add((d, str(tile).lstrip("T").upper()))
    clouds = []
    for v in info.get("cloud") or []:
        try:
            clouds.append(float(v))
        except (TypeError, ValueError):
            pass
    return {
        "granules": int(info.get("n") or 0),
        "dates": dates,
        "tiles": tiles,
        "clouds": clouds,
    }


def fmt_dates(dates: list[str], limit: int) -> str:
    if not dates:
        return "(none)"
    if limit <= 0 or len(dates) <= limit:
        return ", ".join(dates)
    head = ", ".join(dates[:limit])
    return f"{head} ... (+{len(dates) - limit} more)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shapefile", type=Path, required=True)
    p.add_argument("--season-years", type=str, default="2020-2024")
    p.add_argument("--cloud-pct", type=float, default=20.0)
    p.add_argument(
        "--no-cloud-filter",
        action="store_true",
        help="Skip scene cloud metadata (ingest gap only, not pipeline filters)",
    )
    p.add_argument(
        "--list-missing",
        action="store_true",
        help="Print every GEE-only and BDC-only date",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write per-date membership CSV",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    shp = args.shapefile.expanduser()
    if not shp.is_file():
        raise SystemExit(f"Shapefile not found: {shp}")
    years = parse_season_years(args.season_years)
    if not years:
        raise SystemExit("No season years parsed")
    cloud_pct = None if args.no_cloud_filter else float(args.cloud_pct)

    geometry = read_geometry(shp)
    initialize_gee()
    ee_geometry = ee.Geometry(geometry.__geo_interface__)
    client = get_stac_client()

    cloud_label = "none" if cloud_pct is None else f"<{cloud_pct:g}"
    log(
        f"{shp}  seasons={years}  cloud={cloud_label}  "
        f"BDC={S2_L2A_COLLECTION}  GEE={S2_SR_COLLECTION}"
    )
    print(
        f"{'season':>8}  {'BDC gran':>9}  {'GEE gran':>9}  "
        f"{'BDC dates':>10}  {'GEE dates':>10}  {'both':>6}  "
        f"{'GEE only':>8}  {'BDC only':>8}  {'tile GEE-only':>13}",
        flush=True,
    )

    csv_rows: list[dict[str, str]] = []
    all_missing: dict[int, list[str]] = {}

    for year in years:
        start, end = season_dates(year)
        log(f"Season {year}: {start} .. {end}")
        bdc = bdc_season(geometry, start, end, cloud_pct, client)
        gee = gee_season(ee_geometry, start, end, cloud_pct)

        both = sorted(bdc["dates"] & gee["dates"])
        gee_only = sorted(gee["dates"] - bdc["dates"])
        bdc_only = sorted(bdc["dates"] - gee["dates"])
        tile_gee_only = gee["tiles"] - bdc["tiles"]
        all_missing[year] = gee_only

        print(
            f"{year:>8}  {bdc['granules']:>9}  {gee['granules']:>9}  "
            f"{len(bdc['dates']):>10}  {len(gee['dates']):>10}  {len(both):>6}  "
            f"{len(gee_only):>8}  {len(bdc_only):>8}  {len(tile_gee_only):>13}",
            flush=True,
        )
        log(f"  BDC eo:cloud_cover {_cloud_stats(bdc['clouds'])}")
        log(f"  GEE CLOUDY_PIXEL_PERCENTAGE {_cloud_stats(gee['clouds'])}")
        if bdc["usable_dates"] != bdc["dates"]:
            missing_assets = sorted(bdc["dates"] - bdc["usable_dates"])
            log(
                f"  BDC dates missing optical assets: {len(missing_assets)} "
                f"{fmt_dates(missing_assets, 8)}"
            )
        if args.list_missing:
            log(f"  GEE-only dates: {fmt_dates(gee_only, 0)}")
            log(f"  BDC-only dates: {fmt_dates(bdc_only, 0)}")
        else:
            log(f"  GEE-only sample: {fmt_dates(gee_only, 8)}")
            if bdc_only:
                log(f"  BDC-only sample: {fmt_dates(bdc_only, 8)}")

        union_dates = sorted(bdc["dates"] | gee["dates"])
        for d in union_dates:
            csv_rows.append(
                {
                    "season_year": str(year),
                    "date": d,
                    "in_bdc": "1" if d in bdc["dates"] else "0",
                    "in_gee": "1" if d in gee["dates"] else "0",
                }
            )

    if args.csv:
        out = args.csv.expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["season_year", "date", "in_bdc", "in_gee"])
            w.writeheader()
            w.writerows(csv_rows)
        log(f"Wrote {out}")

    print(flush=True)
    log("GEE-only dates are scenes the BDC STAC catalog does not return.")
    log(
        "Daily TIFFs can still drop extra days after this (BDC SCL 4+5 vs "
        "GEE Cloud Score+), which this catalog check cannot see."
    )


if __name__ == "__main__":
    main()
