#!/usr/bin/env python3
"""
Municipal soy zonal MEAN from Brazil Data Cube (no Earth Engine).

STAC search + windowed COG reads (same pattern as GAIA ``bdc_downloader.py``),
then local numpy mean over MapBiomas soy × Sen2Cor SCL-clear pixels.

Writes the same CSV layout as download_soy_gee_zonal_mean.py.

Usage:
  python data_download/download_soy_bdc_zonal_mean.py \\
      --shapefile-dir files/shapefiles --season-years 2020-2024 \\
      --pam-csv files/pam_soy_br_2020_2024.csv \\
      --skip-existing --workers 4
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd

from bdc_zonal import (
    DEFAULT_SCL_CLEAR,
    MAPBIOMAS_COVERAGE_URL,
    S2_L2A_COLLECTION,
    mapbiomas_url,
    search_s2_items,
    soy_pixel_count,
    zonal_mean_for_items,
)

ZONAL_CSV_FIELDS = [
    "system:index",
    "B11",
    "B12",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "clear_fraction",
    "date",
    "doy",
    "mean_cloud_score_soy",
    "municipality_code",
    "season_year",
    "soy_pixel_count",
    "valid_pixel_count",
    ".geo",
]
EMPTY_GEO = '{"type":"MultiPoint","coordinates":[]}'


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_season_years_spec(seasons_spec: str | None, single: int | None) -> list[int]:
    years: list[int] = []
    if seasons_spec:
        for part in seasons_spec.split(","):
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
    if single is not None:
        years.append(int(single))
    out: list[int] = []
    seen: set[int] = set()
    for y in years:
        if y not in seen:
            seen.add(y)
            out.append(y)
    return out


def season_dates(season_year: int) -> tuple[str, str, int, int]:
    start = datetime(season_year - 1, 10, 1)
    end = datetime(season_year, 3, 31)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), start.year, end.year


def season_dir(output_dir: Path, year_start: int, year_end: int) -> Path:
    return output_dir / f"{year_start}-{year_end}"


def load_pam_codes(path: Path, year: int) -> set[str] | None:
    if path is None or not Path(path).is_file():
        return None
    keep: set[str] = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                y = int(float(row["year"]))
                code = f"{int(float(row['municipality_code'])):07d}"
            except (KeyError, TypeError, ValueError):
                continue
            if y != year:
                continue
            planted = produced = 0.0
            try:
                planted = float(row.get("area_planted_ha") or 0)
            except (TypeError, ValueError):
                pass
            try:
                produced = float(row.get("production_t") or 0)
            except (TypeError, ValueError):
                pass
            if planted > 0 or produced > 0:
                keep.add(code)
    return keep


def load_inventory(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    out: dict[str, int] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = str(row.get("municipality_code") or "").strip()
            try:
                out[code] = int(float(row.get("mapbiomas_soy_pixel_count") or 0))
            except (TypeError, ValueError):
                continue
    return out


def write_zonal_csv(path: Path, rows: list[dict[str, Any]], code: str, season_year: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ZONAL_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for i, row in enumerate(rows):
            out = {k: row.get(k, "") for k in ZONAL_CSV_FIELDS}
            out["system:index"] = str(i)
            out["municipality_code"] = code
            out["season_year"] = season_year
            out[".geo"] = EMPTY_GEO
            w.writerow(out)


def write_area_sidecar(
    path: Path, code: str, season_year: int, year_start: int, year_end: int, soy_px: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "municipality_code",
                "season_year",
                "year_range",
                "mapbiomas_soy_pixel_count",
                "mapbiomas_soy_area_ha",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "municipality_code": code,
                "season_year": season_year,
                "year_range": f"{year_start}-{year_end}",
                "mapbiomas_soy_pixel_count": soy_px,
                "mapbiomas_soy_area_ha": f"{soy_px * 30 * 30 / 10000.0:.4f}",
            }
        )


def read_geometry(shapefile: Path):
    gdf = gpd.read_file(shapefile)
    if gdf.empty:
        return None
    geom_col = gdf.geometry
    valid = geom_col.notna() & ~geom_col.is_empty
    if not bool(valid.any()):
        return None
    gdf = gdf.loc[valid]
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    geom = gdf.geometry.iloc[0]
    if geom is None or geom.is_empty:
        return None
    return geom


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--shapefile", type=Path, action="append", dest="shapefiles")
    g.add_argument("--shapefile-dir", type=Path, dest="shapefile_dir")
    p.add_argument("--season-year", type=int, default=None)
    p.add_argument("--season-years", type=str, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("files/zonal_mean"))
    p.add_argument("--pam-csv", type=Path, default=None)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cloud-pct", type=float, default=20.0)
    p.add_argument(
        "--collection",
        type=str,
        default=S2_L2A_COLLECTION,
        help="BDC STAC collection (default: S2_L2A-1 daily L2A COGs)",
    )
    p.add_argument(
        "--mapbiomas-url-template",
        type=str,
        default=MAPBIOMAS_COVERAGE_URL,
        help="MapBiomas GeoTIFF URL with {year} (Collection 10 coverage)",
    )
    p.add_argument(
        "--mapbiomas-tif",
        type=Path,
        default=None,
        help="Local MapBiomas GeoTIFF instead of the public URL (single year; "
        "only use with --season-year)",
    )
    p.add_argument(
        "--scl-clear",
        type=str,
        default="4,5",
        help="Sen2Cor SCL classes treated as clear (default: 4,5)",
    )
    p.add_argument(
        "--harmonize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Subtract 1000 DN on spectral bands when processing baseline >= 04.00 "
            "(GEE S2_SR_HARMONIZED). Default: on."
        ),
    )
    p.add_argument("--limit", type=int, default=None, help="Max municipalities (debug)")
    args = p.parse_args()
    if args.shapefile_dir is not None:
        d = Path(args.shapefile_dir).resolve()
        if not d.is_dir():
            p.error(f"--shapefile-dir is not a directory: {d}")
        args.shapefiles = []
        for subdir in sorted(d.iterdir()):
            if subdir.is_dir():
                shps = list(subdir.glob("*.shp"))
                if shps:
                    args.shapefiles.append(shps[0])
        if not args.shapefiles:
            p.error(f"No .shp files found under {d}")
    else:
        args.shapefiles = [Path(x).resolve() for x in (args.shapefiles or [])]
    args.season_years_list = parse_season_years_spec(args.season_years, args.season_year)
    if not args.season_years_list:
        p.error("Set --season-year or --season-years")
    args.workers = max(1, int(args.workers))
    args.scl_clear = frozenset(int(x) for x in args.scl_clear.split(",") if x.strip())
    if not args.scl_clear:
        args.scl_clear = DEFAULT_SCL_CLEAR
    return args


def _job(payload: dict[str, Any]) -> dict[str, str]:
    """Process one municipality-season (subprocess-safe)."""
    shapefile = Path(payload["shapefile"])
    code = shapefile.stem
    try:
        geom = read_geometry(shapefile)
        if geom is None:
            return {"municipality_code": code, "status": "no_soy", "note": "empty geometry"}
        soy_px = soy_pixel_count(geom, payload["mapbiomas_path"])
        write_area_sidecar(
            Path(payload["area_path"]),
            code,
            payload["season_year"],
            payload["year_start"],
            payload["year_end"],
            soy_px,
        )
        if soy_px <= 0:
            return {"municipality_code": code, "status": "no_soy", "note": "mapbiomas_soy=0"}
        items = search_s2_items(
            geom,
            payload["start_date"],
            payload["end_date"],
            payload["cloud_pct"],
            collection=payload["collection"],
        )
        if not items:
            return {"municipality_code": code, "status": "no_imagery", "note": "no S2 items"}
        rows = zonal_mean_for_items(
            items,
            geom,
            payload["mapbiomas_path"],
            soy_px,
            scl_clear=frozenset(payload["scl_clear"]),
            harmonize=bool(payload.get("harmonize", True)),
        )
        if not rows:
            return {
                "municipality_code": code,
                "status": "no_imagery",
                "note": "no clear soy dates",
            }
        write_zonal_csv(
            Path(payload["zonal_path"]), rows, code, payload["season_year"]
        )
        return {
            "municipality_code": code,
            "status": "success",
            "note": f"{len(rows)} dates / {len(items)} items",
        }
    except Exception as exc:
        return {"municipality_code": code, "status": "error", "note": str(exc)}


def run_season(args: argparse.Namespace, season_year: int) -> tuple[int, int]:
    start_date, end_date, year_start, year_end = season_dates(season_year)
    shapefiles: list[Path] = list(args.shapefiles)
    pam = load_pam_codes(args.pam_csv, season_year) if args.pam_csv else None
    if pam is not None:

        def _code(stem: str) -> str:
            try:
                return f"{int(stem):07d}"
            except ValueError:
                return stem

        shapefiles = [p for p in shapefiles if _code(p.stem) in pam]
        log(f"PAM {season_year}: {len(shapefiles)} municipalities")

    root = season_dir(args.output_dir, year_start, year_end)
    inventory = load_inventory(root / "mapbiomas_soy_inventory.csv")
    mb_path = (
        str(args.mapbiomas_tif)
        if args.mapbiomas_tif
        else mapbiomas_url(season_year, args.mapbiomas_url_template)
    )

    jobs: list[dict[str, Any]] = []
    summary: list[dict[str, str]] = []
    for shp in shapefiles:
        zonal_path = root / shp.stem / f"{shp.stem}_zonal.csv"
        if args.skip_existing and zonal_path.is_file():
            summary.append(
                {"municipality_code": shp.stem, "status": "skipped", "note": "exists"}
            )
            continue
        if inventory.get(shp.stem, 1) <= 0:
            summary.append(
                {
                    "municipality_code": shp.stem,
                    "status": "no_soy",
                    "note": "inventory soy=0",
                }
            )
            continue
        jobs.append(
            {
                "shapefile": str(shp),
                "mapbiomas_path": mb_path,
                "zonal_path": str(zonal_path),
                "area_path": str(root / shp.stem / f"{shp.stem}_area.csv"),
                "season_year": season_year,
                "year_start": year_start,
                "year_end": year_end,
                "start_date": start_date,
                "end_date": end_date,
                "cloud_pct": args.cloud_pct,
                "collection": args.collection,
                "scl_clear": list(args.scl_clear),
                "harmonize": bool(args.harmonize),
            }
        )
        if args.limit is not None and len(jobs) >= args.limit:
            break

    log(
        f"Season {year_start}-{year_end}: {len(jobs)} to download, "
        f"{sum(1 for r in summary if r['status']=='skipped')} skipped  "
        f"workers={args.workers}  source=BDC {args.collection}"
    )
    if args.workers <= 1:
        for job in jobs:
            row = _job(job)
            log(f"{row['municipality_code']}: {row['status']} ({row['note']})")
            summary.append(row)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_job, job): job for job in jobs}
            for fut in as_completed(futs):
                row = fut.result()
                log(f"{row['municipality_code']}: {row['status']} ({row['note']})")
                summary.append(row)

    summary.sort(key=lambda r: r["municipality_code"])
    summary_path = root / "download_summary_bdc.csv"
    root.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["municipality_code", "status", "note"])
        w.writeheader()
        w.writerows(summary)
    ok = sum(1 for r in summary if r["status"] == "success")
    print(f"Season {year_start}-{year_end}: {ok}/{len(jobs)} succeeded → {summary_path}")
    return ok, len(summary)


def main() -> None:
    args = parse_args()
    for shp in args.shapefiles:
        if not Path(shp).is_file():
            sys.exit(f"Shapefile not found: {shp}")
    print(
        f"BDC zonal mean: {len(args.shapefiles)} shapefiles × "
        f"{args.season_years_list}  workers={args.workers}"
    )
    print(
        "Note: SCL (Sen2Cor) replaces GEE Cloud Score+. "
        "Do not mix BDC and GEE CSVs in the same model without checking."
    )
    grand_ok = grand_n = 0
    for sy in args.season_years_list:
        ok, n = run_season(args, sy)
        grand_ok += ok
        grand_n += n
    print(f"\nDone: {grand_ok}/{grand_n} municipality-seasons succeeded.")


if __name__ == "__main__":
    main()
