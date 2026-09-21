#!/usr/bin/env python3
"""
Municipal soy zonal MEAN from local BDC state daily GeoTIFFs.

The TIFFs from ``download_soy_bdc_daily_tiff.py`` are already:
  - MapBiomas soy-masked
  - Sen2Cor SCL-clear
  - GEE-harmonized (−1000 DN when PB ≥ 04.00)

Writes the same CSV layout as ``download_soy_gee_zonal_mean.py`` /
``download_soy_bdc_zonal_mean.py``, so ``zonal_csv_to_muni_npy.py`` builds
municipal aggregate .npy the same way Paraná was built from GEE zonal CSVs.

Pipeline per season:
  1. Each daily TIFF → ``{output}/{season}/_by_date/{date}.csv`` (all munis)
  2. Merge date files → ``{output}/{season}/{code}/{code}_zonal.csv``

Example:
  python preprocessing/state_tiff_to_zonal_csv.py \\
      --tiff-dir files/raw_tiff/2022-2023/rs \\
      --shapefile-dir files/shapefiles \\
      --muni-prefix 43 \\
      --output-root files/zonal_mean \\
      --overwrite-csvs -j 4
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "data_download") not in sys.path:
    sys.path.insert(0, str(_REPO / "data_download"))

from bdc_zonal import DAILY_TIFF_BANDS, GEE_SPECTRAL_BANDS, NO_DATA_VALUE  # noqa: E402

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
DATE_RE = re.compile(r"(?P<y>\d{4})[_-](?P<m>\d{2})[_-](?P<d>\d{2})$")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_date_from_tiff(path: Path) -> str | None:
    m = DATE_RE.search(path.stem)
    if not m:
        return None
    return f"{m.group('y')}-{m.group('m')}-{m.group('d')}"


def harvest_year_from_season(season: str) -> int:
    return int(season.split("-")[1])


def calendar_doy(date_str: str) -> int:
    return int(date.fromisoformat(date_str).timetuple().tm_yday)


def collect_municipalities(
    shapefile_dir: Path,
    muni_prefix: str | None,
) -> list[tuple[str, Any]]:
    import geopandas as gpd

    out: list[tuple[str, Any]] = []
    for subdir in sorted(shapefile_dir.iterdir()):
        if not subdir.is_dir():
            continue
        code = subdir.name.split("_")[0]
        if muni_prefix and not code.startswith(muni_prefix):
            continue
        if not (code.isdigit() and len(code) == 7):
            continue
        shps = list(subdir.glob("*.shp"))
        if not shps:
            continue
        gdf = gpd.read_file(shps[0])
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        out.append((code, gdf.geometry.union_all()))
    return out


def _band_index(src) -> dict[str, int]:
    by_name: dict[str, int] = {}
    for i in range(1, src.count + 1):
        desc = (src.descriptions[i - 1] or "").strip()
        if desc:
            by_name[desc] = i
    if all(b in by_name for b in GEE_SPECTRAL_BANDS):
        return by_name
    fallback = {name: i + 1 for i, name in enumerate(GEE_SPECTRAL_BANDS)}
    if src.count >= len(DAILY_TIFF_BANDS):
        fallback["doy"] = len(DAILY_TIFF_BANDS)
    return fallback


def _to_src_geom(geom_wgs84, src_crs):
    import geopandas as gpd

    gdf = gpd.GeoDataFrame([{"g": geom_wgs84}], geometry="g", crs="EPSG:4326")
    return gdf.to_crs(src_crs).geometry.iloc[0]


def soy_counts_for_munis(
    soy_mask_path: Path,
    municipalities: list[tuple[str, Any]],
    all_touched: bool,
) -> dict[str, int]:
    import rasterio
    from rasterio.mask import mask as rio_mask

    counts: dict[str, int] = {}
    with rasterio.open(soy_mask_path) as mask_src:
        for code, geom in municipalities:
            geom_c = _to_src_geom(geom, mask_src.crs)
            geo = geom_c.__geo_interface__
            try:
                out, _ = rio_mask(
                    mask_src, [geo], crop=True, all_touched=all_touched, filled=True
                )
            except ValueError:
                counts[code] = 0
                continue
            counts[code] = int(np.count_nonzero(out[0] > 0))
    return counts


def zonal_one_muni(src, geom_src_crs, band_idx: dict[str, int], all_touched: bool):
    from rasterio.mask import mask as rio_mask

    geo = geom_src_crs.__geo_interface__
    try:
        out_image, _ = rio_mask(
            src, [geo], crop=True, all_touched=all_touched, filled=False
        )
    except ValueError:
        return None

    data = out_image.data if hasattr(out_image, "data") else np.asarray(out_image)
    if data.size == 0:
        return None

    spec_idxs = [band_idx[b] - 1 for b in GEE_SPECTRAL_BANDS if b in band_idx]
    if not spec_idxs:
        return None
    spec = data[spec_idxs].astype(np.float32, copy=False)

    inside = (
        ~out_image.mask[spec_idxs]
        if hasattr(out_image, "mask")
        else np.ones(spec.shape, dtype=bool)
    )
    valid = inside.copy()
    if src.nodata is not None:
        valid &= spec != src.nodata
    valid &= spec != NO_DATA_VALUE
    valid &= np.isfinite(spec)
    valid_pix = np.all(valid, axis=0)
    n_valid = int(np.count_nonzero(valid_pix))
    if n_valid <= 0:
        return None

    means = {
        name: float(np.mean(data[band_idx[name] - 1][valid_pix].astype(np.float64)))
        for name in GEE_SPECTRAL_BANDS
    }
    means["valid_pixel_count"] = float(n_valid)
    return means


def process_one_tiff(payload: dict[str, Any]) -> tuple[str, int, int]:
    """One state TIFF → one staging CSV with all municipality rows for that date."""
    import rasterio

    tiff_path = Path(payload["tiff"])
    date_str = parse_date_from_tiff(tiff_path)
    if not date_str:
        return (tiff_path.name, 0, 0)

    staging = Path(payload["staging_dir"]) / f"{date_str}.csv"
    if payload["skip_existing"] and staging.is_file():
        return (tiff_path.name, 0, 0)

    season_year = int(payload["season_year"])
    doy = calendar_doy(date_str)
    soy_counts: dict[str, int] = payload["soy_counts"]
    munis: list[tuple[str, Any]] = payload["munis"]
    all_touched = bool(payload["all_touched"])

    rows: list[dict[str, Any]] = []
    empty = 0
    # One open per date process. Parallelism is across dates (-j), not munis:
    # reopening multi-GB TIFFs per municipality thrashs /mnt/c.
    with rasterio.open(tiff_path) as src:
        band_idx = _band_index(src)
        for code, geom in munis:
            geom_c = _to_src_geom(geom, src.crs)
            means = zonal_one_muni(src, geom_c, band_idx, all_touched)
            if means is None:
                empty += 1
                continue
            n_valid = int(means["valid_pixel_count"])
            n_soy = int(soy_counts.get(code, n_valid))
            clear_frac = (
                float(n_valid) / float(n_soy) if n_soy > 0 else float(NO_DATA_VALUE)
            )
            rows.append(
                {
                    "system:index": str(len(rows)),
                    "B11": means["B11"],
                    "B12": means["B12"],
                    "B2": means["B2"],
                    "B3": means["B3"],
                    "B4": means["B4"],
                    "B5": means["B5"],
                    "B6": means["B6"],
                    "B7": means["B7"],
                    "B8": means["B8"],
                    "B8A": means["B8A"],
                    "clear_fraction": clear_frac,
                    "date": date_str,
                    "doy": doy,
                    "mean_cloud_score_soy": NO_DATA_VALUE,
                    "municipality_code": code,
                    "season_year": season_year,
                    "soy_pixel_count": n_soy,
                    "valid_pixel_count": n_valid,
                    ".geo": EMPTY_GEO,
                }
            )

    staging.parent.mkdir(parents=True, exist_ok=True)
    tmp = staging.with_suffix(".tmp.csv")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ZONAL_CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(staging)
    return (tiff_path.name, len(rows), empty)


def merge_staging_to_muni_csvs(
    staging_dir: Path,
    output_root: Path,
    season: str,
    codes: list[str],
    overwrite: bool,
) -> int:
    """Concatenate date CSVs into per-municipality zonal CSVs sorted by date."""
    by_code: dict[str, list[dict[str, Any]]] = {c: [] for c in codes}
    for staging in sorted(staging_dir.glob("*.csv")):
        with open(staging, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                code = str(row["municipality_code"]).zfill(7)
                if code in by_code:
                    by_code[code].append(row)

    n_files = 0
    for code, rows in by_code.items():
        if not rows:
            continue
        rows.sort(key=lambda r: r["date"])
        for i, r in enumerate(rows):
            r["system:index"] = str(i)
        out = output_root / season / code / f"{code}_zonal.csv"
        if out.is_file() and not overwrite:
            # Merge with existing, prefer staging for overlapping dates
            existing = []
            with open(out, newline="", encoding="utf-8") as f:
                existing = list(csv.DictReader(f))
            by_date = {r["date"]: r for r in existing}
            for r in rows:
                by_date[r["date"]] = r
            rows = sorted(by_date.values(), key=lambda r: r["date"])
            for i, r in enumerate(rows):
                r["system:index"] = str(i)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ZONAL_CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)
        n_files += 1
    return n_files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tiff-dir", type=Path, required=True)
    p.add_argument(
        "--shapefile-dir",
        type=Path,
        default=_REPO / "files" / "shapefiles",
    )
    p.add_argument(
        "--muni-prefix",
        type=str,
        default="43",
        help="RS=43, PR=41; empty string = all",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=_REPO / "files" / "zonal_mean",
    )
    p.add_argument("--season", type=str, default=None)
    p.add_argument("--soy-mask", type=Path, default=None)
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip dates that already have a staging CSV",
    )
    p.add_argument(
        "--overwrite-csvs",
        action="store_true",
        help="Replace per-muni zonal CSVs from staging (ignore prior CSV content)",
    )
    p.add_argument(
        "-j",
        "--workers",
        type=int,
        default=2,
        help="Parallel daily TIFFs (processes). Default: 2",
    )
    p.add_argument(
        "--muni-workers",
        type=int,
        default=1,
        help="Unused legacy flag (kept for CLI compat). Parallelism is -j dates only.",
    )
    p.add_argument("--limit-dates", type=int, default=None)
    p.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge existing staging CSVs into per-muni files",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tiff_dir = args.tiff_dir.expanduser().resolve()
    season = args.season or tiff_dir.parent.name
    if not re.match(r"^\d{4}-\d{4}$", season):
        raise SystemExit(f"Bad season {season!r}; pass --season YYYY-YYYY")

    out_root = args.output_root.expanduser().resolve()
    staging_dir = out_root / season / "_by_date"
    prefix = args.muni_prefix.strip() or None

    log(f"Loading municipalities from {args.shapefile_dir} prefix={prefix!r}")
    munis = collect_municipalities(args.shapefile_dir.resolve(), prefix)
    if not munis:
        raise SystemExit("No municipalities found")
    codes = [c for c, _ in munis]
    log(f"Municipalities: {len(munis)}")

    if args.merge_only:
        n = merge_staging_to_muni_csvs(
            staging_dir, out_root, season, codes, overwrite=True
        )
        log(f"Merged staging → {n} municipal CSVs under {out_root / season}")
        return

    if not tiff_dir.is_dir():
        raise SystemExit(f"TIFF dir not found: {tiff_dir}")

    soy_mask = args.soy_mask
    if soy_mask is None:
        cand = tiff_dir / "_mapbiomas_soy_mask.tif"
        soy_mask = cand if cand.is_file() else None
    else:
        soy_mask = soy_mask.expanduser().resolve()

    soy_counts: dict[str, int] = {c: 0 for c in codes}
    if soy_mask is not None and Path(soy_mask).is_file():
        log(f"Counting soy pixels from {soy_mask}")
        soy_counts = soy_counts_for_munis(Path(soy_mask), munis, all_touched=True)
        n_pos = sum(1 for v in soy_counts.values() if v > 0)
        log(f"Soy mask: {n_pos}/{len(soy_counts)} municipalities with soy pixels")

    tiffs = sorted(
        p
        for p in list(tiff_dir.glob("*.tiff")) + list(tiff_dir.glob("*.tif"))
        if not p.name.startswith("_") and parse_date_from_tiff(p)
    )
    if args.limit_dates is not None:
        tiffs = tiffs[: int(args.limit_dates)]
    if not tiffs:
        raise SystemExit(f"No dated daily TIFFs in {tiff_dir}")
    log(
        f"Season {season}: {len(tiffs)} daily TIFFs "
        f"(harmonized BDC, soy+SCL already applied) → staging {staging_dir}"
    )

    season_year = harvest_year_from_season(season)
    muni_workers = max(1, int(args.muni_workers))
    payloads = [
        {
            "tiff": str(t),
            "munis": munis,
            "soy_counts": soy_counts,
            "staging_dir": str(staging_dir),
            "season_year": season_year,
            "all_touched": True,
            "skip_existing": bool(args.skip_existing),
            "muni_workers": muni_workers,
        }
        for t in tiffs
    ]

    workers = max(1, int(args.workers))
    log(f"Parallelism: {workers} date processes × {muni_workers} muni threads")
    total_rows = total_empty = 0
    if workers == 1:
        for payload in payloads:
            name, w, e = process_one_tiff(payload)
            total_rows += w
            total_empty += e
            log(f"  {name}: rows={w} empty_munis={e}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(process_one_tiff, p): p["tiff"] for p in payloads}
            for fut in as_completed(futs):
                name, w, e = fut.result()
                total_rows += w
                total_empty += e
                log(f"  {name}: rows={w} empty_munis={e}")

    n = merge_staging_to_muni_csvs(
        staging_dir,
        out_root,
        season,
        codes,
        overwrite=bool(args.overwrite_csvs),
    )
    log(
        f"Done {season}. staging_rows={total_rows} empty={total_empty} "
        f"muni_csvs={n}. Next:\n"
        f"  python preprocessing/zonal_csv_to_muni_npy.py "
        f"--zonal-root {out_root} "
        f"--output-dir ~/sits_moco_data/npy_muni_mean --skip-existing"
    )


if __name__ == "__main__":
    main()
