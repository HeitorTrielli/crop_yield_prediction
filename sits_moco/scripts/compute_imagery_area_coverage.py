#!/usr/bin/env python3
"""
Compute Sentinel-2 valid-pixel area per (municipality, harvest year) and enrich the yield CSV.

For each row, compares imagery area to IBGE planted soybean area and adds columns used to
scale municipality production to the fraction of planted area covered by valid S2 pixels:

  production_t_s2_adj = production_t * min(pixel_area_ha / area_planted_ha, 1)

``pixel_area_ha`` is the total valid Sentinel-2 footprint (ha), not the size of one pixel.
Productivity (yield_t_ha) is unchanged.

The existing ``split`` column is preserved for training.

Typical workflow:

  python data_download/download_pam_soy_sidra.py
  python add_train_valid_test_split.py --holdout-years 2021
  python scripts/compute_imagery_area_coverage.py

Requires ``files/npy/<season>/<code>/<code>.npy`` and matching
``files/daily_tiff/<season>/<code>/*.tiff`` (for georeferencing).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS
from rasterio.transform import xy

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_download.download_pam_soy_sidra import (  # noqa: E402
    build_sidra_url,
    fetch_sidra,
    sidra_to_dataframe,
)

DEFAULT_YIELD_CSV = Path("files/pam_soy_pr_2019_2025.csv")
DEFAULT_NPY_ROOT = Path("files/npy")
DEFAULT_TIFF_ROOT = Path("files/daily_tiff")

# Columns written to the yield CSV (municipality_code + split kept for the training pipeline).
OUTPUT_COLUMNS = [
    "municipality_code",
    "municipality_name",
    "year",
    "production_t",
    "yield_t_ha",
    "area_planted_ha",
    "pixel_area_ha",
    "coverage_ratio",
    "production_t_s2_adj",
    "split",
]

LEGACY_IMAGERY_COLUMNS = [
    "area_planted_ha_source",
    "s2_n_pixels",
    "s2_valid_area_ha",
    "s2_area_fraction",
    "s2_area_fraction_capped",
]


def harvest_to_season_folder(harvest_year: int) -> str:
    return f"{harvest_year - 1}-{harvest_year}"


def list_season_folders(root: Path) -> dict[int, Path]:
    """Map harvest year -> season directory under ``root``."""
    out: dict[int, Path] = {}
    if not root.is_dir():
        return out
    for item in root.iterdir():
        if not item.is_dir() or "-" not in item.name:
            continue
        parts = item.name.split("-")
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        harvest = int(parts[1])
        if any(item.rglob("*.npy")):
            out[harvest] = item
    return out


def list_daily_tiffs(muni_dir: Path) -> list[Path]:
    if not muni_dir.is_dir():
        return []
    return sorted(muni_dir.glob("*.tiff")) + sorted(muni_dir.glob("*.tif"))


def single_pixel_area_ha(transform, crs, height: int, width: int) -> float:
    """Area of one raster pixel in hectares (at grid centre)."""
    _lon, lat = xy(transform, height // 2, width // 2, offset="center")
    if CRS.from_user_input(crs).is_geographic:
        lat_rad = math.radians(lat)
        width_m = abs(transform.a) * 111_320.0 * math.cos(lat_rad)
        height_m = abs(transform.e) * 111_320.0
    else:
        width_m = abs(transform.a)
        height_m = abs(transform.e)
    return width_m * height_m / 10_000.0


def s2_area_for_municipality(
    npy_path: Path,
    tiff_dir: Path | None,
) -> tuple[int, float] | None:
    """Return (n_pixels, valid_area_ha) for one municipality season."""
    if not npy_path.is_file():
        return None
    try:
        n_pixels = len(np.load(npy_path, mmap_mode="r"))
    except OSError:
        return None
    if n_pixels == 0:
        return 0, 0.0

    if tiff_dir is None:
        return None
    tiffs = list_daily_tiffs(tiff_dir)
    if not tiffs:
        return None

    try:
        with rasterio.open(tiffs[0]) as src:
            if src.crs is None:
                return None
            pixel_ha = single_pixel_area_ha(src.transform, src.crs, src.height, src.width)
    except OSError:
        return None

    return n_pixels, float(n_pixels * pixel_ha)


def scan_imagery_areas(
    npy_root: Path,
    tiff_root: Path | None,
    harvest_years: set[int] | None = None,
) -> pd.DataFrame:
    """Build a table with one row per (municipality_code, harvest_year) that has .npy."""
    seasons = list_season_folders(npy_root)
    if harvest_years is not None:
        seasons = {y: p for y, p in seasons.items() if y in harvest_years}

    rows: list[dict] = []
    for harvest_year, season_dir in sorted(seasons.items()):
        season_name = season_dir.name
        tiff_season = tiff_root / season_name if tiff_root is not None else None
        for muni_dir in sorted(p for p in season_dir.iterdir() if p.is_dir()):
            code = muni_dir.name
            if not code.isdigit():
                continue
            npy_path = muni_dir / f"{code}.npy"
            tiff_dir = tiff_season / code if tiff_season is not None else None
            result = s2_area_for_municipality(npy_path, tiff_dir)
            if result is None:
                continue
            n_pixels, area_ha = result
            rows.append(
                {
                    "municipality_code": int(code),
                    "year": harvest_year,
                    "pixel_area_ha": area_ha,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["municipality_code", "year", "pixel_area_ha"])
    return pd.DataFrame(rows)


def merge_sidra_area_planted(
    yield_df: pd.DataFrame,
    *,
    years: str = "2019-2025",
) -> pd.DataFrame:
    """Fetch SIDRA area planted (v109) and merge; fill gaps with production/yield."""
    url = build_sidra_url(years=years)
    print(f"Fetching SIDRA area planted: {url}")
    records = fetch_sidra(url)
    sidra = sidra_to_dataframe(records)

    keys = ["municipality_code", "year"]
    yield_df = yield_df.copy()
    yield_df["municipality_code"] = yield_df["municipality_code"].astype(int)
    yield_df["year"] = pd.to_numeric(yield_df["year"], errors="coerce").astype("Int64")

    sidra_subset = sidra[keys + ["area_planted_ha"]]
    for col in ("area_planted_ha", *LEGACY_IMAGERY_COLUMNS):
        if col in yield_df.columns:
            yield_df = yield_df.drop(columns=[col])

    merged = yield_df.merge(sidra_subset, on=keys, how="left")

    derived_mask = merged["area_planted_ha"].isna() & merged["production_t"].notna() & (
        merged["yield_t_ha"] > 0
    )
    if derived_mask.any():
        merged.loc[derived_mask, "area_planted_ha"] = (
            merged.loc[derived_mask, "production_t"] / merged.loc[derived_mask, "yield_t_ha"]
        )

    return merged


def enrich_yield_csv(
    yield_df: pd.DataFrame,
    imagery_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add pixel_area_ha and production_t_s2_adj; drop intermediate columns."""
    out = yield_df.copy()
    out["municipality_code"] = out["municipality_code"].astype(int)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")

    drop_cols = [c for c in LEGACY_IMAGERY_COLUMNS + ["pixel_area_ha", "coverage_ratio", "production_t_s2_adj"] if c in out.columns]
    out = out.drop(columns=drop_cols)

    keys = ["municipality_code", "year"]
    out = out.merge(imagery_df[keys + ["pixel_area_ha"]], on=keys, how="left")

    valid = (
        out["area_planted_ha"].notna()
        & (out["area_planted_ha"] > 0)
        & out["pixel_area_ha"].notna()
    )
    coverage = np.where(valid, out["pixel_area_ha"] / out["area_planted_ha"], np.nan)
    out["coverage_ratio"] = coverage
    coverage_capped = np.where(valid, np.minimum(coverage, 1.0), np.nan)
    out["production_t_s2_adj"] = np.where(
        valid & out["production_t"].notna(),
        out["production_t"] * coverage_capped,
        np.nan,
    )

    keep = [c for c in OUTPUT_COLUMNS if c in out.columns]
    return out[keep]


def print_summary(df: pd.DataFrame) -> None:
    print("\nImagery coverage summary:")
    ok = df["pixel_area_ha"].notna() & df["area_planted_ha"].notna() & (df["area_planted_ha"] > 0)
    if not ok.any():
        print("  No rows with both area_planted_ha and pixel_area_ha.")
        return
    ratio = df.loc[ok, "pixel_area_ha"] / df.loc[ok, "area_planted_ha"]
    print(f"  Rows with imagery: {ok.sum()} / {len(df)}")
    print(
        f"  pixel_area_ha / area_planted_ha: min={ratio.min():.3f}, "
        f"median={ratio.median():.3f}, max={ratio.max():.3f}"
    )
    over = (ratio > 1.0).sum()
    if over:
        print(
            f"  Note: {over} rows exceed planted area (whole-muni clip > soy hectares); "
            "production_t_s2_adj uses min(ratio, 1)."
        )
    if "split" in df.columns:
        print("\n  Median coverage ratio by split:")
        for split in ["train", "valid", "test"]:
            sub_ok = ok & (df["split"] == split)
            if not sub_ok.any():
                continue
            sub_ratio = df.loc[sub_ok, "pixel_area_ha"] / df.loc[sub_ok, "area_planted_ha"]
            print(f"    {split}: {sub_ratio.median():.3f} ({sub_ok.sum()} rows)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--yield-csv",
        type=Path,
        default=DEFAULT_YIELD_CSV,
        help=f"Yield CSV to enrich (default: {DEFAULT_YIELD_CSV})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: overwrite --yield-csv)",
    )
    p.add_argument(
        "--report-csv",
        type=Path,
        default=None,
        help="Optional copy of the enriched CSV (same columns as output)",
    )
    p.add_argument(
        "--datapath",
        type=Path,
        default=DEFAULT_NPY_ROOT,
        help=f"Root with season .npy folders (default: {DEFAULT_NPY_ROOT})",
    )
    p.add_argument(
        "--tiffpath",
        type=Path,
        default=DEFAULT_TIFF_ROOT,
        help=f"Root with daily TIFF season folders (default: {DEFAULT_TIFF_ROOT})",
    )
    p.add_argument(
        "--fetch-area-planted",
        action="store_true",
        help="Re-fetch area_planted_ha from SIDRA before scanning imagery",
    )
    p.add_argument(
        "--sidra-years",
        default="2019-2025",
        help="SIDRA period when using --fetch-area-planted (default: 2019-2025)",
    )
    p.add_argument(
        "--harvest-years",
        type=int,
        nargs="*",
        default=None,
        help="Limit imagery scan to these harvest years (default: all seasons under datapath)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    yield_csv = args.yield_csv
    if not yield_csv.is_file():
        raise SystemExit(f"Yield CSV not found: {yield_csv}")

    print(f"Loading {yield_csv} ...")
    yield_df = pd.read_csv(yield_csv)
    if "split" in yield_df.columns:
        print(f"  split column present ({yield_df['split'].value_counts().to_dict()})")

    if args.fetch_area_planted or "area_planted_ha" not in yield_df.columns:
        yield_df = merge_sidra_area_planted(yield_df, years=args.sidra_years)

    harvest_set = set(args.harvest_years) if args.harvest_years else None
    tiff_root = args.tiffpath if args.tiffpath.is_dir() else None
    if tiff_root is None:
        print(f"Warning: tiffpath {args.tiffpath} not found; cannot compute areas.")

    print(f"Scanning imagery under {args.datapath} ...")
    imagery_df = scan_imagery_areas(args.datapath, tiff_root, harvest_years=harvest_set)
    print(f"  Found {len(imagery_df)} municipality-season pairs with .npy")

    enriched = enrich_yield_csv(yield_df, imagery_df)
    print_summary(enriched)

    output = args.output or yield_csv
    output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output, index=False)
    print(f"\nSaved enriched yield CSV to {output}")

    if args.report_csv:
        args.report_csv.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_csv(args.report_csv, index=False)
        print(f"Saved copy to {args.report_csv}")


if __name__ == "__main__":
    main()
