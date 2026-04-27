#!/usr/bin/env python3
"""
Clip state-level daily GeoTIFFs to municipality shapefiles (one TIFF per municipality per day).

State TIFFs live in e.g. files/gee_soy_daily/30/state_41/2022-2023. Municipal shapefiles
are in files/municipal_shapefiles/{municipal_code}_{name}/ with one .shp each. Output:
files/gee_soy_daily/30/{municipal_code}/2022-2023/{municipal_code}_{Y}_{M}_{D}.tiff.
Days with no valid pixels for a municipality are skipped. No-data: 0, -9999, NaN.

CLI usage:
  # Defaults: tiff-dir=files/gee_soy_daily/30/state_41/2022-2023, shapefile-dir=files/municipal_shapefiles, output-dir=files/gee_soy_daily
  python clip_tiffs_to_shapefiles.py

  # Custom paths and 20 parallel workers
  python clip_tiffs_to_shapefiles.py --tiff-dir files/gee_soy_daily/30/state_41/2022-2023 --shapefile-dir files/municipal_shapefiles --output-dir files/gee_soy_daily -j 20

  # Single shapefile(s) instead of directory
  python clip_tiffs_to_shapefiles.py --shapefile path/to/4100103.shp --shapefile path/to/4100202.shp -j 20
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

import numpy as np

try:
    import geopandas as gpd
except ImportError:
    raise SystemExit("Install geopandas: pip install geopandas")

try:
    import rasterio
    from rasterio.mask import mask as rio_mask
except ImportError:
    raise SystemExit("Install rasterio: pip install rasterio")

# Must match GEE export and merge (NO_DATA_VALUE). Output TIFFs and valid-pixel check use this.
NO_DATA_VALUE = -9999


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clip TIFF rasters by shapefile polygons (e.g. state TIFFs → municipal TIFFs)."
    )
    p.add_argument(
        "--tiff-dir",
        type=Path,
        default=Path("files/gee_soy_daily/30/state_41/2022-2023"),
        help="Directory containing state daily GeoTIFFs (default: files/gee_soy_daily/30/state_41/2022-2023)",
    )
    p.add_argument(
        "--shapefile-dir",
        type=Path,
        default=Path("files/municipal_shapefiles"),
        help="Directory of municipal shapefiles: subfolders {municipal_code}_{name} with one .shp each (default: files/municipal_shapefiles)",
    )
    p.add_argument(
        "--shapefile",
        type=Path,
        action="append",
        dest="shapefiles",
        help="Path(s) to shapefile(s) to clip to; can be repeated. Takes precedence over --shapefile-dir when set.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("files/gee_soy_daily"),
        help="Base output directory (default: files/gee_soy_daily). Output: {output_dir}/{resolution}/{municipal_code}/2022-2023/",
    )
    p.add_argument(
        "--resolution",
        type=int,
        default=30,
        metavar="M",
        help="Resolution folder name under output-dir (default: 30)",
    )
    p.add_argument(
        "--all-touched",
        action="store_true",
        default=True,
        help="Include pixels touched by the polygon (default: True)",
    )
    p.add_argument(
        "--workers",
        "-j",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel workers (default: 1). Use e.g. 20 to process 20 days at a time.",
    )
    p.add_argument(
        "--year-range",
        type=str,
        default=None,
        metavar="YYYY-YYYY",
        help="Output subfolder name (e.g. 2022-2023). If not set, inferred from TIFF filenames.",
    )
    return p.parse_args()


def parse_date_from_tiff_path(path: Path) -> str | None:
    """Extract YYYY-MM-DD from a path like PARANA_2022_10_09.tiff or prefix_2022-10-09.tif."""
    stem = path.stem
    # Try underscore: prefix_2022_10_09
    parts = stem.split("_")
    if (
        len(parts) >= 4
        and parts[-3].isdigit()
        and parts[-2].isdigit()
        and parts[-1].isdigit()
    ):
        y, m, d = parts[-3], parts[-2], parts[-1]
        if len(y) == 4 and len(m) == 2 and len(d) == 2:
            return f"{y}-{m}-{d}"
    # Try hyphen in last part: prefix_2022-10-09
    if "-" in stem:
        match = re.search(r"(\d{4})-(\d{2})-(\d{2})$", stem)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return None


def collect_municipalities(
    shapefile_dir: Path | str | None = None,
    shapefiles: list[Path | str] | None = None,
) -> list[tuple[Path, str, Any]]:
    """
    Load municipality shapefiles and return (path, municipal_code, geometry in EPSG:4326).

    Either shapefile_dir or shapefiles must be provided. If shapefiles is non-empty,
    it takes precedence over shapefile_dir.
    - shapefile_dir: directory with one subfolder per municipality, each containing a .shp.
      Municipal code is taken from subdir name (e.g. 4100103_Abatiá -> 4100103).
    - shapefiles: list of paths to .shp files; code is the stem of each path.
    """
    if shapefiles:
        result = []
        for path in shapefiles:
            p = Path(path).resolve()
            if not p.exists():
                raise ValueError(f"Shapefile not found: {p}")
            gdf = gpd.read_file(p)
            if gdf.crs is None or gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            result.append((p, p.stem, gdf.geometry.union_all()))
        return result
    if shapefile_dir is not None:
        d = Path(shapefile_dir).resolve()
        if not d.is_dir():
            raise ValueError(f"Not a directory: {d}")
        result = []
        for subdir in sorted(d.iterdir()):
            if subdir.is_dir():
                shps = list(subdir.glob("*.shp"))
                if shps:
                    path = shps[0]
                    code = (
                        subdir.name.split("_")[0] if "_" in subdir.name else path.stem
                    )
                    gdf = gpd.read_file(path)
                    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
                        gdf = gdf.to_crs(epsg=4326)
                    geom = gdf.geometry.union_all()
                    result.append((path, code, geom))
        return result
    raise ValueError("Provide either shapefile_dir or shapefiles")


def _process_one_tiff(
    tiff_path: Path,
    municipalities: list[tuple[Path, str, Any]],
    output_dir: Path,
    res_dir: str,
    year_range: str,
    all_touched: bool,
) -> tuple[str, int]:
    """
    Process a single day TIFF: clip to all municipalities and write outputs.
    Returns (tiff_basename, number_of_files_written). Used by parallel workers.
    """
    date_str = parse_date_from_tiff_path(tiff_path)
    if not date_str:
        return (tiff_path.name, 0)
    y, m, d = date_str.split("-")
    written = 0
    with rasterio.open(tiff_path) as src:
        src_crs = src.crs
        for _path, code, geom in municipalities:
            out_subdir = output_dir / res_dir / code / year_range
            out_subdir.mkdir(parents=True, exist_ok=True)
            out_file = out_subdir / f"{code}_{y}_{m}_{d}.tiff"

            if geom is not None:
                gdf_one = gpd.GeoDataFrame([{"g": geom}], geometry="g", crs="EPSG:4326")
                gdf_one = gdf_one.to_crs(src_crs)
                geom_clip = gdf_one.geometry.iloc[0]
            else:
                geom_clip = geom
            geojson_geom = (
                geom_clip.__geo_interface__
                if hasattr(geom_clip, "__geo_interface__")
                else geom_clip
            )
            try:
                out_image, out_transform = rio_mask(
                    src, [geojson_geom], crop=True, all_touched=all_touched
                )
            except Exception:
                continue

            if out_image.size == 0:
                continue

            # Skip if no informative pixels. Use same rule as exclude_invalid_tiffs/count:
            # a pixel is valid only if ALL bands are valid (not nodata, 0, -9999, NaN).
            data = out_image.data if hasattr(out_image, "mask") else out_image
            inside = (
                ~out_image.mask
                if hasattr(out_image, "mask")
                else np.ones(out_image.shape, dtype=bool)
            )
            valid = inside.copy()
            if src.nodata is not None:
                valid &= data != src.nodata
            valid &= data != 0
            valid &= data != NO_DATA_VALUE
            if np.issubdtype(data.dtype, np.floating):
                valid &= ~np.isnan(data)
            # Require at least one pixel with all bands valid (multi-band)
            valid_per_pixel = np.all(valid, axis=0)
            if not np.any(valid_per_pixel):
                continue

            # Preserve source encoding (profile has compress, tiled, block sizes) so pixel-less
            # areas don't waste space; only override dimensions and transform.
            out_meta = src.profile.copy()
            out_meta.update(
                {
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "nodata": NO_DATA_VALUE,
                }
            )
            # Use source compression if present, else LZW so nodata regions don't bloat the file
            if out_meta.get("compress") is None:
                out_meta["compress"] = "lzw"

            data = (
                out_image.filled(NO_DATA_VALUE)
                if hasattr(out_image, "filled")
                else out_image
            )
            with rasterio.open(out_file, "w", **out_meta) as dst:
                dst.write(data)
            written += 1
    return (tiff_path.name, written)


def collect_shapefiles(args: argparse.Namespace) -> list[tuple[Path, str, Any]]:
    """CLI helper: return list of (path, code, geometry) from parsed args."""
    shapefiles = getattr(args, "shapefiles", None) or []
    if shapefiles:
        return collect_municipalities(shapefiles=shapefiles)
    if getattr(args, "shapefile_dir", None) is not None:
        return collect_municipalities(shapefile_dir=args.shapefile_dir)
    raise ValueError("Provide either --shapefile-dir or one or more --shapefile")


def clip_state_tiffs_to_municipalities(
    tiff_dir: Path | str,
    shapefile_dir: Path | str | None = None,
    shapefiles: list[Path | str] | None = None,
    output_dir: Path | str = "files/gee_soy_daily",
    resolution: int = 30,
    year_range: str | None = None,
    all_touched: bool = True,
    verbose: bool = True,
    workers: int = 1,
) -> None:
    """
    Clip state-level daily TIFFs to municipal boundaries and save one TIFF per municipality per day.

    Reads all .tif/.tiff from tiff_dir (one file per day for the state), clips each by each
    municipality polygon, and writes to:
      {output_dir}/{resolution}/{municipal_code}/{year_range}/{municipal_code}_{Y}_{M}_{D}.tiff

    If a municipality has no pixels for a given day (e.g. outside raster extent), no file
    is written for that day/municipality.

    Parameters
    ----------
    tiff_dir : path
        Directory containing state daily GeoTIFFs (e.g. PARANA_2022_10_09.tiff).
    shapefile_dir : path, optional
        Directory of municipal shapefiles: one subfolder per municipality with a .shp inside.
        Use either this or shapefiles.
    shapefiles : list of paths, optional
        Explicit list of .shp paths. Use either this or shapefile_dir.
    output_dir : path
        Base output directory (default: files/gee_soy_daily). Output will be
        {output_dir}/{resolution}/{municipal_code}/2022-2023/ etc.
    resolution : int
        Resolution folder name under output_dir (default: 30).
    year_range : str, optional
        Subfolder name for year range (e.g. "2022-2023"). If None, inferred from TIFF filenames.
    all_touched : bool
        Include pixels touched by the polygon when masking (default: True).
    verbose : bool
        If True, print progress (default: True).
    workers : int
        Number of parallel workers (default: 1). Use e.g. 20 to process 20 days at a time.
    """
    tiff_dir = Path(tiff_dir).resolve()
    if not tiff_dir.is_dir():
        raise ValueError(f"Not a directory: {tiff_dir}")

    tiffs = sorted(tiff_dir.glob("*.tif")) + sorted(tiff_dir.glob("*.tiff"))
    if not tiffs:
        raise ValueError(f"No .tif(f) files in {tiff_dir}")

    municipalities = collect_municipalities(
        shapefile_dir=shapefile_dir,
        shapefiles=shapefiles or [],
    )
    if not municipalities:
        raise ValueError("No shapefiles found.")

    dates = []
    for t in tiffs:
        d = parse_date_from_tiff_path(t)
        if d:
            dates.append(d)
    if not dates:
        raise ValueError(
            "Could not parse dates from TIFF filenames (expect e.g. prefix_2022_10_09.tiff)"
        )
    years = [int(d[:4]) for d in dates]
    year_start, year_end = min(years), max(years)
    inferred_year_range = f"{year_start}-{year_end}"
    if year_range is None:
        year_range = inferred_year_range

    output_dir = Path(output_dir).resolve()
    res_dir = str(resolution)

    if workers <= 1:
        for tiff_path in tiffs:
            date_str = parse_date_from_tiff_path(tiff_path)
            if not date_str:
                if verbose:
                    _log(f"Skip (no date): {tiff_path.name}")
                continue
            name, written = _process_one_tiff(
                tiff_path,
                municipalities,
                output_dir,
                res_dir,
                year_range,
                all_touched,
            )
            if verbose:
                _log(f"Clipped {name} -> {written} municipality files")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _process_one_tiff,
                    tiff_path,
                    municipalities,
                    output_dir,
                    res_dir,
                    year_range,
                    all_touched,
                ): tiff_path
                for tiff_path in tiffs
            }
            for future in as_completed(futures):
                try:
                    name, written = future.result()
                    if verbose and name:
                        _log(f"Clipped {name} -> {written} municipality files")
                except Exception as e:
                    tiff_path = futures[future]
                    if verbose:
                        _log(f"Failed {tiff_path.name}: {e}")

    if verbose:
        _log(
            f"Done. Output under {output_dir / res_dir}/{{municipal_code}}/{year_range}/"
        )


def main() -> None:
    """CLI entrypoint: clip state TIFFs to municipality shapefiles."""
    args = parse_args()
    try:
        clip_state_tiffs_to_municipalities(
            tiff_dir=args.tiff_dir,
            shapefile_dir=args.shapefile_dir,
            shapefiles=args.shapefiles or [],
            output_dir=args.output_dir,
            resolution=args.resolution,
            year_range=getattr(args, "year_range", None),
            all_touched=args.all_touched,
            verbose=True,
            workers=getattr(args, "workers", 1),
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e


if __name__ == "__main__":
    main()
