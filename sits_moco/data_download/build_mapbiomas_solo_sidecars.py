#!/usr/bin/env python3
"""
Build per-municipality MapBiomas Solo sidecars [N, 4] next to time-series .npy cubes.

Does not modify the time-series .npy.

Modes:
  pixel      — sample clay/silt/sand + SOC at soy-pixel centroids (needs --tiff-root
               daily TIFFs matching preprocess geometry).
  muni_mean  — municipal zonal means from shapefiles, broadcast to [N,4]
               (no daily TIFFs; use when files/daily_tiff is not ready yet).

Output: {npy_root}/{year_range}/{code}/{code}_soil.npy
  channels: clay_pct, silt_pct, sand_pct, soc_t_ha

Examples:
  python data_download/build_mapbiomas_solo_sidecars.py \\
    --mode muni_mean --npy-root ~/sits_moco_data/npy \\
    --solo-dir files/mapbiomas_solo --year-range 2020-2021 -j 4

  python data_download/build_mapbiomas_solo_sidecars.py \\
    --mode pixel --npy-root ~/sits_moco_data/npy \\
    --tiff-root files/daily_tiff --solo-dir files/mapbiomas_solo \\
    --year-range 2020-2021 -j 4
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.transform import rowcol
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data_download.append_xavier_climate_to_npy import (  # noqa: E402
    _pixel_geometry_from_tiffs,
)
from data_download.xavier_rain_for_daily_npy import pixel_centroids_lonlat  # noqa: E402
from preprocessing.preprocess_daily_to_npy import (  # noqa: E402
    NO_DATA_VALUE,
    _save_npy_atomic,
)
from preprocessing.preprocess_tiff_to_npy import (  # noqa: E402
    season_start_from_year_range,
)

N_SOIL = 4
SOIL_NODATA = float(NO_DATA_VALUE)
SOIL_CRS = "EPSG:4674"


def harvest_year_from_range(year_range: str) -> int:
    """Season label YYYY1-YYYY2 → harvest / MapBiomas year YYYY2."""
    parts = year_range.strip().split("-")
    if len(parts) != 2:
        raise ValueError(f"Expected YYYY-YYYY, got {year_range!r}")
    return int(parts[1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Write {code}_soil.npy sidecars (clay/silt/sand/SOC) without rewriting "
            "time-series .npy."
        )
    )
    p.add_argument(
        "--mode",
        choices=("pixel", "muni_mean"),
        default="muni_mean",
        help="pixel needs daily TIFFs; muni_mean uses shapefile zonal means (default)",
    )
    p.add_argument(
        "--npy-root",
        type=Path,
        required=True,
        help="Base dir with {year_range}/{code}/{code}.npy",
    )
    p.add_argument(
        "--tiff-root",
        type=Path,
        default=None,
        help="Daily TIFF root {year_range}/{code}/*.tiff (required for --mode pixel)",
    )
    p.add_argument(
        "--shapefile-dir",
        type=Path,
        default=Path("files/shapefiles_pr"),
        help="Municipal shapefiles for --mode muni_mean (default: files/shapefiles_pr)",
    )
    p.add_argument(
        "--solo-dir",
        type=Path,
        default=Path("files/mapbiomas_solo"),
        help="Clipped MapBiomas Solo GeoTIFFs (default: files/mapbiomas_solo)",
    )
    p.add_argument(
        "--year-range",
        type=str,
        required=True,
        metavar="YYYY-YYYY",
        help="Season folder label, e.g. 2020-2021",
    )
    p.add_argument(
        "--region-tag",
        type=str,
        default="pr",
        help="Filename prefix for solo products (default: pr)",
    )
    p.add_argument("-j", "--workers", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--force",
        action="store_true",
        help="Rewrite sidecars that already exist",
    )
    p.add_argument(
        "--codes",
        type=str,
        default=None,
        help="Optional comma-separated municipality codes",
    )
    return p.parse_args()


def resolve_texture_paths(solo_dir: Path, tag: str) -> Path | tuple[Path, Path, Path]:
    """Return multiband texture stack path, or (clay, silt, sand) singles."""
    stack = solo_dir / f"{tag}_texture_0_30cm.tif"
    if stack.is_file():
        return stack
    clay = solo_dir / f"{tag}_clay_0_30cm.tif"
    silt = solo_dir / f"{tag}_silt_0_30cm.tif"
    sand = solo_dir / f"{tag}_sand_0_30cm.tif"
    missing = [p for p in (clay, silt, sand) if not p.is_file()]
    if missing:
        raise SystemExit(
            f"Missing texture stack {stack} and singles: "
            + ", ".join(str(p) for p in missing)
        )
    return (clay, silt, sand)


def _xy_to_crs(
    lon: np.ndarray, lat: np.ndarray, crs
) -> tuple[np.ndarray, np.ndarray]:
    if crs is None or str(crs).upper() in ("EPSG:4326", "OGC:CRS84"):
        return lon, lat
    from pyproj import Transformer

    to_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = to_crs.transform(lon, lat)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def _fill_from_band(
    out_col: np.ndarray,
    band: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    nodata,
) -> None:
    h, w = band.shape
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    if not np.any(valid):
        return
    vals = band[rows[valid], cols[valid]].astype(np.float32, copy=False)
    ok = np.isfinite(vals)
    if nodata is not None:
        ok &= vals != float(nodata)
    idx = np.flatnonzero(valid)[ok]
    out_col[idx] = vals[ok]


def _open_texture_bands(texture_paths: Path | tuple[Path, Path, Path]):
    if isinstance(texture_paths, tuple):
        clay_p, silt_p, sand_p = texture_paths
        clay_src = rasterio.open(clay_p)
        silt_src = rasterio.open(silt_p)
        sand_src = rasterio.open(sand_p)
        return clay_src, silt_src, sand_src, True
    src = rasterio.open(texture_paths)
    return src, src, src, False


def _sample_rasters_at_lonlat(
    lon: np.ndarray,
    lat: np.ndarray,
    texture_paths: Path | tuple[Path, Path, Path],
    soc_path: Path,
) -> np.ndarray:
    """Return [N, 4] float32 clay/silt/sand/soc; invalid → SOIL_NODATA."""
    n = int(lon.shape[0])
    out = np.full((n, N_SOIL), SOIL_NODATA, dtype=np.float32)

    clay_src, silt_src, sand_src, separate = _open_texture_bands(texture_paths)
    try:
        with rasterio.open(soc_path) as soc:
            tx, ty = _xy_to_crs(lon, lat, clay_src.crs)
            sx, sy = _xy_to_crs(lon, lat, soc.crs)
            rows_t, cols_t = rowcol(clay_src.transform, tx, ty)
            rows_s, cols_s = rowcol(soc.transform, sx, sy)
            rows_t = np.asarray(rows_t, dtype=np.int64)
            cols_t = np.asarray(cols_t, dtype=np.int64)
            rows_s = np.asarray(rows_s, dtype=np.int64)
            cols_s = np.asarray(cols_s, dtype=np.int64)

            if separate:
                clay = clay_src.read(1)
                silt = silt_src.read(1)
                sand = sand_src.read(1)
                tex_nd = clay_src.nodata
            else:
                clay = clay_src.read(1)
                silt = clay_src.read(2)
                sand = clay_src.read(3)
                tex_nd = clay_src.nodata
            carbon = soc.read(1)
            soc_nd = soc.nodata

            _fill_from_band(out[:, 0], clay, rows_t, cols_t, tex_nd)
            _fill_from_band(out[:, 1], silt, rows_t, cols_t, tex_nd)
            _fill_from_band(out[:, 2], sand, rows_t, cols_t, tex_nd)
            _fill_from_band(out[:, 3], carbon, rows_s, cols_s, soc_nd)
    finally:
        clay_src.close()
        if separate:
            silt_src.close()
            sand_src.close()

    return out


def _mean_masked(src, geoms, band: int = 1) -> float:
    out, _ = mask(src, geoms, crop=True, filled=True, nodata=SOIL_NODATA, indexes=band)
    if getattr(out, "ndim", 0) == 3:
        a = out[0].astype(np.float64)
    else:
        a = np.asarray(out, dtype=np.float64)
    a = np.where(a == SOIL_NODATA, np.nan, a)
    if not np.isfinite(a).any():
        return float("nan")
    return float(np.nanmean(a))


def _normalize_code(code: str) -> str:
    return f"{int(float(code)):07d}"


def compute_muni_soil_means(
    shapefile_dir: Path,
    texture_paths: Path | tuple[Path, Path, Path],
    soc_path: Path,
    *,
    codes_filter: set[str] | None,
) -> dict[str, np.ndarray]:
    """code -> [4] clay/silt/sand/soc municipal means."""
    shps: list[Path] = []
    # Prefer shallow scan: pathlib.rglob / os.walk can miss files on some
    # WSL↔NTFS mounts; municipal folders are one level deep.
    for child in sorted(shapefile_dir.iterdir()):
        if child.is_dir():
            for f in child.iterdir():
                if f.is_file() and f.suffix.lower() == ".shp":
                    shps.append(f)
        elif child.is_file() and child.suffix.lower() == ".shp":
            shps.append(child)
    if not shps:
        raise SystemExit(f"No shapefiles under {shapefile_dir}")

    clay_src, silt_src, sand_src, separate = _open_texture_bands(texture_paths)
    means: dict[str, np.ndarray] = {}
    try:
        with rasterio.open(soc_path) as soc:
            for shp in tqdm(shps, desc="Zonal means", unit="shp"):
                gdf = gpd.read_file(shp)
                if gdf.empty:
                    continue
                if gdf.crs is None:
                    gdf = gdf.set_crs(SOIL_CRS)
                if "code_muni" in gdf.columns:
                    code = _normalize_code(str(gdf["code_muni"].iloc[0]))
                else:
                    code = _normalize_code(shp.parent.name.split("_")[0])
                if codes_filter is not None and code not in codes_filter:
                    continue
                geoms = list(gdf.to_crs(SOIL_CRS).geometry)
                if separate:
                    clay = _mean_masked(clay_src, geoms, 1)
                    silt = _mean_masked(silt_src, geoms, 1)
                    sand = _mean_masked(sand_src, geoms, 1)
                else:
                    clay = _mean_masked(clay_src, geoms, 1)
                    silt = _mean_masked(clay_src, geoms, 2)
                    sand = _mean_masked(clay_src, geoms, 3)
                soc_v = _mean_masked(soc, geoms, 1)
                row = np.array([clay, silt, sand, soc_v], dtype=np.float32)
                row = np.where(np.isfinite(row), row, SOIL_NODATA).astype(np.float32)
                means[code] = row
    finally:
        clay_src.close()
        if separate:
            silt_src.close()
            sand_src.close()
    return means


def build_soil_pixel_one(
    code: str,
    npy_path: Path,
    tiff_dir: Path,
    soil_path: Path,
    texture_paths: Path | tuple[Path, Path, Path],
    soc_path: Path,
    *,
    dry_run: bool,
    force: bool,
) -> tuple[str, str]:
    if not npy_path.is_file():
        return code, f"error: missing npy {npy_path}"
    if soil_path.is_file() and not force:
        return code, "skip: sidecar exists"

    arr = np.load(npy_path, mmap_mode="r")
    if arr.ndim != 3:
        return code, f"error: expected 3D npy, got shape {arr.shape}"
    n, _t, c = arr.shape
    if c < 11:
        return code, f"error: expected >=11 channels, got {c}"

    tiff_paths = sorted(tiff_dir.glob("*.tiff")) + sorted(tiff_dir.glob("*.tif"))
    if not tiff_paths:
        return code, f"error: no TIFFs in {tiff_dir}"

    try:
        _dates, rows, cols, transform, crs = _pixel_geometry_from_tiffs(code, tiff_paths)
    except Exception as e:
        return code, f"error: geometry {e}"

    if len(rows) != n:
        return (
            code,
            f"error: TIFF pixel count {len(rows)} != npy N={n} "
            "(re-preprocess or check tiff-root)",
        )

    if dry_run:
        return code, "dry: would write soil sidecar"

    try:
        lon, lat = pixel_centroids_lonlat(rows, cols, transform, crs)
        soil = _sample_rasters_at_lonlat(lon, lat, texture_paths, soc_path)
    except Exception as e:
        return code, f"error: sample {e}"

    if soil.shape != (n, N_SOIL):
        return code, f"error: bad soil shape {soil.shape}"

    try:
        soil_path.parent.mkdir(parents=True, exist_ok=True)
        _save_npy_atomic(soil_path, soil.astype(np.float32, copy=False))
    except Exception as e:
        return code, f"error: save {e}"
    return code, "ok"


def build_soil_muni_mean_one(
    code: str,
    npy_path: Path,
    soil_path: Path,
    mean_row: np.ndarray | None,
    *,
    dry_run: bool,
    force: bool,
) -> tuple[str, str]:
    if not npy_path.is_file():
        return code, f"error: missing npy {npy_path}"
    if soil_path.is_file() and not force:
        return code, "skip: sidecar exists"
    if mean_row is None:
        return code, "error: no zonal mean for municipality"

    arr = np.load(npy_path, mmap_mode="r")
    if arr.ndim != 3:
        return code, f"error: expected 3D npy, got shape {arr.shape}"
    n = int(arr.shape[0])
    if dry_run:
        return code, "dry: would write soil sidecar"

    soil = np.broadcast_to(
        np.asarray(mean_row, dtype=np.float32).reshape(1, N_SOIL), (n, N_SOIL)
    ).copy()
    try:
        soil_path.parent.mkdir(parents=True, exist_ok=True)
        _save_npy_atomic(soil_path, soil)
    except Exception as e:
        return code, f"error: save {e}"
    return code, "ok"


def _worker_pixel(
    code: str,
    npy_path: str,
    tiff_dir: str,
    soil_path: str,
    texture_paths,
    soc_path: str,
    dry_run: bool,
    force: bool,
) -> tuple[str, str]:
    tex = texture_paths
    if isinstance(tex, list):
        tex = tuple(Path(p) for p in tex)
    elif isinstance(tex, str):
        tex = Path(tex)
    return build_soil_pixel_one(
        code,
        Path(npy_path),
        Path(tiff_dir),
        Path(soil_path),
        tex,
        Path(soc_path),
        dry_run=dry_run,
        force=force,
    )


def main() -> None:
    args = parse_args()
    year_range = args.year_range.strip()
    try:
        season_start_from_year_range(year_range)
        harvest = harvest_year_from_range(year_range)
    except ValueError as e:
        raise SystemExit(str(e)) from e

    npy_root = Path(args.npy_root).expanduser()
    npy_season = npy_root / year_range
    solo_dir = Path(args.solo_dir).expanduser()
    if not solo_dir.is_absolute():
        solo_dir = (_REPO / solo_dir).resolve()
    tag = args.region_tag.strip() or "pr"

    if not npy_season.is_dir():
        raise SystemExit(f"Not a directory: {npy_season}")

    texture_paths = resolve_texture_paths(solo_dir, tag)
    soc_path = solo_dir / f"{tag}_soc_0_30cm_{harvest}.tif"
    if not soc_path.is_file():
        raise SystemExit(f"Missing SOC year {harvest}: {soc_path}")

    codes_filter = None
    if args.codes:
        codes_filter = {_normalize_code(c) for c in args.codes.split(",") if c.strip()}

    jobs: list[tuple[str, Path, Path]] = []
    for muni_dir in sorted(npy_season.iterdir()):
        if not muni_dir.is_dir():
            continue
        code = muni_dir.name
        try:
            code_n = _normalize_code(code)
        except ValueError:
            continue
        if codes_filter is not None and code_n not in codes_filter:
            continue
        npy_path = muni_dir / f"{code}.npy"
        soil_path = muni_dir / f"{code}_soil.npy"
        if npy_path.is_file():
            jobs.append((code_n, npy_path, soil_path))

    if not jobs:
        print("No municipality .npy files found.")
        return

    print(
        f"Build soil sidecars mode={args.mode} for {len(jobs)} municipalities, "
        f"year-range={year_range} harvest={harvest}, dry_run={args.dry_run}"
    )
    print(f"  texture: {texture_paths}")
    print(f"  soc:     {soc_path}")

    counts: dict[str, int] = {}

    if args.mode == "muni_mean":
        shp_dir = Path(args.shapefile_dir).expanduser()
        if not shp_dir.is_absolute():
            shp_dir = (_REPO / shp_dir).resolve()
        if not shp_dir.is_dir():
            raise SystemExit(f"Not a directory: {shp_dir}")
        print(f"  shapefiles: {shp_dir}")
        means = compute_muni_soil_means(
            shp_dir,
            texture_paths,
            soc_path,
            codes_filter={c for c, _, _ in jobs},
        )
        print(f"  zonal means for {len(means)} municipalities")
        for code, npy_path, soil_path in tqdm(jobs, desc="Write sidecars", unit="muni"):
            _c, status = build_soil_muni_mean_one(
                code,
                npy_path,
                soil_path,
                means.get(code),
                dry_run=args.dry_run,
                force=args.force,
            )
            key = status.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            if status.startswith("error"):
                print(f"[ERROR] {code}: {status}")
    else:
        if args.tiff_root is None:
            raise SystemExit("--tiff-root is required for --mode pixel")
        tiff_season = Path(args.tiff_root).expanduser() / year_range
        if not tiff_season.is_dir():
            raise SystemExit(f"Not a directory: {tiff_season}")
        workers = max(1, int(args.workers))
        tex_arg: str | list[str]
        if isinstance(texture_paths, tuple):
            tex_arg = [str(p) for p in texture_paths]
        else:
            tex_arg = str(texture_paths)

        pixel_jobs = [
            (code, npy_path, tiff_season / code, soil_path)
            for code, npy_path, soil_path in jobs
        ]
        if workers <= 1:
            for code, npy_path, tiff_dir, soil_path in tqdm(
                pixel_jobs, desc="Municipalities", unit="muni"
            ):
                _c, status = build_soil_pixel_one(
                    code,
                    npy_path,
                    tiff_dir,
                    soil_path,
                    texture_paths,
                    soc_path,
                    dry_run=args.dry_run,
                    force=args.force,
                )
                key = status.split(":")[0]
                counts[key] = counts.get(key, 0) + 1
                if status.startswith("error"):
                    print(f"[ERROR] {code}: {status}")
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {
                    ex.submit(
                        _worker_pixel,
                        code,
                        str(npy_path),
                        str(tiff_dir),
                        str(soil_path),
                        tex_arg,
                        str(soc_path),
                        args.dry_run,
                        args.force,
                    ): code
                    for code, npy_path, tiff_dir, soil_path in pixel_jobs
                }
                for fut in tqdm(
                    as_completed(futs),
                    total=len(futs),
                    desc="Municipalities",
                    unit="muni",
                ):
                    _c, status = fut.result()
                    key = status.split(":")[0]
                    counts[key] = counts.get(key, 0) + 1
                    if status.startswith("error"):
                        print(f"[ERROR] {_c}: {status}")

    print("Summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
