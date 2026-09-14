#!/usr/bin/env python3
"""
Plot yearly per-pixel NDVI histograms (min / max / median / mean over time).

For every valid pixel in a season, computes the temporal min, max, median, and
mean of NDVI, then draws a 2x2 histogram of those four distributions.

Data source (whichever exists; .npy is much faster):
  1. files/npy/{year-range}/{code}/{code}.npy   shape [N, T, 11+]
  2. files/daily_tiff/{year-range}/{code}/*.tiff  (fallback; warps dates to a common grid)

Example:
  python scripts/plot_yearly_ndvi_histograms.py --year 2021
  python scripts/plot_yearly_ndvi_histograms.py --year-range 2020-2021 -j 8
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _ensure_pyproj_proj_data() -> None:
    pl = os.environ.get("PROJ_LIB", "")
    if pl and ("postgresql" in pl.lower() or "postgis" in pl.lower()):
        os.environ.pop("PROJ_LIB", None)
    try:
        import pyproj.datadir

        d = pyproj.datadir.get_data_dir()
        if d and os.path.isdir(d):
            os.environ["PROJ_DATA"] = d
    except Exception:
        pass


_ensure_pyproj_proj_data()

import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import Resampling, reproject
from tqdm import tqdm

NO_DATA_VALUE = -9999.0
RED_BAND_IDX = 2  # 0-based; Sentinel-2 B4
NIR_BAND_IDX = 6  # 0-based; Sentinel-2 B8
DEFAULT_NPY_ROOT = Path("files/npy")
DEFAULT_TIFF_ROOT = Path("files/daily_tiff")
DEFAULT_OUT_DIR = Path("files/figures/ndvi_histograms")


def harvest_to_season_folder(harvest_year: int) -> str:
    return f"{harvest_year - 1}-{harvest_year}"


def parse_year_range(year_range: str) -> tuple[int, int]:
    m = re.match(r"^(\d{4})-(\d{4})$", year_range.strip())
    if not m:
        raise ValueError(f"Expected year-range like 2020-2021, got {year_range!r}")
    return int(m.group(1)), int(m.group(2))


def resolve_season(args: argparse.Namespace) -> str:
    if args.year_range:
        parse_year_range(args.year_range)
        return args.year_range
    if args.year is not None:
        return harvest_to_season_folder(int(args.year))
    raise SystemExit("Pass --year (harvest, e.g. 2021) or --year-range (e.g. 2020-2021).")


def list_muni_npy(npy_season: Path) -> list[Path]:
    if not npy_season.is_dir():
        return []
    return sorted(npy_season.glob("*/*.npy")) + sorted(npy_season.glob("*.npy"))


def list_muni_tiff_dirs(tiff_season: Path) -> list[Path]:
    if not tiff_season.is_dir():
        return []
    return sorted(p for p in tiff_season.iterdir() if p.is_dir())


def ndvi_from_red_nir(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute NDVI; invalid reflectance -> NaN. Expects raw DN (scale 1e-4)."""
    red_f = red.astype(np.float32) * 1e-4
    nir_f = nir.astype(np.float32) * 1e-4
    missing = (
        ~np.isfinite(red_f)
        | ~np.isfinite(nir_f)
        | (red == 0)
        | (nir == 0)
        | (red == NO_DATA_VALUE)
        | (nir == NO_DATA_VALUE)
    )
    denom = nir_f + red_f
    valid = ~missing & (np.abs(denom) > 1e-8)
    out = np.full(red_f.shape, np.nan, dtype=np.float32)
    out[valid] = np.clip((nir_f[valid] - red_f[valid]) / denom[valid], -1.0, 1.0)
    return out


def _empty_ndvi_stats() -> dict[str, np.ndarray]:
    empty = np.array([], dtype=np.float32)
    return {"min": empty, "max": empty, "median": empty, "mean": empty}


def temporal_ndvi_stats(ndvi_t: np.ndarray, min_valid_days: int = 1) -> dict[str, np.ndarray]:
    """
    ndvi_t: [T, N] or [T, H, W] with NaN for missing.
    Returns 1-D arrays (pixels with enough valid days) for min/max/median/mean.
    """
    if ndvi_t.size == 0 or ndvi_t.ndim < 1 or ndvi_t.shape[0] == 0:
        return _empty_ndvi_stats()

    flat = ndvi_t.reshape(ndvi_t.shape[0], -1)  # [T, P]
    if flat.shape[1] == 0:
        return _empty_ndvi_stats()

    valid_count = np.sum(np.isfinite(flat), axis=0)
    keep = valid_count >= min_valid_days
    if not np.any(keep):
        return _empty_ndvi_stats()

    sub = flat[:, keep]
    with np.errstate(all="ignore"):
        return {
            "min": np.nanmin(sub, axis=0).astype(np.float32),
            "max": np.nanmax(sub, axis=0).astype(np.float32),
            "median": np.nanmedian(sub, axis=0).astype(np.float32),
            "mean": np.nanmean(sub, axis=0).astype(np.float32),
        }


def stats_from_npy(path: Path, min_valid_days: int) -> dict[str, np.ndarray]:
    """Load municipal .npy [N, T, C] and return per-pixel temporal NDVI stats."""
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 3 or arr.shape[2] < 7 or arr.shape[0] == 0:
        return _empty_ndvi_stats()

    # Process in chunks to keep peak RAM low on large municipalities.
    n_pix = arr.shape[0]
    chunk = max(8_000, min(50_000, n_pix))
    parts: dict[str, list[np.ndarray]] = {k: [] for k in ("min", "max", "median", "mean")}
    for start in range(0, n_pix, chunk):
        block = np.asarray(arr[start : start + chunk, :, :10], dtype=np.float32)
        red = block[:, :, RED_BAND_IDX]  # [n, T]
        nir = block[:, :, NIR_BAND_IDX]
        ndvi = ndvi_from_red_nir(red, nir)  # [n, T]
        st = temporal_ndvi_stats(ndvi.T, min_valid_days=min_valid_days)
        for k, v in st.items():
            if v.size:
                parts[k].append(v)
    return {
        k: (np.concatenate(vs) if vs else np.array([], dtype=np.float32))
        for k, vs in parts.items()
    }


def _parse_tiff_date(stem: str, code: str) -> date | None:
    if not stem.startswith(code + "_"):
        return None
    suffix = stem[len(code) + 1 :].replace("-", "_")
    parts = suffix.split("_")
    if len(parts) >= 3 and all(p.isdigit() for p in parts[:3]):
        try:
            return date(int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            return None
    return None


def _read_red_nir_on_grid(
    path: Path,
    ref_transform,
    ref_crs,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp red/nir bands onto the reference grid; nodata -> NaN."""
    red = np.full((height, width), np.nan, dtype=np.float32)
    nir = np.full((height, width), np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        src_crs = src.crs if src.crs is not None else ref_crs
        nodata = src.nodata if src.nodata is not None else NO_DATA_VALUE
        for out, band_idx in ((red, RED_BAND_IDX + 1), (nir, NIR_BAND_IDX + 1)):
            if src.count < band_idx:
                continue
            reproject(
                rasterio.band(src, band_idx),
                out,
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.nearest,
                dst_nodata=np.nan,
            )
            out[out == nodata] = np.nan
            out[out == 0] = np.nan
            out[out == NO_DATA_VALUE] = np.nan
    return red, nir


def stats_from_tiff_dir(muni_dir: Path, min_valid_days: int) -> dict[str, np.ndarray]:
    """Stack daily municipal TIFFs, compute per-pixel temporal NDVI stats."""
    code = muni_dir.name
    dated: list[tuple[date, Path]] = []
    for p in list(muni_dir.glob("*.tiff")) + list(muni_dir.glob("*.tif")):
        d = _parse_tiff_date(p.stem, code)
        if d is not None:
            dated.append((d, p))
    dated.sort(key=lambda x: x[0])
    if not dated:
        return _empty_ndvi_stats()

    paths = [p for _, p in dated]
    with rasterio.open(paths[0]) as src0:
        height, width = src0.height, src0.width
        ref_transform = src0.transform
        ref_crs = src0.crs
        if ref_crs is None:
            raise ValueError(f"TIFF has no CRS: {paths[0]}")

    stack = np.full((len(paths), height, width), np.nan, dtype=np.float32)
    for t, path in enumerate(paths):
        red, nir = _read_red_nir_on_grid(path, ref_transform, ref_crs, height, width)
        stack[t] = ndvi_from_red_nir(red, nir)

    return temporal_ndvi_stats(stack, min_valid_days=min_valid_days)


def _worker_npy(args: tuple) -> dict[str, np.ndarray]:
    path, min_valid_days = args
    try:
        return stats_from_npy(Path(path), min_valid_days)
    except Exception as e:  # noqa: BLE001 — one bad file must not kill the pool
        print(f"  skip npy {path}: {e}", flush=True)
        return _empty_ndvi_stats()


def _worker_tiff(args: tuple) -> dict[str, np.ndarray]:
    path, min_valid_days = args
    try:
        return stats_from_tiff_dir(Path(path), min_valid_days)
    except Exception as e:  # noqa: BLE001
        print(f"  skip tiff {path}: {e}", flush=True)
        return _empty_ndvi_stats()


def accumulate_stats(
    jobs: list[tuple],
    worker,
    workers: int,
    desc: str,
) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {k: [] for k in ("min", "max", "median", "mean")}
    if workers <= 1:
        for job in tqdm(jobs, desc=desc):
            st = worker(job)
            for k, v in st.items():
                if v.size:
                    parts[k].append(v)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(worker, job) for job in jobs]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=desc):
                try:
                    st = fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"  worker failed: {e}", flush=True)
                    continue
                for k, v in st.items():
                    if v.size:
                        parts[k].append(v)
    return {
        k: (np.concatenate(vs) if vs else np.array([], dtype=np.float32))
        for k, vs in parts.items()
    }


def plot_histograms(
    stats: dict[str, np.ndarray],
    *,
    season: str,
    source: str,
    out_path: Path,
    bins: int,
    xlim: tuple[float, float],
) -> None:
    titles = {
        "min": "Per-pixel min NDVI",
        "max": "Per-pixel max NDVI",
        "median": "Per-pixel median NDVI",
        "mean": "Per-pixel mean NDVI",
    }
    order = ("min", "max", "median", "mean")
    colors = {
        "min": "#4c78a8",
        "max": "#f58518",
        "median": "#54a24b",
        "mean": "#e45756",
    }

    n_pix = int(stats["mean"].size)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    fig.suptitle(
        f"NDVI temporal stats — season {season}\n"
        f"{n_pix:,} pixels  ·  source: {source}",
        fontsize=12,
    )

    bin_edges = np.linspace(xlim[0], xlim[1], bins + 1)
    for ax, key in zip(axes.ravel(), order):
        vals = stats[key]
        vals = vals[np.isfinite(vals)]
        ax.hist(
            vals,
            bins=bin_edges,
            color=colors[key],
            edgecolor="white",
            linewidth=0.3,
            alpha=0.9,
        )
        if vals.size:
            med = float(np.median(vals))
            mean = float(np.mean(vals))
            ax.axvline(med, color="0.15", ls="--", lw=1.0, label=f"median={med:.3f}")
            ax.axvline(mean, color="0.35", ls=":", lw=1.0, label=f"mean={mean:.3f}")
            ax.legend(fontsize=8, frameon=False, loc="upper left")
        ax.set_title(titles[key])
        ax.set_ylabel("Pixel count")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[1]:
        ax.set_xlabel("NDVI")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot yearly per-pixel NDVI min/max/median/mean histograms."
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--year",
        type=int,
        help="Harvest year (maps to season folder YYYY-1-YYYY), e.g. 2021 -> 2020-2021",
    )
    g.add_argument(
        "--year-range",
        type=str,
        help="Season folder name YYYY-YYYY, e.g. 2020-2021",
    )
    p.add_argument(
        "--npy-root",
        type=Path,
        default=DEFAULT_NPY_ROOT,
        help=f"Root of municipal .npy files (default: {DEFAULT_NPY_ROOT})",
    )
    p.add_argument(
        "--tiff-root",
        type=Path,
        default=DEFAULT_TIFF_ROOT,
        help=f"Root of daily municipal TIFFs (default: {DEFAULT_TIFF_ROOT})",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for PNG output (default: {DEFAULT_OUT_DIR})",
    )
    p.add_argument(
        "-j",
        "--workers",
        type=int,
        default=1,
        help="Parallel municipality workers (default: 1)",
    )
    p.add_argument("--bins", type=int, default=80, help="Histogram bins (default: 80)")
    p.add_argument(
        "--xmin",
        type=float,
        default=-0.2,
        help="Histogram x-axis min (default: -0.2)",
    )
    p.add_argument(
        "--xmax",
        type=float,
        default=1.0,
        help="Histogram x-axis max (default: 1.0)",
    )
    p.add_argument(
        "--min-valid-days",
        type=int,
        default=1,
        help="Keep pixels with at least this many valid NDVI days (default: 1)",
    )
    p.add_argument(
        "--limit-munis",
        type=int,
        default=None,
        help="Only process the first N municipalities (smoke test)",
    )
    p.add_argument(
        "--force-tiff",
        action="store_true",
        help="Ignore .npy even if present; always read daily_tiff",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    season = resolve_season(args)
    npy_season = args.npy_root / season
    tiff_season = args.tiff_root / season

    npy_files = [] if args.force_tiff else list_muni_npy(npy_season)
    if npy_files:
        source = f"npy ({npy_season})"
        jobs = [(str(p), args.min_valid_days) for p in npy_files]
        if args.limit_munis is not None:
            jobs = jobs[: args.limit_munis]
        print(f"Using {len(jobs)} .npy files under {npy_season}")
        stats = accumulate_stats(jobs, _worker_npy, args.workers, desc="NPY munis")
    else:
        tiff_dirs = list_muni_tiff_dirs(tiff_season)
        if not tiff_dirs:
            raise SystemExit(
                f"No data found for season {season}.\n"
                f"  looked for npy:  {npy_season}\n"
                f"  looked for tiff: {tiff_season}"
            )
        source = f"daily_tiff ({tiff_season})"
        jobs = [(str(p), args.min_valid_days) for p in tiff_dirs]
        if args.limit_munis is not None:
            jobs = jobs[: args.limit_munis]
        print(f"Using {len(jobs)} daily_tiff municipality folders under {tiff_season}")
        print("(Tip: preprocess to .npy for much faster runs.)")
        stats = accumulate_stats(jobs, _worker_tiff, args.workers, desc="TIFF munis")

    if stats["mean"].size == 0:
        raise SystemExit("No valid NDVI pixels found.")

    out_path = args.output_dir / f"ndvi_histograms_{season}.png"
    plot_histograms(
        stats,
        season=season,
        source=source,
        out_path=out_path,
        bins=args.bins,
        xlim=(args.xmin, args.xmax),
    )

    # Quick numeric summary
    for key in ("min", "max", "median", "mean"):
        v = stats[key]
        v = v[np.isfinite(v)]
        print(
            f"  {key:6s}: n={v.size:,}  "
            f"mean={np.mean(v):.4f}  median={np.median(v):.4f}  "
            f"p05={np.percentile(v, 5):.4f}  p95={np.percentile(v, 95):.4f}"
        )


if __name__ == "__main__":
    main()
