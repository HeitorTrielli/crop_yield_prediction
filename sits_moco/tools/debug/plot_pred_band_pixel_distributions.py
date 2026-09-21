#!/usr/bin/env python3
"""
Pixel NDVI + Xavier rain histograms for municipalities whose predicted
productivity falls in a band (default 3.5–3.7 t/ha).

Uses productivity scatter CSVs (municipal forecasts) to pick municipality
codes, then reads municipal .npy stacks:

  channels 0–9  spectral  (NDVI from red=2, NIR=6)
  channel 11    cumulative rain (mm from Oct 1 through each S2 date)
  channel 12    max consecutive dry days in the gap before that S2 date

Per pixel:
  NDVI min / max / median / mean over valid days
  cumulative rain = last finite ch11 (season total)
  dry days        = max finite ch12 (longest dry gap)

Examples:
  python scripts/plot_pred_band_pixel_distributions.py --year 2020 -j 6
  python scripts/plot_pred_band_pixel_distributions.py --year 2020,2021,2023 -j 6
  python scripts/plot_pred_band_pixel_distributions.py --all-years -j 6 --limit-munis 5
"""

from __future__ import annotations

import argparse
import importlib.util
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

_NDVI_MOD = Path(__file__).with_name("plot_yearly_ndvi_histograms.py")
_spec = importlib.util.spec_from_file_location("plot_yearly_ndvi_histograms", _NDVI_MOD)
_ndvi = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_ndvi)

NO_DATA_VALUE = _ndvi.NO_DATA_VALUE
NIR_BAND_IDX = _ndvi.NIR_BAND_IDX
RED_BAND_IDX = _ndvi.RED_BAND_IDX
harvest_to_season_folder = _ndvi.harvest_to_season_folder
ndvi_from_red_nir = _ndvi.ndvi_from_red_nir
temporal_ndvi_stats = _ndvi.temporal_ndvi_stats

import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_SCATTER_DIR = Path(
    "results/Productivity_20_21_22_23_24_Seed101010_tune_moco_parana_trial_004"
    "/spectral_xavier/figures/productivity_scatter"
)
DEFAULT_NPY_ROOT = Path("files/npy")
CUM_RAIN_CH = 11
DRY_DAYS_CH = 12
PRED_COL = "predicted_productivity_t_ha"
CODE_COL = "municipality_code"


def _finite_mask(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & (x != NO_DATA_VALUE)


def last_finite_along_time(x: np.ndarray) -> np.ndarray:
    """x [N, T] → last finite value per row (NaN if none)."""
    n, t = x.shape
    out = np.full(n, np.nan, dtype=np.float32)
    valid = _finite_mask(x)
    if t == 0 or not np.any(valid):
        return out
    rev = valid[:, ::-1]
    offset = np.argmax(rev, axis=1)
    has = rev[np.arange(n), offset]
    idx = t - 1 - offset
    rows = np.flatnonzero(has)
    out[rows] = x[rows, idx[rows]]
    return out


def stats_from_npy(path: Path, min_valid_days: int) -> dict[str, np.ndarray]:
    arr = np.load(path, mmap_mode="r")
    empty = np.array([], dtype=np.float32)
    out = {
        "min": empty,
        "max": empty,
        "median": empty,
        "mean": empty,
        "cum_rain": empty,
        "dry_days": empty,
    }
    if arr.ndim != 3 or arr.shape[2] < 7 or arr.shape[0] == 0:
        return out

    n_pix, _t, n_ch = arr.shape
    chunk = max(8_000, min(40_000, n_pix))
    parts: dict[str, list[np.ndarray]] = {k: [] for k in out}
    has_rain = n_ch > DRY_DAYS_CH

    for start in range(0, n_pix, chunk):
        block = np.asarray(arr[start : start + chunk], dtype=np.float32)
        red = block[:, :, RED_BAND_IDX]
        nir = block[:, :, NIR_BAND_IDX]
        ndvi = ndvi_from_red_nir(red, nir)
        st = temporal_ndvi_stats(ndvi.T, min_valid_days=min_valid_days)
        keep_n = st["mean"].size
        if keep_n == 0:
            continue
        for k in ("min", "max", "median", "mean"):
            parts[k].append(st[k])

        if has_rain:
            valid_count = np.sum(np.isfinite(ndvi), axis=1)
            keep = valid_count >= min_valid_days
            cum = last_finite_along_time(block[keep, :, CUM_RAIN_CH])
            dry = np.full(int(keep.sum()), np.nan, dtype=np.float32)
            dry_block = block[keep, :, DRY_DAYS_CH]
            dry_ok = _finite_mask(dry_block) & (dry_block >= 0)
            with np.errstate(all="ignore"):
                dry_block = np.where(dry_ok, dry_block, np.nan)
                dry = np.nanmax(dry_block, axis=1).astype(np.float32)
            parts["cum_rain"].append(cum)
            parts["dry_days"].append(dry)

    return {
        k: (np.concatenate(vs) if vs else empty.copy()) for k, vs in parts.items()
    }


def _worker(job: tuple) -> dict[str, np.ndarray]:
    path, min_valid_days = job
    try:
        return stats_from_npy(Path(path), min_valid_days)
    except Exception as e:  # noqa: BLE001
        print(f"  skip {path}: {e}", flush=True)
        empty = np.array([], dtype=np.float32)
        return {
            k: empty
            for k in ("min", "max", "median", "mean", "cum_rain", "dry_days")
        }


def load_codes(
    scatter_dir: Path,
    years: list[int],
    pred_min: float,
    pred_max: float,
) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for year in years:
        csv_path = scatter_dir / f"productivity_{year}_direct.csv"
        if not csv_path.is_file():
            raise SystemExit(f"Missing scatter CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        if PRED_COL not in df.columns or CODE_COL not in df.columns:
            raise SystemExit(f"{csv_path} needs {CODE_COL} and {PRED_COL}")
        pred = pd.to_numeric(df[PRED_COL], errors="coerce")
        mask = pred.between(pred_min, pred_max, inclusive="both")
        codes = (
            df.loc[mask, CODE_COL]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .tolist()
        )
        out[year] = sorted(set(codes))
        print(
            f"  {year}: {mask.sum()}/{len(df)} municipalities with "
            f"predicted in [{pred_min}, {pred_max}] t/ha"
        )
    return out


def resolve_npy(npy_root: Path, year: int, code: str) -> Path | None:
    season = harvest_to_season_folder(year)
    candidates = [
        npy_root / season / code / f"{code}.npy",
        npy_root / season / f"{code}.npy",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def accumulate(
    jobs: list[tuple],
    workers: int,
) -> dict[str, np.ndarray]:
    keys = ("min", "max", "median", "mean", "cum_rain", "dry_days")
    parts: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    if workers <= 1:
        iterator = tqdm(jobs, desc="Municipalities")
        results = (_worker(job) for job in iterator)
        for st in results:
            for k, v in st.items():
                if v.size:
                    parts[k].append(v)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_worker, job) for job in jobs]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Municipalities"):
                st = fut.result()
                for k, v in st.items():
                    if v.size:
                        parts[k].append(v)
    empty = np.array([], dtype=np.float32)
    return {k: (np.concatenate(vs) if vs else empty) for k, vs in parts.items()}


def _hist(ax, vals: np.ndarray, *, color: str, bins, xlabel: str, title: str) -> None:
    vals = vals[np.isfinite(vals)]
    ax.hist(vals, bins=bins, color=color, edgecolor="white", linewidth=0.3, alpha=0.9)
    if vals.size:
        med = float(np.median(vals))
        mean = float(np.mean(vals))
        ax.axvline(med, color="0.15", ls="--", lw=1.0, label=f"median={med:.3g}")
        ax.axvline(mean, color="0.35", ls=":", lw=1.0, label=f"mean={mean:.3g}")
        ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Pixel count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_figure(
    stats: dict[str, np.ndarray],
    *,
    title: str,
    out_path: Path,
    ndvi_bins: int,
    ndvi_xlim: tuple[float, float],
    rain_bins: int,
) -> None:
    n_pix = int(stats["mean"].size)
    fig, axes = plt.subplots(3, 2, figsize=(11, 10.5))
    fig.suptitle(f"{title}\n{n_pix:,} pixels", fontsize=12)

    ndvi_edges = np.linspace(ndvi_xlim[0], ndvi_xlim[1], ndvi_bins + 1)
    ndvi_specs = (
        (axes[0, 0], "min", "#4c78a8", "Per-pixel min NDVI"),
        (axes[0, 1], "max", "#f58518", "Per-pixel max NDVI"),
        (axes[1, 0], "median", "#54a24b", "Per-pixel median NDVI"),
        (axes[1, 1], "mean", "#e45756", "Per-pixel mean NDVI"),
    )
    for ax, key, color, ttl in ndvi_specs:
        _hist(
            ax,
            stats[key],
            color=color,
            bins=ndvi_edges,
            xlabel="NDVI",
            title=ttl,
        )

    cum = stats["cum_rain"]
    cum = cum[np.isfinite(cum)]
    if cum.size:
        hi = float(np.nanpercentile(cum, 99.5))
        lo = float(min(0.0, np.nanmin(cum)))
        rain_edges = np.linspace(lo, max(hi, lo + 1.0), rain_bins + 1)
    else:
        rain_edges = rain_bins
    _hist(
        axes[2, 0],
        stats["cum_rain"],
        color="#6b5b95",
        bins=rain_edges,
        xlabel="mm",
        title="Cumulative rain (last S2 date, mm)",
    )

    dry = stats["dry_days"]
    dry = dry[np.isfinite(dry)]
    if dry.size:
        dhi = float(np.nanpercentile(dry, 99.5))
        dry_edges = np.linspace(0.0, max(dhi, 1.0), rain_bins + 1)
    else:
        dry_edges = rain_bins
    _hist(
        axes[2, 1],
        stats["dry_days"],
        color="#8d6e63",
        bins=dry_edges,
        xlabel="days",
        title="Max consecutive dry days (Xavier, between S2 dates)",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def parse_years(args: argparse.Namespace) -> list[int]:
    if args.all_years:
        return [2020, 2021, 2022, 2023, 2024]
    if not args.year:
        raise SystemExit("Pass --year 2020 or --year 2020,2021 or --all-years")
    years = [int(x.strip()) for x in str(args.year).split(",") if x.strip()]
    if not years:
        raise SystemExit("No harvest years parsed from --year")
    return years


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "NDVI and rain histograms for pixels in municipalities predicted "
            "in a productivity band (default 3.5–3.7 t/ha)."
        )
    )
    p.add_argument(
        "--year",
        type=str,
        default=None,
        help="Harvest year(s), comma-separated (e.g. 2020 or 2020,2021,2023)",
    )
    p.add_argument(
        "--all-years",
        action="store_true",
        help="Use harvest years 2020–2024",
    )
    p.add_argument("--pred-min", type=float, default=3.5)
    p.add_argument("--pred-max", type=float, default=3.7)
    p.add_argument(
        "--scatter-dir",
        type=Path,
        default=DEFAULT_SCATTER_DIR,
        help="Directory with productivity_{year}_direct.csv",
    )
    p.add_argument(
        "--npy-root",
        type=Path,
        default=DEFAULT_NPY_ROOT,
        help=f"Municipal .npy root (default: {DEFAULT_NPY_ROOT})",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="PNG output dir (default: <scatter-dir>/../ceiling_pixel_distributions)",
    )
    p.add_argument("-j", "--workers", type=int, default=4)
    p.add_argument("--bins", type=int, default=80)
    p.add_argument("--xmin", type=float, default=-0.2)
    p.add_argument("--xmax", type=float, default=1.0)
    p.add_argument("--rain-bins", type=int, default=80)
    p.add_argument("--min-valid-days", type=int, default=3)
    p.add_argument(
        "--limit-munis",
        type=int,
        default=None,
        help="Cap municipalities per year (smoke test)",
    )
    p.add_argument(
        "--pool-years",
        action="store_true",
        help="One figure pooling all requested years instead of one PNG per year",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    years = parse_years(args)
    scatter_dir = args.scatter_dir
    out_dir = args.output_dir or (
        scatter_dir.parent / "ceiling_pixel_distributions"
    )
    pred_min, pred_max = float(args.pred_min), float(args.pred_max)
    band = f"{pred_min:g}-{pred_max:g}"

    print(f"Scatter CSVs: {scatter_dir}")
    print(f"NPY root:     {args.npy_root}")
    codes_by_year = load_codes(scatter_dir, years, pred_min, pred_max)

    jobs_by_year: dict[int, list[tuple]] = {}
    missing = 0
    for year, codes in codes_by_year.items():
        if args.limit_munis is not None:
            codes = codes[: args.limit_munis]
        jobs = []
        for code in codes:
            path = resolve_npy(args.npy_root, year, code)
            if path is None:
                missing += 1
                continue
            jobs.append((str(path), args.min_valid_days))
        jobs_by_year[year] = jobs
        print(f"  {year}: {len(jobs)} .npy files ({missing} missing so far)")
    if missing:
        print(f"Warning: {missing} municipality-years had no .npy")

    def run_jobs(jobs: list[tuple], label: str) -> dict[str, np.ndarray]:
        if not jobs:
            raise SystemExit(f"No .npy files for {label}")
        print(f"Processing {len(jobs)} municipalities ({label})")
        stats = accumulate(jobs, args.workers)
        if stats["mean"].size == 0:
            raise SystemExit(f"No valid pixels for {label}")
        for key in ("min", "max", "median", "mean", "cum_rain", "dry_days"):
            v = stats[key]
            v = v[np.isfinite(v)]
            if v.size == 0:
                print(f"  {key:10s}: empty")
                continue
            print(
                f"  {key:10s}: n={v.size:,}  mean={np.mean(v):.4g}  "
                f"median={np.median(v):.4g}  p05={np.percentile(v, 5):.4g}  "
                f"p95={np.percentile(v, 95):.4g}"
            )
        return stats

    if args.pool_years:
        jobs = [j for year in years for j in jobs_by_year[year]]
        stats = run_jobs(jobs, f"pooled {years}")
        plot_figure(
            stats,
            title=(
                f"Pixels in municipalities with predicted yield "
                f"{pred_min:g}–{pred_max:g} t/ha  ·  harvest {years}"
            ),
            out_path=out_dir / f"pred_{band}_ndvi_rain_pooled.png",
            ndvi_bins=args.bins,
            ndvi_xlim=(args.xmin, args.xmax),
            rain_bins=args.rain_bins,
        )
        return

    for year in years:
        stats = run_jobs(jobs_by_year[year], str(year))
        plot_figure(
            stats,
            title=(
                f"Pixels in municipalities with predicted yield "
                f"{pred_min:g}–{pred_max:g} t/ha  ·  harvest {year}"
            ),
            out_path=out_dir / f"pred_{band}_ndvi_rain_{year}.png",
            ndvi_bins=args.bins,
            ndvi_xlim=(args.xmin, args.xmax),
            rain_bins=args.rain_bins,
        )


if __name__ == "__main__":
    main()
