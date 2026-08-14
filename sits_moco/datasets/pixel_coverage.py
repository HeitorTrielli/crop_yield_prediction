"""
Reusable per-pixel temporal coverage from daily municipal TIFFs.

Shared by QA scripts and (later) dataset sanitization / MoCo pretraining filters.

Rules match ``preprocess_daily_to_npy``:
  - valid day: band 1 finite, not 0, not nodata (-9999)
  - pixel kept in stack: valid on > 2 days total

Season window for month span: Oct..Mar (calendar months 10,11,12,1,2,3).

Example (training filter, later)::

    from pathlib import Path
    from datasets.pixel_coverage import keep_mask_path, load_keep_mask

    mask = load_keep_mask(
        keep_mask_path("results/pixel_temporal_coverage/masks", "2022-2023", "4100103")
    )
    # mask[i] aligns with row i in the municipal .npy (preprocess keep order)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NO_DATA_VALUE = -9999
MIN_VALID_DAYS_PREPROCESS = 3  # kept if count > 2

SEASON_RE = re.compile(r"^\d{4}-\d{4}$")
FNAME_RE = re.compile(r"^(\d+)_(\d{4})_(\d{2})_(\d{2})\.")

SEASON_MONTHS = (10, 11, 12, 1, 2, 3)
MONTH_BIT = {10: 1, 11: 2, 12: 4, 1: 8, 2: 16, 3: 32}
MONTH_LABEL = {
    10: "Oct",
    11: "Nov",
    12: "Dec",
    1: "Jan",
    2: "Feb",
    3: "Mar",
}
MONTH_LABELS = [(m, MONTH_LABEL[m]) for m in SEASON_MONTHS]

EARLY_BITS = 1 | 2 | 4  # Oct-Dec
LATE_BITS = 8 | 16 | 32  # Jan-Mar

DAY_BINS = [3, 5, 8, 11, 15, 20, 25, 30, 40, 10_000]
DAY_BIN_LABELS = [
    "3-4",
    "5-7",
    "8-10",
    "11-14",
    "15-19",
    "20-24",
    "25-29",
    "30-39",
    "40+",
]
MAX_DAYS_HIST = 64


def parse_exclude(parts: list[str]) -> set[str]:
    out: set[str] = set()
    for raw in parts:
        for tok in raw.split(","):
            s = tok.strip()
            if s:
                out.add(s)
    return out


def season_directories(root: Path) -> list[Path]:
    if SEASON_RE.match(root.name):
        return [root.resolve()]
    return sorted(
        p.resolve()
        for p in root.iterdir()
        if p.is_dir() and SEASON_RE.match(p.name)
    )


def list_daily_tiffs(muni_dir: Path) -> list[Path]:
    if not muni_dir.is_dir():
        return []
    return sorted(muni_dir.glob("*.tiff")) + sorted(muni_dir.glob("*.tif"))


def parse_daily_tiff_month(filename: str) -> int | None:
    mo = FNAME_RE.match(filename)
    if not mo:
        return None
    return int(mo.group(3))


def band_valid_mask(band: np.ndarray, nodata: float | int | None) -> np.ndarray:
    nd = NO_DATA_VALUE if nodata is None else nodata
    return (
        (band != nd)
        & (band != 0)
        & np.isfinite(band)
        & (band != NO_DATA_VALUE)
    )


def popcount_season_months(bits: np.ndarray) -> np.ndarray:
    x = bits.astype(np.uint8)
    return (
        (x & 1)
        + ((x >> 1) & 1)
        + ((x >> 2) & 1)
        + ((x >> 3) & 1)
        + ((x >> 4) & 1)
        + ((x >> 5) & 1)
    ).astype(np.uint8)


def pixel_passes_temporal_filter(
    n_valid_days: int | np.ndarray,
    n_season_months: int | np.ndarray,
    *,
    min_valid_days: int = 8,
    min_season_months: int = 4,
) -> bool | np.ndarray:
    return (n_valid_days >= min_valid_days) & (n_season_months >= min_season_months)


def keep_mask_path(mask_root: Path | str, season: str, municipality_code: str) -> Path:
    return Path(mask_root) / season / f"{municipality_code}_keep.npy"


def load_keep_mask(path: Path | str) -> np.ndarray:
    """Boolean mask over preprocess-kept pixels (row-major)."""
    return np.load(path).astype(bool, copy=False)


def save_keep_mask(path: Path | str, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mask.astype(np.bool_))


def digitize_valid_days(counts: np.ndarray) -> list[int]:
    hist, _ = np.histogram(counts, bins=DAY_BINS)
    return hist.astype(int).tolist()


def survival_from_joint(
    joint: np.ndarray,
    min_days: int,
    min_months: int,
    *,
    max_days: int = MAX_DAYS_HIST,
) -> int:
    d0 = min(max(min_days, 0), max_days)
    m0 = min(max(min_months, 0), 6)
    return int(joint[d0:, m0:].sum())


def survival_from_day_hist(
    day_hist: np.ndarray,
    min_days: int,
    *,
    max_days: int = MAX_DAYS_HIST,
) -> int:
    d0 = min(max(min_days, 0), max_days)
    return int(day_hist[d0:].sum())


def survival_from_month_hist(month_hist: np.ndarray, min_months: int) -> int:
    m0 = min(max(min_months, 0), 6)
    return int(month_hist[m0:].sum())


@dataclass
class MuniCoverageGrid:
    """Raw per-pixel validity accumulated from daily TIFF band 1."""

    counts: np.ndarray
    month_bits: np.ndarray
    keep: np.ndarray
    height: int
    width: int
    n_tiffs: int


@dataclass
class MuniPixelStats:
    season: str
    code: str
    n_tiffs: int
    height: int
    width: int
    n_kept: int
    n_early_only: int
    n_late_only: int
    n_both: int
    n_leq_2_months: int
    n_leq_3_months: int
    n_full_6: int
    n_no_oct_nov: int
    day_hist: list[int] = field(default_factory=lambda: [0] * len(DAY_BIN_LABELS))
    month_present_pixels: dict[str, int] = field(
        default_factory=lambda: {lab: 0 for _, lab in MONTH_LABELS}
    )
    mean_valid_days: float = 0.0
    median_valid_days: float = 0.0
    p10_valid_days: float = 0.0
    p90_valid_days: float = 0.0
    max_valid_days: int = 0
    n_pass: int = 0
    empty: bool = False
    error: str | None = None


def _empty_stats(season: str, code: str, *, n_tiffs: int = 0, error: str | None = None) -> MuniPixelStats:
    return MuniPixelStats(
        season=season,
        code=code,
        n_tiffs=n_tiffs,
        height=0,
        width=0,
        n_kept=0,
        n_early_only=0,
        n_late_only=0,
        n_both=0,
        n_leq_2_months=0,
        n_leq_3_months=0,
        n_full_6=0,
        n_no_oct_nov=0,
        day_hist=[0] * len(DAY_BIN_LABELS),
        month_present_pixels={lab: 0 for _, lab in MONTH_LABELS},
        empty=error is None,
        error=error,
    )


def scan_municipality_grid(muni_dir: Path) -> tuple[MuniCoverageGrid | None, str | None]:
    """
    Scan all daily TIFFs in a municipality folder.

    Returns (grid, error). grid is None when folder is empty or no kept pixels.
    """
    files = list_daily_tiffs(muni_dir)
    if not files:
        return None, None

    try:
        import rasterio
    except ImportError as e:
        return None, f"rasterio missing: {e}"

    with rasterio.open(files[0]) as src0:
        height, width = src0.height, src0.width

    counts = np.zeros((height, width), dtype=np.uint16)
    month_bits = np.zeros((height, width), dtype=np.uint8)
    n_used = 0

    for f in files:
        cal_month = parse_daily_tiff_month(f.name)
        if cal_month is None:
            continue
        bit = MONTH_BIT.get(cal_month, 0)
        with rasterio.open(f) as src:
            if src.height != height or src.width != width:
                continue
            band = src.read(1)
            nodata = src.nodata if src.nodata is not None else NO_DATA_VALUE
        ok = band_valid_mask(band, nodata)
        counts += ok.astype(np.uint16)
        if bit:
            month_bits |= np.where(ok, bit, 0).astype(np.uint8)
        n_used += 1

    keep = counts > (MIN_VALID_DAYS_PREPROCESS - 1)
    if not np.any(keep):
        return None, None

    return (
        MuniCoverageGrid(
            counts=counts,
            month_bits=month_bits,
            keep=keep,
            height=height,
            width=width,
            n_tiffs=n_used,
        ),
        None,
    )


def stats_from_grid(
    grid: MuniCoverageGrid,
    *,
    season: str,
    code: str,
    min_valid_days: int = 8,
    min_season_months: int = 4,
    keep_mask_path: Path | None = None,
) -> MuniPixelStats:
    """Summarize kept pixels and optionally write a boolean keep mask."""
    c = grid.counts[grid.keep].astype(np.int32)
    mb = grid.month_bits[grid.keep]
    n_mo = popcount_season_months(mb)

    has_early = (mb & EARLY_BITS) != 0
    has_late = (mb & LATE_BITS) != 0
    pass_mask = pixel_passes_temporal_filter(
        c, n_mo, min_valid_days=min_valid_days, min_season_months=min_season_months
    )

    if keep_mask_path is not None:
        save_keep_mask(keep_mask_path, pass_mask)

    month_present = {}
    for cal, lab in MONTH_LABELS:
        bit = MONTH_BIT[cal]
        month_present[lab] = int(((mb & bit) != 0).sum())

    return MuniPixelStats(
        season=season,
        code=code,
        n_tiffs=grid.n_tiffs,
        height=grid.height,
        width=grid.width,
        n_kept=int(grid.keep.sum()),
        n_pass=int(pass_mask.sum()),
        n_early_only=int((has_early & ~has_late).sum()),
        n_late_only=int((has_late & ~has_early).sum()),
        n_both=int((has_early & has_late).sum()),
        n_leq_2_months=int((n_mo <= 2).sum()),
        n_leq_3_months=int((n_mo <= 3).sum()),
        n_full_6=int((n_mo == 6).sum()),
        n_no_oct_nov=int(((mb & (1 | 2)) == 0).sum()),
        day_hist=digitize_valid_days(c),
        month_present_pixels=month_present,
        mean_valid_days=float(np.mean(c)),
        median_valid_days=float(np.median(c)),
        p10_valid_days=float(np.percentile(c, 10)),
        p90_valid_days=float(np.percentile(c, 90)),
        max_valid_days=int(np.max(c)),
        empty=False,
    )


def histograms_from_grid(
    grid: MuniCoverageGrid,
    *,
    min_valid_days: int = 8,
    min_season_months: int = 4,
    keep_mask_path: Path | None = None,
) -> dict:
    """Fine-grained histograms for survival analysis (0..MAX_DAYS_HIST days)."""
    c = grid.counts[grid.keep].astype(np.int32)
    mb = grid.month_bits[grid.keep]
    n_mo = popcount_season_months(mb)
    c_clip = np.clip(c, 0, MAX_DAYS_HIST)

    day_hist = np.bincount(c_clip, minlength=MAX_DAYS_HIST + 1).astype(np.int64)
    month_hist = np.bincount(n_mo, minlength=7).astype(np.int64)
    idx = c_clip * 7 + n_mo
    joint_flat = np.bincount(idx, minlength=(MAX_DAYS_HIST + 1) * 7)
    joint = joint_flat.reshape(MAX_DAYS_HIST + 1, 7)

    pass_mask = pixel_passes_temporal_filter(
        c, n_mo, min_valid_days=min_valid_days, min_season_months=min_season_months
    )
    if keep_mask_path is not None:
        save_keep_mask(keep_mask_path, pass_mask)

    has_early = (mb & EARLY_BITS) != 0
    has_late = (mb & LATE_BITS) != 0

    return {
        "n_kept": int(grid.keep.sum()),
        "n_pass": int(pass_mask.sum()),
        "day_hist": day_hist,
        "month_hist": month_hist,
        "joint_hist": joint,
        "early_only": int((has_early & ~has_late).sum()),
        "late_only": int((has_late & ~has_early).sum()),
        "mean_days": float(np.mean(c)),
        "median_days": float(np.median(c)),
    }


def analyze_municipality(
    season: str,
    code: str,
    muni_dir: Path | str,
    *,
    min_valid_days: int = 8,
    min_season_months: int = 4,
    keep_mask_path: Path | None = None,
) -> MuniPixelStats:
    """Full per-municipality stats (used by CLI workers)."""
    muni_dir = Path(muni_dir)
    grid, err = scan_municipality_grid(muni_dir)
    if err:
        return _empty_stats(season, code, n_tiffs=len(list_daily_tiffs(muni_dir)), error=err)
    if grid is None:
        return _empty_stats(season, code)
    try:
        return stats_from_grid(
            grid,
            season=season,
            code=code,
            min_valid_days=min_valid_days,
            min_season_months=min_season_months,
            keep_mask_path=keep_mask_path,
        )
    except Exception as e:  # noqa: BLE001
        return _empty_stats(season, code, n_tiffs=grid.n_tiffs, error=str(e))


def analyze_municipality_histograms(
    season: str,
    code: str,
    muni_dir: Path | str,
    *,
    min_valid_days: int = 8,
    min_season_months: int = 4,
    keep_mask_path: Path | None = None,
) -> dict:
    """Histogram-focused stats for survival plots."""
    muni_dir = Path(muni_dir)
    grid, err = scan_municipality_grid(muni_dir)
    base = {
        "season": season,
        "code": code,
        "n_tiffs": len(list_daily_tiffs(muni_dir)),
        "n_kept": 0,
        "n_pass": 0,
        "day_hist": np.zeros(MAX_DAYS_HIST + 1, dtype=np.int64),
        "month_hist": np.zeros(7, dtype=np.int64),
        "joint_hist": np.zeros((MAX_DAYS_HIST + 1, 7), dtype=np.int64),
        "early_only": 0,
        "late_only": 0,
        "empty": True,
        "error": err,
    }
    if err or grid is None:
        base["empty"] = err is None and grid is None
        return base

    try:
        h = histograms_from_grid(
            grid,
            min_valid_days=min_valid_days,
            min_season_months=min_season_months,
            keep_mask_path=keep_mask_path,
        )
        base.update(h)
        base["n_tiffs"] = grid.n_tiffs
        base["empty"] = False
        base["error"] = None
        return base
    except Exception as e:  # noqa: BLE001
        base["error"] = str(e)
        base["empty"] = False
        return base
