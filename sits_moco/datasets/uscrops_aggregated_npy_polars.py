"""
Dataset class for aggregated regression: groups pixels by municipality.
Polars-optimized version for faster index loading.

Each sample = pixel metadata for one municipality + municipality-level yield target.
Training code loads pixels in chunks on-demand.
"""

from collections import OrderedDict, defaultdict
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
import math
import random
from typing import Iterable, Optional

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from .feature_layout import (
    feature_layout_choices,
    normalize_feature_layout,
    resolve_feature_layout,
)
from .pixel_transform import (
    DOY_CHANNEL,
    NUM_SPECTRAL_CHANNELS,
    PixelTransform,
    scale_xavier_rain_channels,
)


def _doy_to_season_month(doy, reference_date):
    """DOY 1 = reference_date (e.g. 2022-10-01). Use date + timedelta to get calendar month, then map to season 1..6."""
    d = reference_date + timedelta(days=int(doy) - 1)
    cal = d.month
    ref_month = reference_date.month
    season_cal_months = [(ref_month - 1 + i) % 12 + 1 for i in range(6)]
    if cal not in season_cal_months:
        return 0
    return season_cal_months.index(cal) + 1


@lru_cache(maxsize=8)
def _season_month_lut(reference_date: date, max_doy: int = 400) -> np.ndarray:
    """Lookup table: DOY index → season month (0 = outside 6-month window, else 1..6)."""
    lut = np.zeros(max_doy + 1, dtype=np.int8)
    for d in range(1, max_doy + 1):
        lut[d] = _doy_to_season_month(d, reference_date)
    return lut


def _season_months_from_doys(doys: np.ndarray, reference_date: date) -> np.ndarray:
    """Vectorized season-month mapping for any-shaped DOY array."""
    lut = _season_month_lut(reference_date)
    doy_i = np.asarray(doys, dtype=np.int64)
    doy_i = np.clip(doy_i, 0, lut.shape[0] - 1)
    return lut[doy_i]


# Missing markers stored by preprocess_daily_to_npy.
NO_DATA_VALUE = -9999
# Daily .npy DOY is season-relative with day 1 = Oct 1 (see season_start_from_year_range).
DEFAULT_SEASON_REFERENCE_DATE = date(2000, 10, 1)


def pixel_temporal_keep_mask(
    data: np.ndarray,
    *,
    min_images: int = 0,
    min_months: int = 0,
    reference_date: Optional[date] = None,
) -> np.ndarray:
    """
    Keep pixels with enough valid daily images and distinct season months.

    A day counts as an image if any spectral band (0..9) is finite and not 0/-9999.
    Months are season months 1..6 derived from the DOY channel (index 10).
    """
    if data.ndim != 3 or data.shape[0] == 0:
        return np.zeros(data.shape[0] if data.ndim >= 1 else 0, dtype=bool)

    min_images = max(0, int(min_images))
    min_months = max(0, int(min_months))
    if min_images <= 0 and min_months <= 0:
        return np.ones(data.shape[0], dtype=bool)

    spec = data[:, :, :NUM_SPECTRAL_CHANNELS]
    day_valid = np.any(
        (spec != NO_DATA_VALUE) & (spec != 0) & np.isfinite(spec),
        axis=2,
    )
    n_images = day_valid.sum(axis=1)
    keep = n_images >= min_images if min_images > 0 else np.ones(
        data.shape[0], dtype=bool
    )

    if min_months > 0:
        ref = reference_date or DEFAULT_SEASON_REFERENCE_DATE
        season_m = _season_months_from_doys(data[:, :, DOY_CHANNEL], ref)
        masked_months = np.where(day_valid, season_m, 0)
        n_months = np.zeros(data.shape[0], dtype=np.int32)
        for month in range(1, 7):
            n_months += np.any(masked_months == month, axis=1)
        keep &= n_months >= min_months

    return keep


# Default: no coverage filter (use all municipality–year rows).
# Optional suggested band for S2 footprint vs IBGE planted area:
#   --min-coverage-ratio 0.8 --max-coverage-ratio 1.2
DEFAULT_MIN_COVERAGE_RATIO = None
DEFAULT_MAX_COVERAGE_RATIO = None
SUGGESTED_MIN_COVERAGE_RATIO = 0.8
SUGGESTED_MAX_COVERAGE_RATIO = 1.2
COVERAGE_RATIO_COL = "coverage_ratio"
PRODUCTIVITY_DEV_SOURCE_COL = "yield_t_ha"
PRODUCTIVITY_DEV_COL = "yield_t_ha_dev"


def ensure_productivity_dev_column(
    yield_df: pl.DataFrame,
    *,
    harvest_years: Iterable[int] | None = None,
    source_column: str = PRODUCTIVITY_DEV_SOURCE_COL,
    out_column: str = PRODUCTIVITY_DEV_COL,
) -> tuple[pl.DataFrame, dict]:
    """
    Add ``out_column`` = ``source_column`` − per-year train mean.

    Baselines use ``split == train`` only (after any coverage filter already
    applied by the caller). Valid/test rows reuse those year means so there is
    no leakage from held-out labels. Years with no train rows fall back to the
    global train mean.

    If ``out_column`` already exists, returns the frame unchanged.
    """
    if out_column in yield_df.columns:
        return yield_df, {"already_present": True, "out_column": out_column}

    if source_column not in yield_df.columns:
        raise ValueError(
            f"Cannot derive {out_column!r}: missing source column {source_column!r}. "
            f"Available: {list(yield_df.columns)}"
        )
    if "split" not in yield_df.columns:
        raise ValueError(
            f"Cannot derive {out_column!r}: yield CSV has no 'split' column. "
            "Run add_train_valid_test_split.py first."
        )
    if "year" not in yield_df.columns:
        raise ValueError(
            f"Cannot derive {out_column!r}: yield CSV has no 'year' column."
        )

    train_df = yield_df.filter(pl.col("split") == "train")
    if harvest_years is not None:
        hset = {int(y) for y in harvest_years}
        train_df = train_df.filter(pl.col("year").is_in(sorted(hset)))

    train_vals = (
        train_df.select(source_column).drop_nulls()[source_column].to_list()
    )
    train_vals = [float(v) for v in train_vals if v is not None and v != ""]
    if not train_vals:
        raise ValueError(
            f"Cannot derive {out_column!r}: no valid train values in {source_column!r}"
        )
    global_train_mean = float(np.mean(train_vals))

    year_means: dict[int, float] = {}
    # Prefer group_by (Polars >=0.19); fall back for older installs.
    _groupby = getattr(train_df, "group_by", None) or getattr(train_df, "groupby")
    year_stats = _groupby("year").agg(
        pl.col(source_column).drop_nulls().mean().alias("year_baseline")
    )
    for row in year_stats.iter_rows(named=True):
        year_val = row["year"]
        mean_y = row["year_baseline"]
        if year_val is None or mean_y is None:
            continue
        try:
            year_means[int(year_val)] = float(mean_y)
        except (TypeError, ValueError):
            continue

    baseline_df = pl.DataFrame(
        {
            "year": list(year_means.keys()),
            "_year_baseline": list(year_means.values()),
        }
    )
    if len(baseline_df) == 0:
        out_df = yield_df.with_columns(
            (pl.col(source_column).cast(pl.Float64) - global_train_mean).alias(
                out_column
            )
        )
    else:
        out_df = (
            yield_df.join(baseline_df, on="year", how="left")
            .with_columns(
                pl.col("_year_baseline").fill_null(global_train_mean).alias(
                    "_year_baseline"
                )
            )
            .with_columns(
                (
                    pl.col(source_column).cast(pl.Float64) - pl.col("_year_baseline")
                ).alias(out_column)
            )
            .drop("_year_baseline")
        )

    meta = {
        "already_present": False,
        "out_column": out_column,
        "source_column": source_column,
        "baseline": "year_train_mean",
        "global_train_mean": global_train_mean,
        "year_train_means": {str(k): v for k, v in sorted(year_means.items())},
        "n_train_for_baseline": len(train_vals),
    }
    print(
        f"  Derived {out_column} = {source_column} - year_train_mean "
        f"(global_train_mean={global_train_mean:.4f}, years={sorted(year_means)})"
    )
    return out_df, meta


def filter_yield_df_by_coverage(
    yield_df: pl.DataFrame,
    *,
    min_ratio: float | None = DEFAULT_MIN_COVERAGE_RATIO,
    max_ratio: float | None = DEFAULT_MAX_COVERAGE_RATIO,
    column: str = COVERAGE_RATIO_COL,
) -> pl.DataFrame:
    """
    Keep rows whose ``coverage_ratio`` is inside ``[min_ratio, max_ratio]``.

    If ``min_ratio`` and ``max_ratio`` are both None, returns ``yield_df`` unchanged.
    If the column is missing while a filter is requested, raises ``ValueError``.
    """
    if min_ratio is None and max_ratio is None:
        return yield_df
    if column not in yield_df.columns:
        raise ValueError(
            f"Yield CSV missing {column!r} required for coverage filtering. "
            "Run scripts/compute_imagery_area_coverage.py to add it, "
            "or pass --no-coverage-filter."
        )

    before = len(yield_df)
    expr = pl.col(column).is_not_null()
    if min_ratio is not None:
        expr = expr & (pl.col(column) >= float(min_ratio))
    if max_ratio is not None:
        expr = expr & (pl.col(column) <= float(max_ratio))
    filtered = yield_df.filter(expr)
    after = len(filtered)
    lo = "-inf" if min_ratio is None else f"{min_ratio:g}"
    hi = "+inf" if max_ratio is None else f"{max_ratio:g}"
    print(
        f"  Coverage filter {column} in [{lo}, {hi}]: "
        f"kept {after}/{before} rows (dropped {before - after})"
    )
    return filtered


def filter_yield_pandas_by_coverage(
    yield_df,
    *,
    min_ratio: float | None = DEFAULT_MIN_COVERAGE_RATIO,
    max_ratio: float | None = DEFAULT_MAX_COVERAGE_RATIO,
    column: str = COVERAGE_RATIO_COL,
):
    """Pandas equivalent of :func:`filter_yield_df_by_coverage`."""
    if min_ratio is None and max_ratio is None:
        return yield_df
    if column not in yield_df.columns:
        raise ValueError(
            f"Yield CSV missing {column!r} required for coverage filtering. "
            "Run scripts/compute_imagery_area_coverage.py to add it, "
            "or pass --no-coverage-filter."
        )
    before = len(yield_df)
    mask = yield_df[column].notna()
    if min_ratio is not None:
        mask &= yield_df[column] >= float(min_ratio)
    if max_ratio is not None:
        mask &= yield_df[column] <= float(max_ratio)
    filtered = yield_df.loc[mask].copy()
    after = len(filtered)
    lo = "-inf" if min_ratio is None else f"{min_ratio:g}"
    hi = "+inf" if max_ratio is None else f"{max_ratio:g}"
    print(
        f"  Coverage filter {column} in [{lo}, {hi}]: "
        f"kept {after}/{before} rows (dropped {before - after})"
    )
    return filtered


# Mid-band train undersample defaults (disabled unless keep_fraction is set).
# Suggested: keep ~0.52 of yield ∈ [3.0, 4.25) so mid bins average ~70 rows
# while preserving relative shape (see entering-productivity-density canvas).
DEFAULT_TRAIN_MID_YIELD_KEEP_FRACTION = None
DEFAULT_TRAIN_MID_YIELD_LO = 3.0
DEFAULT_TRAIN_MID_YIELD_HI = 4.25
DEFAULT_TRAIN_MID_YIELD_BIN_WIDTH = 0.25
PRODUCTIVITY_YIELD_COL = "yield_t_ha"


def undersample_train_mid_yield_band(
    yield_df: pl.DataFrame,
    *,
    keep_fraction: float | None = DEFAULT_TRAIN_MID_YIELD_KEEP_FRACTION,
    yield_lo: float = DEFAULT_TRAIN_MID_YIELD_LO,
    yield_hi: float = DEFAULT_TRAIN_MID_YIELD_HI,
    bin_width: float = DEFAULT_TRAIN_MID_YIELD_BIN_WIDTH,
    yield_column: str = PRODUCTIVITY_YIELD_COL,
    seed: int = 101010,
    split_column: str = "split",
    train_value: str = "train",
) -> pl.DataFrame:
    """
    Randomly downscale **train** rows whose yield is in ``[yield_lo, yield_hi)``.

    Uses a uniform keep fraction within each ``bin_width`` slice so the mid-band
    histogram shape is preserved (not a hard per-bin cap). Valid/test rows and
    train rows outside the band are never dropped.

    If ``keep_fraction`` is None or >= 1, returns ``yield_df`` unchanged.
    """
    if keep_fraction is None:
        return yield_df
    keep_fraction = float(keep_fraction)
    if keep_fraction >= 1.0:
        return yield_df
    if keep_fraction < 0.0:
        raise ValueError(
            f"train mid-yield keep_fraction must be in [0, 1], got {keep_fraction}"
        )
    if yield_column not in yield_df.columns:
        raise ValueError(
            f"Yield CSV missing {yield_column!r} required for mid-band undersampling. "
            f"Available: {list(yield_df.columns)}"
        )
    if bin_width <= 0:
        raise ValueError(f"bin_width must be > 0, got {bin_width}")

    n = len(yield_df)
    if n == 0:
        return yield_df

    yields = yield_df[yield_column].to_list()
    if split_column in yield_df.columns:
        splits = yield_df[split_column].to_list()
    else:
        splits = [train_value] * n

    bins: dict[float, list[int]] = defaultdict(list)
    keep_mask = [True] * n
    for i in range(n):
        if splits[i] != train_value:
            continue
        y_raw = yields[i]
        if y_raw is None or y_raw == "" or y_raw == "-":
            continue
        try:
            yf = float(y_raw)
        except (TypeError, ValueError):
            continue
        if math.isnan(yf):
            continue
        if yield_lo <= yf < yield_hi:
            bin_key = math.floor(yf / float(bin_width)) * float(bin_width)
            bins[bin_key].append(i)
            keep_mask[i] = False

    rng = random.Random(int(seed))
    kept_mid = 0
    dropped_mid = 0
    for _bin_key, indices in sorted(bins.items()):
        rng.shuffle(indices)
        k = int(round(len(indices) * keep_fraction))
        k = max(0, min(len(indices), k))
        for idx in indices[:k]:
            keep_mask[idx] = True
        kept_mid += k
        dropped_mid += len(indices) - k

    before = n
    filtered = (
        yield_df.with_columns(pl.Series("_mid_yield_keep", keep_mask))
        .filter(pl.col("_mid_yield_keep"))
        .drop("_mid_yield_keep")
    )
    after = len(filtered)
    print(
        f"  Train mid-yield undersample {yield_column} in [{yield_lo:g}, {yield_hi:g}) "
        f"keep_fraction={keep_fraction:g} bin_width={bin_width:g} seed={seed}: "
        f"mid kept {kept_mid}/{kept_mid + dropped_mid} "
        f"(dropped {dropped_mid}); rows {after}/{before}"
    )
    return filtered


def filter_yield_pandas_by_mid_yield_undersample(
    yield_df,
    *,
    keep_fraction: float | None = DEFAULT_TRAIN_MID_YIELD_KEEP_FRACTION,
    yield_lo: float = DEFAULT_TRAIN_MID_YIELD_LO,
    yield_hi: float = DEFAULT_TRAIN_MID_YIELD_HI,
    bin_width: float = DEFAULT_TRAIN_MID_YIELD_BIN_WIDTH,
    yield_column: str = PRODUCTIVITY_YIELD_COL,
    seed: int = 101010,
    split_column: str = "split",
    train_value: str = "train",
):
    """Pandas wrapper around :func:`undersample_train_mid_yield_band`."""
    if keep_fraction is None or float(keep_fraction) >= 1.0:
        return yield_df
    out = undersample_train_mid_yield_band(
        pl.from_pandas(yield_df),
        keep_fraction=keep_fraction,
        yield_lo=yield_lo,
        yield_hi=yield_hi,
        bin_width=bin_width,
        yield_column=yield_column,
        seed=seed,
        split_column=split_column,
        train_value=train_value,
    )
    return out.to_pandas()


def _indices_first_n_months(row, num_periods, reference_date):
    """Return indices of rows in (T, C) where season month is in 1..num_periods (DOY at channel 10)."""
    season_m = _season_months_from_doys(row[:, DOY_CHANNEL], reference_date)
    valid = (season_m >= 1) & (season_m <= num_periods)
    return np.nonzero(valid)[0]


class USCropsAggregatedNPY(Dataset):
    """
    Dataset that groups pixels by municipality for yield prediction.
    Polars-optimized version for faster index loading.

    Returns pixel metadata (lazy loading) and municipality-level yield target.
    Training code loads pixels in chunks on-demand.

    ``feature_layout`` selects the STNet input stack explicitly; see ``datasets/feature_layout.py``.
    """

    def __init__(
        self,
        mode,
        root,
        yield_csv,
        year=2023,
        sequencelength=6,
        dataaug=None,
        randomchoice=False,
        interp=False,
        seed=27,
        preload_ram=False,
        target_mean=None,
        target_std=None,
        harvest_years: Optional[Iterable[int]] = None,
        feature_layout: str = "spectral",
        target_column: str | list[str] | tuple[str, ...] = "production_t",
        npy_cache_size: int = 20,
        npy_defer_copy: bool = True,  # ignored; always deferred (kept for API compat)
        min_coverage_ratio: float | None = DEFAULT_MIN_COVERAGE_RATIO,
        max_coverage_ratio: float | None = DEFAULT_MAX_COVERAGE_RATIO,
        min_images: int = 0,
        min_months: int = 0,
        season_reference_date: Optional[date] = None,
        train_mid_yield_keep_fraction: float | None = DEFAULT_TRAIN_MID_YIELD_KEEP_FRACTION,
        train_mid_yield_lo: float = DEFAULT_TRAIN_MID_YIELD_LO,
        train_mid_yield_hi: float = DEFAULT_TRAIN_MID_YIELD_HI,
        train_mid_yield_bin_width: float = DEFAULT_TRAIN_MID_YIELD_BIN_WIDTH,
    ):
        super(USCropsAggregatedNPY, self).__init__()

        mode = mode.lower()
        assert mode in ["train", "valid", "eval", "test", "all"]

        self.feature_layout = normalize_feature_layout(feature_layout)
        _lay = resolve_feature_layout(self.feature_layout)
        self.input_feature_dim = int(_lay["input_dim"])
        self._extra_channels_slice: tuple[int, int] | None = _lay[
            "extra_channels_slice"
        ]
        # Backward-compatible flag for logging / meta (any Xavier extras starting at ch 11)
        self.use_xavier = (
            self._extra_channels_slice is not None
            and int(self._extra_channels_slice[0]) == 11
        )

        self.root = Path(root)
        self.year = year
        self.mode = mode
        self.sequencelength = sequencelength
        self.rc = randomchoice
        self.interp = interp
        self.min_coverage_ratio = min_coverage_ratio
        self.max_coverage_ratio = max_coverage_ratio
        self.train_mid_yield_keep_fraction = train_mid_yield_keep_fraction
        self.train_mid_yield_lo = float(train_mid_yield_lo)
        self.train_mid_yield_hi = float(train_mid_yield_hi)
        self.train_mid_yield_bin_width = float(train_mid_yield_bin_width)
        self.min_images = max(0, int(min_images))
        self.min_months = max(0, int(min_months))
        self.season_reference_date = (
            season_reference_date or DEFAULT_SEASON_REFERENCE_DATE
        )
        self._temporal_filter_enabled = self.min_images > 0 or self.min_months > 0

        # Store normalization parameters (scalar or per-head vectors)
        self.target_mean = target_mean if target_mean is not None else 0.0
        self.target_std = target_std if target_std is not None else 1.0
        self.normalize_targets = target_mean is not None and target_std is not None

        if isinstance(target_column, (list, tuple)):
            self.target_columns = list(target_column)
        else:
            self.target_columns = [target_column]
        self.num_outputs = len(self.target_columns)
        # Backward-compatible single-column attribute
        self.target_column = (
            self.target_columns[0]
            if self.num_outputs == 1
            else list(self.target_columns)
        )

        # Load yield targets with Polars
        print(f"Loading yield targets from {yield_csv}...")
        # Handle null values: treat '-' and empty strings as null (same as pandas version)
        yield_df = pl.read_csv(
            yield_csv,
            null_values=["-", "", "nan", "NaN", "null", "NULL"],
            infer_schema_length=10000,  # Increase to better infer schema with null values
        )

        municipality_code_col = "municipality_code"
        cols = list(yield_df.columns)
        if municipality_code_col not in cols:
            raise ValueError(
                f"Yield CSV missing {municipality_code_col!r}. Available: {cols}"
            )

        # Convert municipality_code to string and filter by split
        yield_df = yield_df.with_columns(
            pl.col(municipality_code_col).cast(pl.Utf8).alias(municipality_code_col)
        )

        yield_df = filter_yield_df_by_coverage(
            yield_df,
            min_ratio=min_coverage_ratio,
            max_ratio=max_coverage_ratio,
        )

        # Restrict years before mid-band undersample so keep/drop matches the
        # municipality–years that actually enter training.
        if harvest_years is not None and "year" in yield_df.columns:
            hset = sorted({int(y) for y in harvest_years})
            before_y = len(yield_df)
            yield_df = yield_df.filter(pl.col("year").is_in(hset))
            print(
                f"  Restricted yield CSV to harvest years {hset}: "
                f"{len(yield_df)}/{before_y} rows"
            )

        # Train-only mid-band undersample (before productivity_dev baselines / split filter).
        yield_df = undersample_train_mid_yield_band(
            yield_df,
            keep_fraction=train_mid_yield_keep_fraction,
            yield_lo=self.train_mid_yield_lo,
            yield_hi=self.train_mid_yield_hi,
            bin_width=self.train_mid_yield_bin_width,
            seed=int(seed),
        )

        self.productivity_dev_meta = None
        if PRODUCTIVITY_DEV_COL in self.target_columns:
            yield_df, self.productivity_dev_meta = ensure_productivity_dev_column(
                yield_df,
                harvest_years=harvest_years,
            )

        cols = list(yield_df.columns)
        for yield_col in self.target_columns:
            if yield_col not in cols:
                raise ValueError(f"Yield CSV missing {yield_col!r}. Available: {cols}")
        self.municipality_code_column = municipality_code_col

        # Filter by split (skip filter when mode is "all" to use train/valid/test entries)
        if "split" in yield_df.columns and mode != "all":
            split_mapping = {
                "train": "train",
                "valid": "valid",
                "eval": "eval",
                "test": "test",
            }
            target_split = split_mapping.get(mode, mode)
            yield_df = yield_df.filter(pl.col("split") == target_split)
            print(f"Filtered to {target_split} split: {len(yield_df)} rows")
        elif mode == "all" and "split" in yield_df.columns:
            print(f"Using all splits (train/valid/test): {len(yield_df)} rows")
        elif mode != "all":
            raise ValueError(
                f"Yield CSV has no 'split' column; cannot filter for mode={mode!r}. "
                "Run add_train_valid_test_split.py or use mode='all'."
            )

        harvest_years_set = (
            set(int(y) for y in harvest_years) if harvest_years is not None else None
        )

        self.split_pairs = set()
        if "year" in yield_df.columns and self.year is None:
            select_cols = [municipality_code_col, "year", *self.target_columns]
            trip = yield_df.select(select_cols).to_dict()
            self.yield_map = {}
            n_rows = len(trip[municipality_code_col])
            for i in range(n_rows):
                muni_code = trip[municipality_code_col][i]
                year_val = trip["year"][i]
                if year_val is None:
                    continue
                try:
                    yi = int(year_val)
                except (ValueError, TypeError):
                    continue
                vals = []
                for col in self.target_columns:
                    yld = trip[col][i]
                    if yld is None or yld == "-" or yld == "":
                        vals.append(0.0)
                    else:
                        try:
                            vals.append(float(yld))
                        except (ValueError, TypeError):
                            vals.append(0.0)
                self.split_pairs.add((str(muni_code), yi))
                self.yield_map[(str(muni_code), yi)] = (
                    vals[0] if self.num_outputs == 1 else vals
                )
            if harvest_years_set is not None:
                self.split_pairs = {
                    (m, y) for m, y in self.split_pairs if y in harvest_years_set
                }
                self.yield_map = {
                    k: v for k, v in self.yield_map.items() if k[1] in harvest_years_set
                }
                print(
                    f"  Restricted to harvest years {sorted(harvest_years_set)}: "
                    f"{len(self.split_pairs)} (municipality, year) pairs"
                )
            print(
                f"Loaded yield targets for {len(self.yield_map)} (municipality, year) entries"
                f" ({self.num_outputs} head(s): {self.target_columns})"
            )
        else:
            select_cols = [municipality_code_col, *self.target_columns]
            yield_dict = yield_df.select(select_cols).to_dict()
            self.yield_map = {}
            for i, muni_code in enumerate(yield_dict[municipality_code_col]):
                vals = []
                for col in self.target_columns:
                    yld = yield_dict[col][i]
                    if yld is None or yld == "-" or yld == "":
                        vals.append(0.0)
                    else:
                        try:
                            vals.append(float(yld))
                        except (ValueError, TypeError):
                            vals.append(0.0)
                self.yield_map[str(muni_code)] = (
                    vals[0] if self.num_outputs == 1 else vals
                )
            if harvest_years_set is not None:
                raise ValueError(
                    "harvest_years filter requires year=None and a 'year' column in the yield CSV"
                )
            print(
                f"Loaded yield targets for {len(self.yield_map)} municipality entries"
                f" ({self.num_outputs} head(s): {self.target_columns})"
            )

        # Build cached mapping: year -> list of year-dir Paths, or detect "flat" structure
        # Flat = one season folder (e.g. 2022-2023) with municipality subdirs directly inside (no year subdirs)
        self._year_dirs_by_end_year = defaultdict(list)
        self._flat_season = False  # root is the data dir, subdirs are muni codes
        for item in self.root.iterdir():
            if not item.is_dir():
                continue
            # Only treat as "year dir" if 4-digit year or YYYY-YYYY range (not 7-digit muni codes)
            if item.name.isdigit() and len(item.name) == 4:
                self._year_dirs_by_end_year[int(item.name)].append(item)
            elif "-" in item.name:
                parts = item.name.split("-")
                if (
                    len(parts) == 2
                    and parts[0].isdigit()
                    and parts[1].isdigit()
                    and len(parts[0]) == 4
                    and len(parts[1]) == 4
                ):
                    _y1, y2 = int(parts[0]), int(parts[1])
                    # Harvest year = second year; imagery for that season lives under Y1-Y2/
                    self._year_dirs_by_end_year[y2].append(item)
        if not self._year_dirs_by_end_year:
            # No year subdirs -> root is the season folder, subdirs are municipalities
            self._flat_season = True

        # Sorted (year, path) for multi-year scan
        self._year_dir_tuples = sorted(
            (year, d)
            for year, dirs in self._year_dirs_by_end_year.items()
            for d in dirs
        )

        # No index needed - load pixel counts directly from .npy files
        print(f"Loading pixel counts from .npy files...")
        municipality_pixel_counts = (
            {}
        )  # {(muni_code, year): num_pixels} or {muni_code: num_pixels}
        municipality_years = {}  # {(muni_code, year): year} or {muni_code: year}

        if self.split_pairs:
            target_muni_codes = sorted(set(m for m, _ in self.split_pairs))
        else:
            target_muni_codes = sorted(self.yield_map.keys())

        missing_municipalities = []
        if self._flat_season:
            # Root = one season folder, subdirs = municipality codes (e.g. 2022-2023/4100103/4100103.npy)
            for item in self.root.iterdir():
                if not item.is_dir() or not item.name.isdigit():
                    continue
                muni_code = item.name
                muni_npy_file = item / f"{muni_code}.npy"
                if not muni_npy_file.exists():
                    continue
                if self.year is not None:
                    if muni_code not in self.yield_map:
                        continue
                    try:
                        data = np.load(muni_npy_file, mmap_mode="r")
                        num_pixels = len(data)
                        if num_pixels > 0:
                            municipality_pixel_counts[muni_code] = num_pixels
                            municipality_years[muni_code] = self.year
                    except Exception:
                        missing_municipalities.append(muni_code)
                else:
                    # Multi-year: same .npy used for every (muni_code, year) in split_pairs
                    if not self.split_pairs:
                        if muni_code in self.yield_map:
                            try:
                                data = np.load(muni_npy_file, mmap_mode="r")
                                num_pixels = len(data)
                                if num_pixels > 0:
                                    municipality_pixel_counts[muni_code] = num_pixels
                                    municipality_years[muni_code] = None
                            except Exception:
                                missing_municipalities.append(muni_code)
                    else:
                        pairs_for_muni = [
                            (m, y) for m, y in self.split_pairs if m == muni_code
                        ]
                        if not pairs_for_muni:
                            continue
                        try:
                            data = np.load(muni_npy_file, mmap_mode="r")
                            num_pixels = len(data)
                            if num_pixels > 0:
                                for m, y in pairs_for_muni:
                                    municipality_pixel_counts[(m, y)] = num_pixels
                                    municipality_years[(m, y)] = y
                        except Exception:
                            missing_municipalities.append(muni_code)
            if self.split_pairs:
                found_munis = set(m for m, y in municipality_pixel_counts.keys())
                missing_municipalities = [
                    m for m in target_muni_codes if m not in found_munis
                ]
            else:
                missing_municipalities = [
                    m for m in target_muni_codes if m not in municipality_pixel_counts
                ]
        else:
            for muni_code in target_muni_codes:
                found = False
                if self.year is not None:
                    muni_npy_file = None
                    for year_dir in self._year_dir_candidates(self.year):
                        candidate = year_dir / muni_code / f"{muni_code}.npy"
                        if not candidate.exists():
                            candidate = year_dir / f"{muni_code}.npy"
                        if candidate.exists():
                            muni_npy_file = candidate
                            break
                    if muni_npy_file is not None and muni_npy_file.exists():
                        try:
                            data = np.load(muni_npy_file, mmap_mode="r")
                            num_pixels = len(data)
                            if num_pixels > 0:
                                municipality_pixel_counts[muni_code] = num_pixels
                                municipality_years[muni_code] = self.year
                                found = True
                        except Exception as e:
                            if len(missing_municipalities) < 3:
                                print(
                                    f"  ⚠️  Warning: Could not read {muni_npy_file}: {e}"
                                )
                            missing_municipalities.append(muni_code)
                else:
                    for year, year_dir in self._year_dir_tuples:
                        if (
                            self.split_pairs
                            and (muni_code, year) not in self.split_pairs
                        ):
                            continue
                        muni_npy_file = year_dir / muni_code / f"{muni_code}.npy"
                        if not muni_npy_file.exists():
                            muni_npy_file = year_dir / f"{muni_code}.npy"
                        if muni_npy_file.exists():
                            try:
                                data = np.load(muni_npy_file, mmap_mode="r")
                                num_pixels = len(data)
                                if num_pixels > 0:
                                    key = (muni_code, year)
                                    municipality_pixel_counts[key] = num_pixels
                                    municipality_years[key] = year
                                    found = True
                            except Exception as e:
                                if len(missing_municipalities) < 3:
                                    print(
                                        f"  ⚠️  Warning: Could not read {muni_npy_file}: {e}"
                                    )
                                continue
                if not found:
                    missing_municipalities.append(muni_code)

        if len(missing_municipalities) > 0:
            print(
                f"  ⚠️  Warning: {len(missing_municipalities)} municipalities without .npy files"
            )

        # Filter to only municipalities with pixels
        # When year=None, keys are (muni_code, year) tuples; otherwise just muni_code
        self.municipality_list = sorted(municipality_pixel_counts.keys())
        self.municipality_pixel_counts = municipality_pixel_counts
        self.municipality_years = municipality_years
        self.use_multi_year = bool(self.municipality_list) and isinstance(
            self.municipality_list[0], tuple
        )

        print(
            f"Found {len(self.municipality_list)} municipalities with pixels and yield data"
        )
        total_pixels = sum(self.municipality_pixel_counts.values())
        print(f"Total pixels: {total_pixels:,}")
        print(
            f"Feature layout: {self.feature_layout!r} → STNet input_dim={self.input_feature_dim} "
            f"(choices: {', '.join(feature_layout_choices())})"
        )
        self._validate_npy_channel_requirement()

        self.pixel_transform = PixelTransform(
            sequencelength,
            feature_layout=self.feature_layout,
            randomchoice=randomchoice,
            interp=interp,
            seed=seed,
        )

        # Cache for .npy files (OrderedDict for LRU: move to end on access)
        self.npy_cache = OrderedDict()
        self.npy_cache_max_size = int(npy_cache_size)
        self._temporal_keep_cache: OrderedDict = OrderedDict()
        self._temporal_keep_cache_max = 64

        if self._temporal_filter_enabled:
            print(
                f"  Temporal pixel filter: min_images>={self.min_images}, "
                f"min_months>={self.min_months} "
                f"(season_reference={self.season_reference_date.isoformat()})"
            )
            raw_total = sum(municipality_pixel_counts.values())
            filtered_counts = {}
            dropped_keys = []
            for key, _raw_n in municipality_pixel_counts.items():
                if isinstance(key, tuple):
                    code, year_key = key
                else:
                    code, year_key = key, municipality_years.get(key)
                data = self.mmap_municipality(code, year=year_key)
                if data is None:
                    dropped_keys.append(key)
                    continue
                kept = int(self._temporal_keep_mask(data).sum())
                if kept <= 0:
                    dropped_keys.append(key)
                    continue
                filtered_counts[key] = kept
            for key in dropped_keys:
                municipality_pixel_counts.pop(key, None)
                municipality_years.pop(key, None)
            municipality_pixel_counts.update(filtered_counts)
            self.municipality_list = sorted(municipality_pixel_counts.keys())
            self.municipality_pixel_counts = municipality_pixel_counts
            self.municipality_years = municipality_years
            self.use_multi_year = bool(self.municipality_list) and isinstance(
                self.municipality_list[0], tuple
            )
            kept_total = sum(municipality_pixel_counts.values())
            print(
                f"  Temporal filter kept {kept_total:,}/{raw_total:,} pixels "
                f"across {len(self.municipality_list)} municipality–years "
                f"(dropped {len(dropped_keys)} empty after filter)"
            )
            print(
                f"Found {len(self.municipality_list)} municipalities with pixels and yield data"
            )
            total_pixels = sum(self.municipality_pixel_counts.values())
            print(f"Total pixels: {total_pixels:,}")

    @property
    def is_single_season(self):
        """True when root is one season folder (e.g. 2022-2023) with muni subdirs; one .npy per muni."""
        return getattr(self, "_flat_season", False)

    def _year_dir_candidates(self, year):
        """Return list of candidate dirs to look for municipality .npy (year subdirs or root when flat)."""
        if self._flat_season:
            return [self.root]
        out = [self.root / str(year)]
        out.extend(self._year_dirs_by_end_year.get(year, []))
        return out

    def _resolve_npy_path(self, municipality_code, year):
        """Return Path to municipality .npy file or None if not found."""
        for year_dir in self._year_dir_candidates(year):
            candidate = year_dir / municipality_code / f"{municipality_code}.npy"
            if not candidate.exists():
                candidate = year_dir / f"{municipality_code}.npy"
            if candidate.exists():
                return candidate
        return None

    def _validate_npy_channel_requirement(self) -> None:
        """Ensure on-disk .npy files have enough channels for the chosen layout."""
        if self._extra_channels_slice is None:
            return
        _, hi = self._extra_channels_slice
        bad: list[tuple[str, int]] = []
        checked = 0
        for key in self.municipality_list:
            if checked >= 120:
                break
            if isinstance(key, tuple):
                code, year = key
            else:
                code = key
                year = self.municipality_years.get(code, self.year)
            p = self._resolve_npy_path(code, year)
            if p is None or not p.is_file():
                continue
            checked += 1
            try:
                data = np.load(p, mmap_mode="r")
                if data.ndim < 3:
                    continue
                c = int(data.shape[2])
                if c < hi:
                    bad.append((str(p), c))
            except OSError:
                continue
        if bad:
            examples = "; ".join(f"{path} (C={c})" for path, c in bad[:5])
            raise ValueError(
                f"Feature layout {self.feature_layout!r} requires each .npy to have at least {hi} channels "
                f"(extra fields at indices {self._extra_channels_slice}). "
                f"Found files with fewer channels, e.g.: {examples}. "
                f"Use --feature-layout spectral or rebuild .npy with the extra channels."
            )
        if checked == 0 and len(self.municipality_list) > 0:
            print(
                f"  ⚠️  Could not read any .npy to verify channel count for layout {self.feature_layout!r}."
            )

    def _transform_chunk(self, chunk_arr):
        """Vectorized transform (delegates to PixelTransform — same as training)."""
        return self.pixel_transform.transform_chunk(chunk_arr)

    def _temporal_keep_mask(self, municipality_data) -> np.ndarray:
        """Boolean keep-mask for temporal image/month thresholds."""
        if not self._temporal_filter_enabled:
            return np.ones(len(municipality_data), dtype=bool)
        # Compute in chunks so large mmap files do not fully materialize at once.
        n = len(municipality_data)
        keep = np.zeros(n, dtype=bool)
        step = 8192
        for start in range(0, n, step):
            end = min(start + step, n)
            chunk = np.asarray(municipality_data[start:end], dtype=np.float32)
            keep[start:end] = pixel_temporal_keep_mask(
                chunk,
                min_images=self.min_images,
                min_months=self.min_months,
                reference_date=self.season_reference_date,
            )
        return keep

    def filter_municipality_data(self, municipality_data, *, cache_key=None):
        """
        Return municipality pixels after optional min_images / min_months filter.

        When no temporal filter is configured, returns ``municipality_data`` unchanged
        (including mmap views). Otherwise returns a contiguous subset of kept rows.
        """
        if municipality_data is None or len(municipality_data) == 0:
            return municipality_data
        if not self._temporal_filter_enabled:
            return municipality_data

        idx = None
        if cache_key is not None:
            cached = self._temporal_keep_cache.get(cache_key)
            if cached is not None:
                self._temporal_keep_cache.move_to_end(cache_key)
                idx = cached
        if idx is None:
            keep = self._temporal_keep_mask(municipality_data)
            idx = np.flatnonzero(keep)
            if cache_key is not None:
                if len(self._temporal_keep_cache) >= self._temporal_keep_cache_max:
                    self._temporal_keep_cache.popitem(last=False)
                self._temporal_keep_cache[cache_key] = idx

        if idx.size == 0:
            return municipality_data[:0]
        if idx.size == len(municipality_data):
            return municipality_data
        return np.ascontiguousarray(municipality_data[idx])

    def _resolve_load_year(self, municipality_code, year=None):
        """Return the harvest year used to locate a municipality .npy file."""
        if year is None and self._flat_season:
            return self.year if self.year is not None else 2023
        if year is None:
            if self.year is not None:
                return self.year
            if self.use_multi_year:
                for y, _ in self._year_dir_tuples:
                    if self._resolve_npy_path(municipality_code, y) is not None:
                        return y
                return None
            if municipality_code in self.municipality_years:
                return self.municipality_years[municipality_code]
            for y, _ in self._year_dir_tuples:
                if self._resolve_npy_path(municipality_code, y) is not None:
                    return y
            return None
        return year

    def mmap_municipality(self, municipality_code, year=None):
        """Memory-map a municipality .npy array (LRU-cached). Returns None if missing."""
        year = self._resolve_load_year(municipality_code, year)
        if year is None:
            return None
        muni_npy_file = self._resolve_npy_path(municipality_code, year)
        if muni_npy_file is None:
            return None

        if self.npy_cache_max_size <= 0:
            municipality_data = np.load(muni_npy_file, mmap_mode="r")
        else:
            if muni_npy_file not in self.npy_cache:
                if len(self.npy_cache) >= self.npy_cache_max_size:
                    self.npy_cache.popitem(last=False)
                self.npy_cache[muni_npy_file] = np.load(muni_npy_file, mmap_mode="r")
            self.npy_cache.move_to_end(muni_npy_file)
            municipality_data = self.npy_cache[muni_npy_file]
        if len(municipality_data) == 0:
            return None
        return municipality_data

    def load_pixels_from_municipality(
        self, municipality_code, year=None, chunk_size=10000
    ):
        """Load pixels from municipality .npy file in chunks.

        Args:
            municipality_code: Municipality code
            year: Year to load (required if use_multi_year=True, optional otherwise)
            chunk_size: Number of pixels per chunk
        """
        municipality_data = self.mmap_municipality(municipality_code, year=year)
        if municipality_data is None:
            return
        year_resolved = self._resolve_load_year(municipality_code, year)
        cache_key = self._resolve_npy_path(municipality_code, year_resolved)
        municipality_data = self.filter_municipality_data(
            municipality_data, cache_key=cache_key
        )
        if municipality_data is None or len(municipality_data) == 0:
            return

        num_pixels = len(municipality_data)
        for start in range(0, num_pixels, chunk_size):
            end = min(start + chunk_size, num_pixels)
            yield self._transform_chunk(municipality_data[start:end])

    def iter_period_pixel_chunks_from_data(
        self,
        municipality_data,
        *,
        num_periods: int,
        chunk_size: int,
        reference_date: date,
        cache_key=None,
    ):
        """Yield batched STNet inputs after incomplete-series month filtering."""
        municipality_data = self.filter_municipality_data(
            municipality_data, cache_key=cache_key
        )
        if municipality_data is None or len(municipality_data) == 0:
            return

        season_lut = _season_month_lut(reference_date)
        num_pixels = len(municipality_data)
        for start in range(0, num_pixels, chunk_size):
            end = min(start + chunk_size, num_pixels)
            chunk_view = municipality_data[start:end]
            if chunk_view.dtype == np.float32:
                chunk_arr = np.ascontiguousarray(chunk_view)
            else:
                chunk_arr = np.asarray(chunk_view, dtype=np.float32)
            n_pixels = chunk_arr.shape[0]
            doy_i = np.clip(
                chunk_arr[:, :, DOY_CHANNEL].astype(np.int64, copy=False),
                0,
                season_lut.shape[0] - 1,
            )
            season_m = season_lut[doy_i]
            valid = (season_m >= 1) & (season_m <= num_periods)
            filtered: list[np.ndarray | None] = [None] * n_pixels
            for i in range(n_pixels):
                idx = np.flatnonzero(valid[i])
                if idx.size == 0:
                    continue
                sub = chunk_arr[i, idx, :]
                order = np.argsort(sub[:, DOY_CHANNEL], kind="stable")
                filtered[i] = sub[order]
            by_len: dict[int, list[int]] = defaultdict(list)
            for i, sub in enumerate(filtered):
                if sub is not None:
                    by_len[sub.shape[0]].append(i)
            for indices in by_len.values():
                stacked = np.asarray([filtered[i] for i in indices], dtype=np.float32)
                if stacked.shape[0] == 0:
                    continue
                yield self._transform_chunk(stacked)

    def load_pixels_from_municipality_with_periods(
        self, municipality_code, year, num_periods, chunk_size=400, reference_date=None
    ):
        """Load pixels using all daily rows in the first num_periods season months (1, then 1–2, … 1–6).
        For each calendar month in that window, every day/image for that pixel is included. Rows are
        sorted by DOY. If more than `sequencelength` days remain, the first `sequencelength` (earliest
        in season) are kept. Otherwise the sequence is padded to `sequencelength`.
        reference_date: date where DOY 1 falls (e.g. date(2022, 10, 1)). No random subsampling.
        Yields batched 4-tuples (x, mask, doy, weight) like load_pixels_from_municipality.
        """
        if reference_date is None:
            reference_date = date(2022, 10, 1)
        municipality_data = self.mmap_municipality(municipality_code, year=year)
        if municipality_data is None:
            return
        year_resolved = self._resolve_load_year(municipality_code, year)
        cache_key = self._resolve_npy_path(municipality_code, year_resolved)
        yield from self.iter_period_pixel_chunks_from_data(
            municipality_data,
            num_periods=num_periods,
            chunk_size=chunk_size,
            reference_date=reference_date,
            cache_key=cache_key,
        )

    def __len__(self):
        return len(self.municipality_list)

    def __getitem__(self, index):
        """Return municipality code and municipality-level yield target."""
        key = self.municipality_list[index]

        # Handle both single-year (key is muni_code) and multi-year (key is (muni_code, year)) modes
        if self.use_multi_year:
            municipality_code, year = key
        else:
            municipality_code = key
            year = self.municipality_years[municipality_code]

        num_pixels = self.municipality_pixel_counts[key]

        if self.use_multi_year:
            target = self.yield_map[(municipality_code, year)]
        else:
            target = self.yield_map[municipality_code]

        if self.num_outputs == 1:
            if target is None or target == "-" or target == "":
                target_vec = [0.0]
            else:
                try:
                    target_vec = [float(target)]
                except (ValueError, TypeError):
                    target_vec = [0.0]
        else:
            target_vec = []
            raw = target if isinstance(target, (list, tuple)) else [target]
            for v in raw:
                if v is None or v == "-" or v == "":
                    target_vec.append(0.0)
                else:
                    try:
                        target_vec.append(float(v))
                    except (ValueError, TypeError):
                        target_vec.append(0.0)

        target_t = torch.tensor(target_vec, dtype=torch.float32)
        if self.normalize_targets:
            mean = torch.as_tensor(self.target_mean, dtype=torch.float32)
            std = torch.as_tensor(self.target_std, dtype=torch.float32)
            target_t = (target_t - mean) / std

        if self.num_outputs == 1:
            target_t = target_t.squeeze(0)

        return (
            municipality_code,
            target_t,
            num_pixels,
            year,
        )
