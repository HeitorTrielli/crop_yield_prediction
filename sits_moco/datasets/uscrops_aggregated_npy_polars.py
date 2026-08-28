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


_INVALID_DOY_SORT = np.int64(10_000)


def sort_in_season_chunk(
    chunk_arr: np.ndarray,
    *,
    season_lut: np.ndarray,
    sequencelength: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Copy a pixel chunk, keep season months 1..6, sort by DOY, truncate to ``sequencelength``.

    Returns ``(packed, season_month)`` with shapes ``[N, T, C]`` and ``[N, T]``.
    ``season_month`` is 0 for padded / out-of-window slots. Pixels with no in-season
    days are dropped. One sort can then be reused for every k by masking ``season <= k``.
    """
    if chunk_arr.dtype != np.float32 or not chunk_arr.flags.c_contiguous:
        chunk_arr = np.ascontiguousarray(chunk_arr, dtype=np.float32)
    _n, t, _c = chunk_arr.shape
    doy = np.clip(
        chunk_arr[:, :, DOY_CHANNEL].astype(np.int64, copy=False),
        0,
        season_lut.shape[0] - 1,
    )
    season = season_lut[doy]
    in_season = season >= 1
    n_in = in_season.sum(axis=1)
    keep = n_in > 0
    if not np.any(keep):
        return None
    if not bool(keep.all()):
        chunk_arr = chunk_arr[keep]
        season = season[keep]
        doy = doy[keep]
        in_season = in_season[keep]
        n_in = n_in[keep]

    t_keep = min(int(sequencelength), int(t))
    sort_key = np.where(in_season, doy, _INVALID_DOY_SORT)
    order = np.argsort(sort_key, axis=1, kind="mergesort")
    packed = np.take_along_axis(chunk_arr, order[:, :, None], axis=1)[:, :t_keep]
    season_p = np.take_along_axis(season, order, axis=1)[:, :t_keep]
    prefix = np.arange(t_keep, dtype=np.int32)[None, :] < np.minimum(n_in, t_keep)[
        :, None
    ]
    season_p = np.where(prefix, season_p, np.int8(0))
    return packed, season_p


def pack_incomplete_series_chunk(
    chunk_arr: np.ndarray,
    *,
    season_lut: np.ndarray,
    num_periods: int,
    sequencelength: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Vectorized incomplete-series pack: keep season months 1..k, earliest DOY first.

    Returns ``(packed, timestep_valid)`` with shapes ``[N, T, C]`` and ``[N, T]``,
    ``T = min(sequencelength, original T)``. Pixels with no in-window days are dropped.
    """
    if chunk_arr.dtype != np.float32 or not chunk_arr.flags.c_contiguous:
        chunk_arr = np.ascontiguousarray(chunk_arr, dtype=np.float32)
    n, t, _c = chunk_arr.shape
    doy = np.clip(
        chunk_arr[:, :, DOY_CHANNEL].astype(np.int64, copy=False),
        0,
        season_lut.shape[0] - 1,
    )
    valid = (season_lut[doy] >= 1) & (season_lut[doy] <= int(num_periods))
    n_valid = valid.sum(axis=1)
    keep = n_valid > 0
    if not np.any(keep):
        return None
    if not bool(keep.all()):
        chunk_arr = chunk_arr[keep]
        valid = valid[keep]
        doy = doy[keep]
        n_valid = n_valid[keep]
        n = int(chunk_arr.shape[0])

    t_keep = min(int(sequencelength), int(t))
    sort_key = np.where(valid, doy, _INVALID_DOY_SORT)
    order = np.argsort(sort_key, axis=1, kind="mergesort")
    packed = np.take_along_axis(chunk_arr, order[:, :, None], axis=1)[:, :t_keep]
    timestep_valid = np.arange(t_keep, dtype=np.int32)[None, :] < np.minimum(
        n_valid, t_keep
    )[:, None]
    return packed, timestep_valid


# Default: no coverage filter (use all municipality–year rows).
# Optional suggested band for S2 footprint vs IBGE planted area:
#   --min-coverage-ratio 0.8 --max-coverage-ratio 1.2
DEFAULT_MIN_COVERAGE_RATIO = None
DEFAULT_MAX_COVERAGE_RATIO = None
SUGGESTED_MIN_COVERAGE_RATIO = 0.8
SUGGESTED_MAX_COVERAGE_RATIO = 1.2
COVERAGE_RATIO_COL = "coverage_ratio"


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
        target_column: str = "production_t",
        npy_cache_size: int = 20,
        npy_defer_copy: bool = True,  # ignored; always deferred (kept for API compat)
        min_coverage_ratio: float | None = DEFAULT_MIN_COVERAGE_RATIO,
        max_coverage_ratio: float | None = DEFAULT_MAX_COVERAGE_RATIO,
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
        # Backward-compatible flag for logging / meta
        self.use_xavier = self._extra_channels_slice == (11, 13)

        self.root = Path(root)
        self.year = year
        self.mode = mode
        self.sequencelength = sequencelength
        self.rc = randomchoice
        self.interp = interp
        self.min_coverage_ratio = min_coverage_ratio
        self.max_coverage_ratio = max_coverage_ratio

        # Store normalization parameters
        self.target_mean = target_mean if target_mean is not None else 0.0
        self.target_std = target_std if target_std is not None else 1.0
        self.normalize_targets = target_mean is not None and target_std is not None

        # Load yield targets with Polars
        print(f"Loading yield targets from {yield_csv}...")
        # Handle null values: treat '-' and empty strings as null (same as pandas version)
        yield_df = pl.read_csv(
            yield_csv,
            null_values=["-", "", "nan", "NaN", "null", "NULL"],
            infer_schema_length=10000,  # Increase to better infer schema with null values
        )

        municipality_code_col = "municipality_code"
        yield_col = target_column
        cols = list(yield_df.columns)
        if municipality_code_col not in cols:
            raise ValueError(
                f"Yield CSV missing {municipality_code_col!r}. Available: {cols}"
            )
        if yield_col not in cols:
            raise ValueError(f"Yield CSV missing {yield_col!r}. Available: {cols}")
        self.municipality_code_column = municipality_code_col
        self.target_column = yield_col

        # Convert municipality_code to string and filter by split
        yield_df = yield_df.with_columns(
            pl.col(municipality_code_col).cast(pl.Utf8).alias(municipality_code_col)
        )

        yield_df = filter_yield_df_by_coverage(
            yield_df,
            min_ratio=min_coverage_ratio,
            max_ratio=max_coverage_ratio,
        )

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
            trip = yield_df.select([municipality_code_col, "year", yield_col]).to_dict()
            self.yield_map = {}
            for muni_code, year_val, yld in zip(
                trip[municipality_code_col],
                trip["year"],
                trip[yield_col],
            ):
                if year_val is None:
                    continue
                try:
                    yi = int(year_val)
                    self.split_pairs.add((str(muni_code), yi))
                    self.yield_map[(str(muni_code), yi)] = yld
                except (ValueError, TypeError):
                    continue
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
            )
        else:
            yield_dict = yield_df.select([municipality_code_col, yield_col]).to_dict()
            self.yield_map = dict(
                zip(yield_dict[municipality_code_col], yield_dict[yield_col])
            )
            if harvest_years_set is not None:
                raise ValueError(
                    "harvest_years filter requires year=None and a 'year' column in the yield CSV"
                )
            print(
                f"Loaded yield targets for {len(self.yield_map)} municipality entries"
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

    def _transform_chunk(self, chunk_arr, timestep_valid=None):
        """Vectorized transform (delegates to PixelTransform — same as training)."""
        return self.pixel_transform.transform_chunk(
            chunk_arr, timestep_valid=timestep_valid
        )

    def pack_sorted_season_chunk(self, chunk_arr, reference_date: date):
        """Sort in-season days once so every k can reuse the same packed chunk."""
        packed = sort_in_season_chunk(
            chunk_arr,
            season_lut=_season_month_lut(reference_date),
            sequencelength=int(self.sequencelength),
        )
        if packed is None:
            return None
        chunk_arr, season_p = packed
        valid = season_p >= 1
        seq_len = int(self.sequencelength)
        n, t_keep = season_p.shape
        if t_keep < seq_len:
            season_pad = np.zeros((n, seq_len), dtype=np.int16)
            season_pad[:, :t_keep] = season_p
        else:
            season_pad = np.ascontiguousarray(season_p, dtype=np.int16)
        return self._transform_chunk(chunk_arr, timestep_valid=valid), season_pad

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
    ):
        """Yield batched STNet inputs after incomplete-series month filtering."""
        if len(municipality_data) == 0:
            return

        season_lut = _season_month_lut(reference_date)
        seq_len = int(self.sequencelength)
        num_pixels = len(municipality_data)
        for start in range(0, num_pixels, chunk_size):
            end = min(start + chunk_size, num_pixels)
            chunk_view = municipality_data[start:end]
            packed = pack_incomplete_series_chunk(
                chunk_view,
                season_lut=season_lut,
                num_periods=num_periods,
                sequencelength=seq_len,
            )
            if packed is None:
                continue
            chunk_arr, timestep_valid = packed
            yield self._transform_chunk(chunk_arr, timestep_valid=timestep_valid)

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
        yield from self.iter_period_pixel_chunks_from_data(
            municipality_data,
            num_periods=num_periods,
            chunk_size=chunk_size,
            reference_date=reference_date,
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
        # Handle null/NaN values (Polars uses None for null)
        if target is None or target == "-" or target == "":
            target = 0.0
        else:
            try:
                target = float(target)
            except (ValueError, TypeError):
                target = 0.0

        if self.normalize_targets:
            target = (target - self.target_mean) / self.target_std

        return (
            municipality_code,
            torch.tensor(target, dtype=torch.float32),
            num_pixels,
            year,
        )
