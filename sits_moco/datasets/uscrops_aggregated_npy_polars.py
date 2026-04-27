"""
Dataset class for aggregated regression: groups pixels by municipality.
Polars-optimized version for faster index loading.

Each sample = pixel metadata for one municipality + municipality-level yield target.
Training code loads pixels in chunks on-demand.
"""

from collections import OrderedDict, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


def _doy_to_season_month(doy, reference_date):
    """DOY 1 = reference_date (e.g. 2022-10-01). Use date + timedelta to get calendar month, then map to season 1..6."""
    d = reference_date + timedelta(days=int(doy) - 1)
    cal = d.month
    ref_month = reference_date.month
    season_cal_months = [(ref_month - 1 + i) % 12 + 1 for i in range(6)]
    if cal not in season_cal_months:
        return 0
    return season_cal_months.index(cal) + 1


def _indices_first_n_months(row, num_periods, reference_date):
    """Return indices of rows in (T, 11) where season month is in 1..num_periods."""
    doys = row[:, -1].astype(np.float64)
    months = np.array([_doy_to_season_month(d, reference_date) for d in doys])
    valid = (months >= 1) & (months <= num_periods)
    return np.nonzero(valid)[0]


class USCropsAggregatedNPY(Dataset):
    """
    Dataset that groups pixels by municipality for yield prediction.
    Polars-optimized version for faster index loading.

    Returns pixel metadata (lazy loading) and municipality-level yield target.
    Training code loads pixels in chunks on-demand.
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
    ):
        super(USCropsAggregatedNPY, self).__init__()

        mode = mode.lower()
        assert mode in ["train", "valid", "eval", "test", "all"]

        self.root = Path(root)
        self.year = year
        self.mode = mode
        self.sequencelength = sequencelength
        self.rc = randomchoice
        self.interp = interp

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

        # Find columns (try common names)
        municipality_code_col = None
        for col in ["municipality_code", "code", "municipality", "muni_code"]:
            if col in yield_df.columns:
                municipality_code_col = col
                break
        if municipality_code_col is None:
            raise ValueError(
                f"Could not find municipality code column. Available: {list(yield_df.columns)}"
            )

        yield_col = None
        for col in ["production", "yield", "yield_tons", "tons"]:
            if col in yield_df.columns:
                yield_col = col
                break
        if yield_col is None:
            raise ValueError(
                f"Could not find yield column. Available: {list(yield_df.columns)}"
            )

        # Convert municipality_code to string and filter by split
        yield_df = yield_df.with_columns(
            pl.col(municipality_code_col).cast(pl.Utf8).alias(municipality_code_col)
        )

        # Filter by split (skip filter when mode is "all" to use train/valid/test entries)
        if "split" in yield_df.columns and mode != "all":
            split_mapping = {"train": "train", "valid": "valid", "eval": "eval", "test": "test"}
            target_split = split_mapping.get(mode, mode)
            yield_df = yield_df.filter(pl.col("split") == target_split)
            print(f"Filtered to {target_split} split: {len(yield_df)} rows")
        elif mode == "all" and "split" in yield_df.columns:
            print(f"Using all splits (train/valid/test): {len(yield_df)} rows")
        elif "mode" in yield_df.columns and mode != "all":
            yield_df = yield_df.filter(pl.col("mode") == mode)
            print(f"Filtered to {mode} mode: {len(yield_df)} rows")

        harvest_years_set = (
            set(int(y) for y in harvest_years) if harvest_years is not None else None
        )

        self.split_pairs = set()
        if "year" in yield_df.columns and self.year is None:
            trip = yield_df.select(
                [municipality_code_col, "year", yield_col]
            ).to_dict()
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
            print(f"Loaded yield targets for {len(self.yield_map)} municipality entries")

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
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit() and len(parts[0]) == 4 and len(parts[1]) == 4:
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
                        pairs_for_muni = [(m, y) for m, y in self.split_pairs if m == muni_code]
                        if not pairs_for_muni:
                            continue
                        try:
                            data = np.load(muni_npy_file, mmap_mode="r")
                            num_pixels = len(data)
                            if num_pixels > 0:
                                for (m, y) in pairs_for_muni:
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
                                print(f"  ⚠️  Warning: Could not read {muni_npy_file}: {e}")
                            missing_municipalities.append(muni_code)
                else:
                    for year, year_dir in self._year_dir_tuples:
                        if self.split_pairs and (muni_code, year) not in self.split_pairs:
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
                                    print(f"  ⚠️  Warning: Could not read {muni_npy_file}: {e}")
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

        # Transform parameters
        from .datautils import getWeight_batch

        self.getWeight_batch = getWeight_batch
        self.mean = np.array(
            [[0.147, 0.169, 0.186, 0.221, 0.273, 0.297, 0.308, 0.316, 0.256, 0.188]]
        )
        self.std = np.array(
            [0.227, 0.219, 0.222, 0.22, 0.2, 0.193, 0.192, 0.182, 0.123, 0.106]
        )

        # Cache for .npy files (OrderedDict for LRU: move to end on access)
        self.npy_cache = OrderedDict()
        self.npy_cache_max_size = 50

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

    def _transform_chunk(self, chunk_arr):
        """Vectorized transform for a chunk of pixels. chunk_arr: (N, T, 11). Returns list of N 4-tuples (x_pad, mask, doy_pad, weight_pad)."""
        N, T, _ = chunk_arr.shape
        doy = chunk_arr[:, :, -1].astype(np.int32)  # (N, T)
        x = chunk_arr[:, :, :10] * 1e-4  # (N, T, 10)

        weight = self.getWeight_batch(x)  # (N, T)
        x = (x - self.mean) / self.std

        seq_len = self.sequencelength
        if self.interp:
            doy_pad = np.linspace(0, 366, seq_len).astype(np.int32)
            x_pad = np.zeros((N, seq_len, 10), dtype=np.float32)
            for n in range(N):
                x_pad[n] = np.array(
                    [np.interp(doy_pad, doy[n], x[n, :, i]) for i in range(10)]
                ).T
            weight_pad = self.getWeight_batch(x_pad * self.std + self.mean)
            mask = np.ones((N, seq_len), dtype=np.int32)
            doy_pad_broadcast = np.tile(doy_pad.astype(np.int64), (N, 1))
        elif self.rc:
            replace = T >= seq_len
            idxs = np.random.choice(T, seq_len, replace=replace)
            idxs.sort()
            x_pad = x[:, idxs, :]
            doy_pad_broadcast = doy[:, idxs]
            weight_pad = weight[:, idxs]
            weight_pad /= weight_pad.sum(axis=1, keepdims=True)
            mask = np.ones((N, seq_len), dtype=np.int32)
        else:
            if T == seq_len:
                x_pad = x
                doy_pad_broadcast = doy
                weight_pad = weight
                weight_pad /= weight_pad.sum(axis=1, keepdims=True)
                mask = np.ones((N, seq_len), dtype=np.int32)
            elif T < seq_len:
                mask = np.zeros((N, seq_len), dtype=np.int32)
                mask[:, :T] = 1
                x_pad = np.zeros((N, seq_len, 10), dtype=np.float32)
                x_pad[:, :T, :] = x
                doy_pad_broadcast = np.zeros((N, seq_len), dtype=np.int32)
                doy_pad_broadcast[:, :T] = doy
                weight_pad = np.zeros((N, seq_len), dtype=np.float64)
                weight_pad[:, :T] = weight
                wsum = weight_pad.sum(axis=1, keepdims=True)
                weight_pad /= np.where(wsum > 0, wsum, 1.0)
            else:
                idxs = np.random.choice(T, seq_len, replace=False)
                idxs.sort()
                x_pad = x[:, idxs, :]
                doy_pad_broadcast = doy[:, idxs]
                weight_pad = weight[:, idxs]
                weight_pad /= weight_pad.sum(axis=1, keepdims=True)
                mask = np.ones((N, seq_len), dtype=np.int32)

        x_pad = np.ascontiguousarray(x_pad.astype(np.float32))
        mask_bool = mask == 0  # (N, seq_len) True = padding
        doy_pad_broadcast = np.ascontiguousarray(doy_pad_broadcast.astype(np.int64))
        weight_pad = np.ascontiguousarray(weight_pad.astype(np.float32))

        x_t = torch.from_numpy(x_pad)
        mask_t = torch.from_numpy(mask_bool)
        doy_t = torch.from_numpy(doy_pad_broadcast)
        weight_t = torch.from_numpy(weight_pad)

        # Return same interface as before: list of N 4-tuples so training code's torch.stack([p[0] for p in pixel_chunk]) works
        return [(x_t[i], mask_t[i], doy_t[i], weight_t[i]) for i in range(N)]

    def _transform_chunk_incomplete_eval(self, chunk_arr):
        """(N, T, 11) with T timesteps already in chronological (DOY) order for each pixel.
        Pad to sequencelength, or if T > sequencelength keep the earliest sequencelength days.
        Deterministic: no random subsample, no interp, no randomchoice. Matches incomplete-season eval.
        """
        N, T, _ = chunk_arr.shape
        doy = chunk_arr[:, :, -1].astype(np.int32)
        x = chunk_arr[:, :, :10] * 1e-4
        weight = self.getWeight_batch(x)
        x = (x - self.mean) / self.std
        seq_len = self.sequencelength
        if T > seq_len:
            x = x[:, :seq_len, :]
            doy = doy[:, :seq_len]
            weight = weight[:, :seq_len]
            T = seq_len
        if T == seq_len:
            x_pad = x
            doy_pad_broadcast = doy
            weight_pad = weight
            weight_pad = weight_pad / weight_pad.sum(axis=1, keepdims=True)
            mask = np.ones((N, seq_len), dtype=np.int32)
        else:
            mask = np.zeros((N, seq_len), dtype=np.int32)
            mask[:, :T] = 1
            x_pad = np.zeros((N, seq_len, 10), dtype=np.float32)
            x_pad[:, :T, :] = x
            doy_pad_broadcast = np.zeros((N, seq_len), dtype=np.int32)
            doy_pad_broadcast[:, :T] = doy
            weight_pad = np.zeros((N, seq_len), dtype=np.float64)
            weight_pad[:, :T] = weight
            wsum = weight_pad.sum(axis=1, keepdims=True)
            weight_pad /= np.where(wsum > 0, wsum, 1.0)
        x_pad = np.ascontiguousarray(x_pad.astype(np.float32))
        mask_bool = mask == 0
        doy_pad_broadcast = np.ascontiguousarray(doy_pad_broadcast.astype(np.int64))
        weight_pad = np.ascontiguousarray(weight_pad.astype(np.float32))
        x_t = torch.from_numpy(x_pad)
        mask_t = torch.from_numpy(mask_bool)
        doy_t = torch.from_numpy(doy_pad_broadcast)
        weight_t = torch.from_numpy(weight_pad)
        return [(x_t[i], mask_t[i], doy_t[i], weight_t[i]) for i in range(N)]

    def load_pixels_from_municipality(
        self, municipality_code, year=None, chunk_size=10000
    ):
        """Load pixels from municipality .npy file in chunks.

        Args:
            municipality_code: Municipality code
            year: Year to load (required if use_multi_year=True, optional otherwise)
            chunk_size: Number of pixels per chunk
        """
        # Determine which year to use
        if year is None:
            if self.year is not None:
                year = self.year
            elif self.use_multi_year:
                # Fallback: try each year (cached) until we find a file
                for y, _ in self._year_dir_tuples:
                    if self._resolve_npy_path(municipality_code, y) is not None:
                        year = y
                        break
                if year is None:
                    return  # No file found
            else:
                # Single-year mode: use stored year
                if municipality_code in self.municipality_years:
                    year = self.municipality_years[municipality_code]
                else:
                    for y, _ in self._year_dir_tuples:
                        if self._resolve_npy_path(municipality_code, y) is not None:
                            year = y
                            break
                    if year is None:
                        return  # No file found

        muni_npy_file = self._resolve_npy_path(municipality_code, year)
        if muni_npy_file is None:
            return  # No .npy found for this (municipality, year)

        if muni_npy_file not in self.npy_cache:
            if len(self.npy_cache) >= self.npy_cache_max_size:
                self.npy_cache.popitem(last=False)
            self.npy_cache[muni_npy_file] = np.load(muni_npy_file, mmap_mode="r")

        # LRU: move to end so this file is evicted last
        self.npy_cache.move_to_end(muni_npy_file)
        municipality_data = self.npy_cache[muni_npy_file]
        if len(municipality_data) == 0:
            return

        num_pixels = len(municipality_data)
        for start in range(0, num_pixels, chunk_size):
            end = min(start + chunk_size, num_pixels)
            chunk_arr = np.asarray(municipality_data[start:end], dtype=np.float32)
            yield self._transform_chunk(chunk_arr)

    def load_pixels_from_municipality_with_periods(
        self, municipality_code, year, num_periods, chunk_size=400, reference_date=None
    ):
        """Load pixels using all daily rows in the first num_periods season months (1, then 1–2, … 1–6).
        For each calendar month in that window, every day/image for that pixel is included. Rows are
        sorted by DOY. If more than `sequencelength` days remain, the first `sequencelength` (earliest
        in season) are kept. Otherwise the sequence is padded to `sequencelength`.
        reference_date: date where DOY 1 falls (e.g. date(2022, 10, 1)). No random subsampling.
        Yields lists of 4-tuples (x_pad, mask, doy_pad, weight_pad) like load_pixels_from_municipality.
        """
        if reference_date is None:
            reference_date = date(2022, 10, 1)
        if year is None and not self._flat_season:
            if self.year is not None:
                year = self.year
            elif self.use_multi_year:
                for y, _ in self._year_dir_tuples:
                    if self._resolve_npy_path(municipality_code, y) is not None:
                        year = y
                        break
                if year is None:
                    return
            else:
                if municipality_code in self.municipality_years:
                    year = self.municipality_years[municipality_code]
                else:
                    for y, _ in self._year_dir_tuples:
                        if self._resolve_npy_path(municipality_code, y) is not None:
                            year = y
                            break
                    if year is None:
                        return
        if year is None and self._flat_season:
            year = 2023  # dummy; path resolution ignores it in flat mode

        muni_npy_file = self._resolve_npy_path(municipality_code, year)
        if muni_npy_file is None:
            return

        if muni_npy_file not in self.npy_cache:
            if len(self.npy_cache) >= self.npy_cache_max_size:
                self.npy_cache.popitem(last=False)
            self.npy_cache[muni_npy_file] = np.load(muni_npy_file, mmap_mode="r")
        self.npy_cache.move_to_end(muni_npy_file)
        municipality_data = self.npy_cache[muni_npy_file]
        if len(municipality_data) == 0:
            return

        num_pixels = len(municipality_data)
        for start in range(0, num_pixels, chunk_size):
            end = min(start + chunk_size, num_pixels)
            chunk_arr = np.asarray(municipality_data[start:end], dtype=np.float32)
            N = chunk_arr.shape[0]
            # All images in first num_periods months (1, 1–2, 1–3, …) per pixel
            filtered = []
            for i in range(N):
                idx = _indices_first_n_months(chunk_arr[i], num_periods, reference_date)
                if len(idx) == 0:
                    filtered.append(None)
                    continue
                sub = chunk_arr[i, idx, :]
                doys = sub[:, -1].astype(np.float64)
                sub = sub[np.argsort(doys, kind="stable"), :]
                filtered.append(sub)
            by_len = defaultdict(list)
            for i in range(N):
                if filtered[i] is not None:
                    by_len[filtered[i].shape[0]].append(i)
            result = [None] * N
            for k, indices in by_len.items():
                stacked = np.asarray([filtered[i] for i in indices], dtype=np.float32)
                tuples_list = self._transform_chunk_incomplete_eval(stacked)
                for j, i in enumerate(indices):
                    result[i] = tuples_list[j]
            yield [r for r in result if r is not None]

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
