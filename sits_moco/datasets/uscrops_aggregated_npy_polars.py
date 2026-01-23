"""
Dataset class for aggregated regression: groups pixels by municipality.
Polars-optimized version for faster index loading.

Each sample = pixel metadata for one municipality + municipality-level yield target.
Training code loads pixels in chunks on-demand.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


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
    ):
        super(USCropsAggregatedNPY, self).__init__()

        mode = mode.lower()
        assert mode in ["train", "valid", "eval"]

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

        # Filter by split
        if "split" in yield_df.columns:
            split_mapping = {"train": "train", "valid": "valid", "eval": "eval"}
            target_split = split_mapping.get(mode, mode)
            yield_df = yield_df.filter(pl.col("split") == target_split)
            print(f"Filtered to {target_split} split: {len(yield_df)} rows")
        elif "mode" in yield_df.columns:
            yield_df = yield_df.filter(pl.col("mode") == mode)
            print(f"Filtered to {mode} mode: {len(yield_df)} rows")

        # Build yield map and track (municipality, year) pairs for multi-year mode
        yield_dict = yield_df.select([municipality_code_col, yield_col]).to_dict()
        self.yield_map = dict(
            zip(yield_dict[municipality_code_col], yield_dict[yield_col])
        )
        
        # For multi-year mode, track which (municipality, year) pairs are in this split
        self.split_pairs = set()  # Set of (municipality_code, year) tuples
        if "year" in yield_df.columns and self.year is None:
            # Multi-year mode: track which (municipality, year) pairs belong to this split
            year_dict = yield_df.select([municipality_code_col, "year"]).to_dict()
            for muni_code, year_val in zip(year_dict[municipality_code_col], year_dict["year"]):
                if year_val is not None:
                    try:
                        year_int = int(year_val)
                        self.split_pairs.add((str(muni_code), year_int))
                    except (ValueError, TypeError):
                        continue
        
        print(f"Loaded yield targets for {len(self.yield_map)} municipality entries")

        # No index needed - load pixel counts directly from .npy files
        print(f"Loading pixel counts from .npy files...")
        municipality_pixel_counts = (
            {}
        )  # {(muni_code, year): num_pixels} or {muni_code: num_pixels}
        municipality_years = {}  # {(muni_code, year): year} or {muni_code: year}

        target_municipalities = set(self.yield_map.keys())

        # Check which municipalities have .npy files and get pixel counts
        # Files are organized as root/year/municipality_code/municipality_code.npy
        missing_municipalities = []
        for muni_code in target_municipalities:
            found = False
            if self.year is not None:
                # Check specific year
                muni_npy_file = self.root / str(self.year) / f"{muni_code}.npy"
                if not muni_npy_file.exists():
                    # Try subdirectory path: root/year/municipality_code/municipality_code.npy
                    muni_npy_file = (
                        self.root / str(self.year) / muni_code / f"{muni_code}.npy"
                    )
                if muni_npy_file.exists():
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
                # Check ALL year directories - create separate samples for each year
                # BUT only for (municipality, year) pairs that are in the current split
                for year_dir in sorted(self.root.iterdir()):
                    if year_dir.is_dir() and year_dir.name.isdigit():
                        year = int(year_dir.name)
                        
                        # If we have split_pairs, only include this year if (muni_code, year) is in the split
                        if self.split_pairs and (muni_code, year) not in self.split_pairs:
                            continue
                        
                        # Try direct path first: root/year/municipality_code.npy
                        muni_npy_file = year_dir / f"{muni_code}.npy"
                        if not muni_npy_file.exists():
                            # Try subdirectory path: root/year/municipality_code/municipality_code.npy
                            muni_npy_file = year_dir / muni_code / f"{muni_code}.npy"
                        if muni_npy_file.exists():
                            try:
                                data = np.load(muni_npy_file, mmap_mode="r")
                                num_pixels = len(data)
                                if num_pixels > 0:
                                    # Store as (muni_code, year) tuple for multi-year support
                                    key = (muni_code, year)
                                    municipality_pixel_counts[key] = num_pixels
                                    municipality_years[key] = year
                                    found = True
                                    # Don't break - continue to find all years in this split
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
        self.use_multi_year = self.year is None  # Flag to indicate multi-year mode

        print(
            f"Found {len(self.municipality_list)} municipalities with pixels and yield data"
        )
        total_pixels = sum(self.municipality_pixel_counts.values())
        print(f"Total pixels: {total_pixels:,}")

        # Transform parameters
        from .datautils import getWeight

        self.mean = np.array(
            [[0.147, 0.169, 0.186, 0.221, 0.273, 0.297, 0.308, 0.316, 0.256, 0.188]]
        )
        self.std = np.array(
            [0.227, 0.219, 0.222, 0.22, 0.2, 0.193, 0.192, 0.182, 0.123, 0.106]
        )
        self.getWeight = getWeight

        # Cache for .npy files
        self.npy_cache = {}
        self.npy_cache_max_size = 50

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
                # In multi-year mode, year should be provided
                # Fallback: try to find any year
                for year_dir in sorted(self.root.iterdir()):
                    if year_dir.is_dir() and year_dir.name.isdigit():
                        candidate = (
                            year_dir / municipality_code / f"{municipality_code}.npy"
                        )
                        if candidate.exists():
                            year = int(year_dir.name)
                            break
                if year is None:
                    return  # No file found
            else:
                # Single-year mode: use stored year
                if municipality_code in self.municipality_years:
                    year = self.municipality_years[municipality_code]
                else:
                    # Fallback: Find first available year directory with this municipality
                    for year_dir in sorted(self.root.iterdir()):
                        if year_dir.is_dir() and year_dir.name.isdigit():
                            candidate = (
                                year_dir
                                / municipality_code
                                / f"{municipality_code}.npy"
                            )
                            if candidate.exists():
                                year = int(year_dir.name)
                                break
                    if year is None:
                        return  # No file found

        # Files are organized as root/year/municipality_code/municipality_code.npy
        muni_npy_file = self.root / str(year) / f"{municipality_code}.npy"
        if not muni_npy_file.exists():
            # Try subdirectory path
            muni_npy_file = (
                self.root / str(year) / municipality_code / f"{municipality_code}.npy"
            )

        if muni_npy_file not in self.npy_cache:
            if muni_npy_file.exists():
                if len(self.npy_cache) >= self.npy_cache_max_size:
                    oldest_key = next(iter(self.npy_cache))
                    del self.npy_cache[oldest_key]
                self.npy_cache[muni_npy_file] = np.load(muni_npy_file, mmap_mode="r")
            else:
                return  # No file, yield nothing

        municipality_data = self.npy_cache[muni_npy_file]
        if len(municipality_data) == 0:
            return

        current_chunk = []
        for pixel_index in range(len(municipality_data)):
            X = municipality_data[pixel_index].copy()
            X_tuple = self._transform_pixel(X)
            current_chunk.append(X_tuple)

            if len(current_chunk) >= chunk_size:
                yield current_chunk
                current_chunk = []

        if current_chunk:
            yield current_chunk

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

        target = self.yield_map[municipality_code]
        # Handle null/NaN values (Polars uses None for null)
        if target is None or target == "-" or target == "":
            target = 0.0
        else:
            try:
                target = float(target)
            except (ValueError, TypeError):
                target = 0.0

        # Normalize target if normalization is enabled
        if self.normalize_targets:
            target = (target - self.target_mean) / self.target_std

        # Return municipality code, target, num_pixels, and year (year needed for loading correct .npy file)
        return (
            municipality_code,
            torch.tensor(target, dtype=torch.float32),
            num_pixels,
            year,
        )

    def _transform_pixel(self, x):
        """Transform pixel data: normalize, pad/sample to sequencelength, extract DOY."""
        # Extract DOY (stored as float32, convert to int for indexing)
        doy = x[:, -1].astype(np.int32)
        x = x[:, :10] * 1e-4

        weight = self.getWeight(x)
        x = (x - self.mean) / self.std

        if self.interp:
            doy_pad = np.linspace(0, 366, self.sequencelength).astype("int")
            # np.interp only works for 1D arrays, so we iterate over bands
            # This is already efficient - list comprehension with vectorized np.interp
            x_pad = np.array([np.interp(doy_pad, doy, x[:, i]) for i in range(10)]).T
            weight_pad = self.getWeight(x_pad * self.std + self.mean)
            mask = np.ones((self.sequencelength,), dtype=int)
        elif self.rc:
            replace = False if x.shape[0] >= self.sequencelength else True
            idxs = np.random.choice(x.shape[0], self.sequencelength, replace=replace)
            idxs.sort()
            x_pad = x[idxs]
            mask = np.ones((self.sequencelength,), dtype=int)
            doy_pad = doy[idxs]
            weight_pad = weight[idxs]
            weight_pad /= weight_pad.sum()
        else:
            x_length, c_length = x.shape
            if x_length == self.sequencelength:
                mask = np.ones((self.sequencelength,), dtype=int)
                x_pad = x
                doy_pad = doy
                weight_pad = weight
                weight_pad /= weight_pad.sum()
            elif x_length < self.sequencelength:
                mask = np.zeros((self.sequencelength,), dtype=int)
                mask[:x_length] = 1
                x_pad = np.zeros((self.sequencelength, c_length))
                x_pad[:x_length, :] = x[:x_length, :]
                doy_pad = np.zeros((self.sequencelength,), dtype=int)
                doy_pad[:x_length] = doy[:x_length]
                weight_pad = np.zeros((self.sequencelength,), dtype=float)
                weight_pad[:x_length] = weight[:x_length]
                weight_pad /= weight_pad.sum() if weight_pad.sum() > 0 else 1
            else:
                idxs = np.random.choice(x.shape[0], self.sequencelength, replace=False)
                idxs.sort()
                x_pad = x[idxs]
                mask = np.ones((self.sequencelength,), dtype=int)
                doy_pad = doy[idxs]
                weight_pad = weight[idxs]
                weight_pad /= weight_pad.sum()

        return (
            torch.from_numpy(x_pad).type(torch.FloatTensor),
            torch.from_numpy(mask == 0),
            torch.from_numpy(doy_pad).type(torch.LongTensor),
            torch.from_numpy(weight_pad).type(torch.FloatTensor),
        )
