"""
Dataset class for aggregated regression: groups pixels by municipality.
Each sample = pixel metadata for one municipality + municipality-level yield target.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class USCropsAggregatedNPY(Dataset):
    """
    Dataset that groups pixels by municipality for yield prediction.

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
        seed=111,
        preload_ram=False,
    ):
        super(USCropsAggregatedNPY, self).__init__()

        mode = mode.lower()
        assert mode in ["train", "valid", "eval"]

        self.root = Path(root)
        self.mode = mode
        self.sequencelength = sequencelength
        self.rc = randomchoice
        self.interp = interp

        # Load yield targets
        print(f"Loading yield targets from {yield_csv}...")
        yield_df = pd.read_csv(yield_csv)

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

        yield_df[municipality_code_col] = yield_df[municipality_code_col].astype(str)

        # Filter by split
        if "split" in yield_df.columns:
            split_mapping = {"train": "train", "valid": "valid", "eval": "eval"}
            target_split = split_mapping.get(mode, mode)
            yield_df = yield_df[yield_df["split"] == target_split]
            print(f"Filtered to {target_split} split: {len(yield_df)} municipalities")
        elif "mode" in yield_df.columns:
            yield_df = yield_df[yield_df["mode"] == mode]
            print(f"Filtered to {mode} mode: {len(yield_df)} municipalities")

        self.yield_map = dict(zip(yield_df[municipality_code_col], yield_df[yield_col]))
        print(f"Loaded yield targets for {len(self.yield_map)} municipalities")

        # No index needed - load pixel counts directly from .npy files
        print(f"Loading pixel counts from .npy files...")
        municipality_pixel_counts = {}  # {muni_code: num_pixels}

        target_municipalities = set(self.yield_map.keys())

        # Check which municipalities have .npy files and get pixel counts
        for muni_code in target_municipalities:
            muni_npy_file = self.root / muni_code / f"{muni_code}.npy"
            if muni_npy_file.exists():
                try:
                    # Load with memory mapping to get shape without loading full file
                    data = np.load(muni_npy_file, mmap_mode="r")
                    num_pixels = len(data)
                    if num_pixels > 0:
                        municipality_pixel_counts[muni_code] = num_pixels
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not read {muni_npy_file}: {e}")
                    continue

        # Filter to only municipalities with pixels
        self.municipality_list = sorted(municipality_pixel_counts.keys())

        # Limit number of municipalities if specified
        if max_municipalities is not None and max_municipalities > 0:
            if len(self.municipality_list) > max_municipalities:
                # Use random seed for reproducible sampling
                import random

                random.seed(seed)
                self.municipality_list = random.sample(
                    self.municipality_list, max_municipalities
                )
                print(
                    f"Limited to {max_municipalities} municipalities (randomly sampled from {len(municipality_pixel_counts)} total)"
                )

        # Update pixel counts to only include selected municipalities
        self.municipality_pixel_counts = {
            muni: municipality_pixel_counts[muni] for muni in self.municipality_list
        }

        print(
            f"Found {len(self.municipality_list)} municipalities with pixels and yield data"
        )
        total_pixels = sum(municipality_pixel_counts.values())
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

    def load_pixels_from_municipality(self, municipality_code, chunk_size=10000):
        """Load pixels from municipality .npy file in chunks."""
        muni_npy_file = self.root / municipality_code / f"{municipality_code}.npy"

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
        municipality_code = self.municipality_list[index]
        num_pixels = self.municipality_pixel_counts[municipality_code]

        target = self.yield_map[municipality_code]
        if pd.isna(target) or target == "-" or target == "":
            target = 0.0
        else:
            try:
                target = float(target)
            except (ValueError, TypeError):
                target = 0.0

        # Return municipality code (no metadata needed - we load directly from .npy)
        return municipality_code, torch.tensor(target, dtype=torch.float32), num_pixels

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
            weight_sum = weight_pad.sum()
            if weight_sum > 1e-8:
                weight_pad /= weight_sum
            else:
                # If all weights are zero, use uniform weights
                weight_pad = np.ones_like(weight_pad) / len(weight_pad)
        else:
            x_length, c_length = x.shape
            if x_length == self.sequencelength:
                mask = np.ones((self.sequencelength,), dtype=int)
                x_pad = x
                doy_pad = doy
                weight_pad = weight
                weight_sum = weight_pad.sum()
                if weight_sum > 1e-8:
                    weight_pad /= weight_sum
                else:
                    # If all weights are zero, use uniform weights
                    weight_pad = np.ones_like(weight_pad) / len(weight_pad)
            elif x_length < self.sequencelength:
                mask = np.zeros((self.sequencelength,), dtype=int)
                mask[:x_length] = 1
                x_pad = np.zeros((self.sequencelength, c_length))
                x_pad[:x_length, :] = x[:x_length, :]
                doy_pad = np.zeros((self.sequencelength,), dtype=int)
                doy_pad[:x_length] = doy[:x_length]
                weight_pad = np.zeros((self.sequencelength,), dtype=float)
                weight_pad[:x_length] = weight[:x_length]
                weight_sum = weight_pad.sum()
                if weight_sum > 1e-8:  # Only normalize if sum is significant
                    weight_pad /= weight_sum
                else:
                    # If all weights are zero, use uniform weights
                    weight_pad = np.ones_like(weight_pad) / len(weight_pad)
                    weight_pad[:x_length] = weight[:x_length]
                    weight_sum = weight_pad[:x_length].sum()
                    if weight_sum > 1e-8:
                        weight_pad[:x_length] /= weight_sum
                    else:
                        weight_pad[:x_length] = 1.0 / x_length if x_length > 0 else 1.0 / len(weight_pad)
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
