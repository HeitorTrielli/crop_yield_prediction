"""
MoCo pretraining dataset over Paraná municipal .npy files.

Layout expected under ``root``::

    {year_range}/{muni_code}/{muni_code}.npy   # [N, T, C], C>=11 (10 spectral + DOY + …)

Unlike the US-toy MoCoDataset (tens of thousands of series), Paraná has hundreds of
millions of pixels. This class builds a stratified subsample and caches it as an
object .npy (same in-RAM format as MoCoDataset) under ``root/.moco_cache/``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import moco.loader
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets.feature_layout import (
    feature_layout_extra_slice,
    feature_layout_input_dim,
    normalize_feature_layout,
)
from datasets.pixel_transform import (
    DOY_CHANNEL,
    NO_DATA_VALUE,
    NUM_SPECTRAL_CHANNELS,
    PixelTransform,
)

_YEAR_RANGE_RE = re.compile(r"^(\d{4})-(\d{4})$")


def is_parana_npy_layout(datapath: Path | str) -> bool:
    """True if datapath looks like season folders ``YYYY-YYYY`` with muni .npy files."""
    root = Path(datapath)
    if not root.is_dir():
        return False
    for p in root.iterdir():
        if p.is_dir() and _YEAR_RANGE_RE.match(p.name):
            return True
    return False


def harvest_years_to_year_ranges(harvest_years: list[int] | tuple[int, ...]) -> list[str]:
    """Harvest year Y → folder ``(Y-1)-Y`` (soy season Oct Y-1 … Mar Y)."""
    return [f"{int(y) - 1}-{int(y)}" for y in sorted({int(y) for y in harvest_years})]


def list_muni_npy_files(root: Path, year_ranges: list[str]) -> list[Path]:
    """Return municipal .npy paths (skip ``*.keep_mask.npy``)."""
    files: list[Path] = []
    for yr in year_ranges:
        season_dir = root / yr
        if not season_dir.is_dir():
            continue
        for muni_dir in sorted(season_dir.iterdir()):
            if not muni_dir.is_dir():
                continue
            npy = muni_dir / f"{muni_dir.name}.npy"
            if npy.is_file():
                files.append(npy)
    return files


def _min_channels_for_layout(feature_layout: str) -> int:
    """Minimum .npy channel count to keep DOY (+ extras when required)."""
    sl = feature_layout_extra_slice(feature_layout)
    if sl is None:
        return DOY_CHANNEL + 1
    return max(DOY_CHANNEL + 1, int(sl[1]))


def _clean_pixel_row(
    row: np.ndarray, *, min_channels: int = DOY_CHANNEL + 1
) -> np.ndarray | None:
    """
    Drop timesteps with invalid spectral/DOY; keep full channel layout (incl. extras).

    Returns float32 array (T', C) with C >= min_channels, or None if nothing usable.
    """
    row = np.asarray(row, dtype=np.float32)
    if row.ndim != 2 or row.shape[1] < min_channels:
        return None
    spec = row[:, :NUM_SPECTRAL_CHANNELS]
    doy = row[:, DOY_CHANNEL]
    invalid = (
        ~np.isfinite(spec).all(axis=1)
        | (spec == NO_DATA_VALUE).any(axis=1)
        | ~np.isfinite(doy)
    )
    keep = ~invalid
    if not np.any(keep):
        return None
    return np.ascontiguousarray(row[keep], dtype=np.float32)


class ParanaMoCoDataset(Dataset):
    """
    Stratified subsample of Paraná pixels for MoCo, compatible with MoCoDataset.__getitem__.
    """

    def __init__(
        self,
        root: Path | str,
        year_ranges: list[str],
        sequencelength: int = 70,
        dataaug=None,
        max_samples: int = 500_000,
        seed: int = 111,
        rebuild_cache: bool = False,
        feature_layout: str = "spectral",
    ):
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        self.year_ranges = list(year_ranges)
        self.sequencelength = int(sequencelength)
        self.max_samples = int(max_samples)
        self.seed = int(seed)
        self.feature_layout = normalize_feature_layout(feature_layout)
        self.input_dim = feature_layout_input_dim(self.feature_layout)
        self._min_channels = _min_channels_for_layout(self.feature_layout)
        self._pixel_tx = PixelTransform(
            sequencelength=sequencelength,
            feature_layout=self.feature_layout,
            randomchoice=False,
            interp=False,
            seed=seed,
        )

        if dataaug is not None:
            self.dataaug = moco.loader.TwoCropsTransform(dataaug)
        else:
            self.dataaug = None

        years_tag = "_".join(self.year_ranges).replace("-", "")
        cache_dir = self.root / ".moco_cache"
        self.cache = (
            cache_dir
            / (
                f"Unsupervised_Parana_{years_tag}_{self.feature_layout}"
                f"_N{self.max_samples}_S{self.seed}.npy"
            )
        )

        print(
            f"Load Paraná unsupervised MoCo set "
            f"(years={self.year_ranges}, layout={self.feature_layout}, "
            f"max_samples={self.max_samples}, seed={self.seed})"
        )
        if self.cache.exists() and not rebuild_cache:
            self.load_cached_dataset()
        else:
            self.cache_dataset()

    def transform(self, x: np.ndarray):
        """Normalize features with the same layout rules as yield training."""
        chunk = np.ascontiguousarray(x[None, ...], dtype=np.float32)
        feats, _weight, doy = self._pixel_tx.features_from_chunk(chunk)
        return feats[0].astype(np.float32), doy[0].astype(np.float32)

    def load_cached_dataset(self) -> None:
        print(f"precached dataset files found at {self.cache}")
        self.X_list = np.load(self.cache, allow_pickle=True).tolist()
        print(f"  loaded {len(self.X_list)} time series")

    def cache_dataset(self) -> None:
        files = list_muni_npy_files(self.root, self.year_ranges)
        if not files:
            raise FileNotFoundError(
                f"No municipal .npy under {self.root} for year ranges {self.year_ranges}"
            )

        rng = np.random.default_rng(self.seed)
        # Count pixels per file (mmap header only)
        counts: list[int] = []
        for f in tqdm(files, desc="indexing Paraná .npy sizes"):
            a = np.load(f, mmap_mode="r")
            counts.append(int(a.shape[0]))
        total = int(sum(counts))
        n_take = min(self.max_samples, total) if self.max_samples > 0 else total
        if n_take <= 0:
            raise RuntimeError("No pixels to sample for MoCo cache")

        # Proportional allocation, at least 1 from each file when possible
        weights = np.asarray(counts, dtype=np.float64)
        weights = weights / weights.sum()
        alloc = np.floor(weights * n_take).astype(np.int64)
        # distribute remainder to largest munis
        remainder = int(n_take - alloc.sum())
        if remainder > 0:
            order = np.argsort(-weights)
            for i in order[:remainder]:
                alloc[i] += 1
        # cap at available
        for i, c in enumerate(counts):
            if alloc[i] > c:
                alloc[i] = c

        self.cache.parent.mkdir(parents=True, exist_ok=True)
        X_list: list[np.ndarray] = []
        for f, n_file, k in tqdm(
            list(zip(files, counts, alloc)),
            desc="sampling pixels into MoCo cache",
            total=len(files),
        ):
            if k <= 0:
                continue
            a = np.load(f, mmap_mode="r")
            idxs = rng.choice(n_file, size=int(k), replace=False)
            # Fancy-index once per file (much faster than per-pixel reads on network FS)
            rows = np.asarray(a[np.sort(idxs)], dtype=np.float32)
            for row in rows:
                cleaned = _clean_pixel_row(row, min_channels=self._min_channels)
                if cleaned is not None and cleaned.shape[0] >= 3:
                    X_list.append(cleaned)

        if len(X_list) < 2:
            raise RuntimeError(
                f"MoCo cache ended with only {len(X_list)} usable series "
                f"(need at least 2). Check nodata / year ranges."
            )

        # Shuffle so train/val split is not municipality-ordered
        order = rng.permutation(len(X_list))
        X_list = [X_list[i] for i in order]
        np.save(self.cache, np.array(X_list, dtype=object), allow_pickle=True)
        self.X_list = X_list
        print(f"  cached {len(self.X_list)} series → {self.cache}")

    def __len__(self) -> int:
        return len(self.X_list)

    def __getitem__(self, index: int):
        X = self.X_list[index]
        X, doy = self.transform(X)
        sample = {"x": X, "doy": doy}
        if self.dataaug is not None:
            q, k = self.dataaug(sample)
            return q, k
        return sample
