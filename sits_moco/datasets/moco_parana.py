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

from datasets.daily_climate import (
    N_DAILY_CLIMATE,
    climate_sidecar_path,
    load_climate_sidecar,
    season_doy_axis,
)
from datasets.extra_scaler import DEFAULT_INPUT_SCALER_PATH, N_SOIL, InputScaler
from datasets.feature_layout import (
    feature_layout_extra_slice,
    feature_layout_input_dim,
    feature_layout_is_climate_only,
    feature_layout_is_dual,
    feature_layout_needs_soil_sidecar,
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


def soil_path_for_npy(npy: Path) -> Path:
    """``{code}.npy`` → sibling ``{code}_soil.npy``."""
    return npy.with_name(f"{npy.stem}_soil.npy")


def list_muni_npy_files(
    root: Path,
    year_ranges: list[str],
    *,
    require_soil: bool = False,
) -> list[Path]:
    """Return municipal .npy paths (skip ``*.keep_mask.npy``).

    When ``require_soil`` is true, only include files that have a sibling
    ``{code}_soil.npy`` (MapBiomas Solo sidecars for soil layouts).
    """
    files: list[Path] = []
    skipped_no_soil = 0
    for yr in year_ranges:
        season_dir = root / yr
        if not season_dir.is_dir():
            continue
        for muni_dir in sorted(season_dir.iterdir()):
            if not muni_dir.is_dir():
                continue
            npy = muni_dir / f"{muni_dir.name}.npy"
            if not npy.is_file():
                continue
            if require_soil and not soil_path_for_npy(npy).is_file():
                skipped_no_soil += 1
                continue
            files.append(npy)
    if require_soil and skipped_no_soil:
        print(
            f"  Skipped {skipped_no_soil} municipality–year(s) missing "
            f"{{code}}_soil.npy (required for soil MoCo layout)"
        )
    return files


def list_climate_sidecar_files(root: Path, year_ranges: list[str]) -> list[Path]:
    """Return ``{code}_climate_daily.npy`` paths that sit next to a municipal .npy."""
    out: list[Path] = []
    skipped = 0
    for npy in list_muni_npy_files(root, year_ranges, require_soil=False):
        clim = climate_sidecar_path(npy)
        if clim.is_file():
            out.append(clim)
        else:
            skipped += 1
    if skipped:
        print(
            f"  Skipped {skipped} municipality–year(s) missing "
            f"{{code}}_climate_daily.npy (required for daily_climate MoCo)"
        )
    return out


def _clean_climate_row(arr: np.ndarray) -> np.ndarray | None:
    """Drop all-nodata days from a daily climate series ``[T, 5]``."""
    row = np.asarray(arr, dtype=np.float32)
    if row.ndim != 2 or row.shape[1] < N_DAILY_CLIMATE:
        return None
    row = row[:, :N_DAILY_CLIMATE]
    invalid = (row == NO_DATA_VALUE) | ~np.isfinite(row)
    keep = ~np.all(invalid, axis=1)
    if int(np.count_nonzero(keep)) < 3:
        return None
    return np.ascontiguousarray(row[keep], dtype=np.float32)


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
        extra_scaler_path: Path | str | None = None,
    ):
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        self.year_ranges = list(year_ranges)
        self.sequencelength = int(sequencelength)
        self.max_samples = int(max_samples)
        self.seed = int(seed)
        self.feature_layout = normalize_feature_layout(feature_layout)
        if feature_layout_is_dual(self.feature_layout):
            raise ValueError(
                f"MoCo does not support dual yield layout {self.feature_layout!r}. "
                "Pretrain separate trunks with feature_layout='spectral' and "
                "'daily_climate', then load them into DualSTNetRegression."
            )
        self.input_dim = feature_layout_input_dim(self.feature_layout)
        self._climate_only = feature_layout_is_climate_only(self.feature_layout)
        self._needs_soil = (
            False
            if self._climate_only
            else feature_layout_needs_soil_sidecar(self.feature_layout)
        )
        self._min_channels = (
            N_DAILY_CLIMATE
            if self._climate_only
            else _min_channels_for_layout(self.feature_layout)
        )
        self.extra_scaler_path = (
            Path(extra_scaler_path).expanduser()
            if extra_scaler_path is not None
            else DEFAULT_INPUT_SCALER_PATH
        )
        self._pixel_tx: PixelTransform | None = None
        self._climate_scaler: InputScaler | None = None
        if not self._climate_only:
            self._pixel_tx = PixelTransform(
                sequencelength=sequencelength,
                feature_layout=self.feature_layout,
                randomchoice=False,
                interp=False,
                seed=seed,
                extra_scaler_path=self.extra_scaler_path,
            )
        # Parallel list of [4] soil vectors when layout needs sidecars; else unused.
        self.soil_list: list[np.ndarray] | None = None

        if dataaug is not None:
            self.dataaug = moco.loader.TwoCropsTransform(dataaug)
        else:
            self.dataaug = None

        years_tag = "_".join(self.year_ranges).replace("-", "")
        cache_dir = self.root / ".moco_cache"
        # Soil layouts append _v2soil so climate caches stay reusable; soil
        # records are {"x", "soil"} object arrays.
        soil_tag = "_v2soil" if self._needs_soil else ""
        clim_tag = "_clim" if self._climate_only else ""
        self.cache = (
            cache_dir
            / (
                f"Unsupervised_Parana_{years_tag}_{self.feature_layout}"
                f"_N{self.max_samples}_S{self.seed}{soil_tag}{clim_tag}.npy"
            )
        )

        print(
            f"Load Paraná unsupervised MoCo set "
            f"(years={self.year_ranges}, layout={self.feature_layout}, "
            f"input_dim={self.input_dim}, soil_sidecar={self._needs_soil}, "
            f"climate_only={self._climate_only}, "
            f"max_samples={self.max_samples}, seed={self.seed})"
        )
        if self.cache.exists() and not rebuild_cache:
            self.load_cached_dataset()
        else:
            self.cache_dataset()

    def transform(self, x: np.ndarray, soil: np.ndarray | None = None):
        """Normalize features with the same layout rules as yield training."""
        if self._climate_only:
            return self._transform_climate(x)
        assert self._pixel_tx is not None
        chunk = np.ascontiguousarray(x[None, ...], dtype=np.float32)
        soil_chunk = None
        if self._needs_soil:
            if soil is None:
                raise ValueError(
                    f"Layout {self.feature_layout!r} requires a soil vector [4] "
                    "alongside each cached time series"
                )
            soil_arr = np.asarray(soil, dtype=np.float32).reshape(-1)
            if soil_arr.shape[0] < N_SOIL:
                raise ValueError(
                    f"soil must have >= {N_SOIL} channels, got {soil_arr.shape}"
                )
            soil_chunk = np.ascontiguousarray(
                soil_arr[:N_SOIL][None, :], dtype=np.float32
            )
        feats, _weight, doy = self._pixel_tx.features_from_chunk(
            chunk, soil=soil_chunk
        )
        return feats[0].astype(np.float32), doy[0].astype(np.float32)

    def _transform_climate(self, x: np.ndarray):
        """Z-score daily climate and attach season-relative DOY (1 = Oct 1)."""
        if self._climate_scaler is None:
            self._climate_scaler = InputScaler.require_load(self.extra_scaler_path)
        clim = np.asarray(x, dtype=np.float32)
        if clim.ndim != 2 or clim.shape[1] < N_DAILY_CLIMATE:
            raise ValueError(
                f"daily climate series must be [T, {N_DAILY_CLIMATE}], got {clim.shape}"
            )
        clim_z = self._climate_scaler.transform_daily_climate(
            clim[:, :N_DAILY_CLIMATE]
        )
        doy = season_doy_axis(clim_z.shape[0]).astype(np.float32)
        return clim_z.astype(np.float32), doy

    def load_cached_dataset(self) -> None:
        print(f"precached dataset files found at {self.cache}")
        raw = np.load(self.cache, allow_pickle=True).tolist()
        if not raw:
            raise RuntimeError(f"Empty MoCo cache at {self.cache}")
        first = raw[0]
        if isinstance(first, dict) and "x" in first:
            self.X_list = [np.asarray(item["x"], dtype=np.float32) for item in raw]
            if self._needs_soil:
                self.soil_list = [
                    np.asarray(item["soil"], dtype=np.float32).reshape(N_SOIL)
                    for item in raw
                ]
            else:
                self.soil_list = None
        elif isinstance(first, (tuple, list)) and len(first) == 2:
            self.X_list = [np.asarray(item[0], dtype=np.float32) for item in raw]
            self.soil_list = [
                np.asarray(item[1], dtype=np.float32).reshape(N_SOIL) for item in raw
            ]
        else:
            # Legacy: list of [T, C] arrays (no soil).
            self.X_list = [np.asarray(item, dtype=np.float32) for item in raw]
            self.soil_list = None
            if self._needs_soil:
                raise RuntimeError(
                    f"MoCo cache {self.cache} has no soil vectors but layout "
                    f"{self.feature_layout!r} requires them. Delete the cache "
                    f"or pass --rebuild-cache."
                )
        print(f"  loaded {len(self.X_list)} time series")

    def cache_dataset(self) -> None:
        if self._climate_only:
            self._cache_climate_dataset()
            return

        files = list_muni_npy_files(
            self.root, self.year_ranges, require_soil=self._needs_soil
        )
        if not files:
            hint = (
                " Build soil sidecars with data_download/build_mapbiomas_solo_sidecars.py."
                if self._needs_soil
                else ""
            )
            raise FileNotFoundError(
                f"No municipal .npy under {self.root} for year ranges "
                f"{self.year_ranges}.{hint}"
            )

        rng = np.random.default_rng(self.seed)
        # Count pixels per file (mmap header only); for soil layouts also check N match.
        counts: list[int] = []
        usable_files: list[Path] = []
        for f in tqdm(files, desc="indexing Paraná .npy sizes"):
            a = np.load(f, mmap_mode="r")
            n = int(a.shape[0])
            if self._needs_soil:
                soil_p = soil_path_for_npy(f)
                soil_a = np.load(soil_p, mmap_mode="r")
                if soil_a.ndim != 2 or int(soil_a.shape[0]) != n:
                    print(
                        f"  skip {f}: soil shape {getattr(soil_a, 'shape', None)} "
                        f"!= npy N={n}"
                    )
                    continue
                if int(soil_a.shape[1]) < N_SOIL:
                    print(
                        f"  skip {f}: soil has {soil_a.shape[1]} channels "
                        f"(need >= {N_SOIL})"
                    )
                    continue
            usable_files.append(f)
            counts.append(n)
        files = usable_files
        if not files:
            raise FileNotFoundError(
                "No usable municipality files after soil shape checks."
            )

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
        records: list[dict[str, np.ndarray]] = []
        for f, n_file, k in tqdm(
            list(zip(files, counts, alloc)),
            desc="sampling pixels into MoCo cache",
            total=len(files),
        ):
            if k <= 0:
                continue
            a = np.load(f, mmap_mode="r")
            idxs = rng.choice(n_file, size=int(k), replace=False)
            order = np.argsort(idxs)
            idxs_sorted = idxs[order]
            # Fancy-index once per file (much faster than per-pixel reads on network FS)
            rows = np.asarray(a[idxs_sorted], dtype=np.float32)
            soil_rows = None
            if self._needs_soil:
                soil_a = np.load(soil_path_for_npy(f), mmap_mode="r")
                soil_rows = np.asarray(soil_a[idxs_sorted, :N_SOIL], dtype=np.float32)
            for i, row in enumerate(rows):
                cleaned = _clean_pixel_row(row, min_channels=self._min_channels)
                if cleaned is None or cleaned.shape[0] < 3:
                    continue
                rec: dict[str, np.ndarray] = {"x": cleaned}
                if self._needs_soil:
                    assert soil_rows is not None
                    rec["soil"] = np.ascontiguousarray(
                        soil_rows[i, :N_SOIL], dtype=np.float32
                    )
                records.append(rec)

        if len(records) < 2:
            raise RuntimeError(
                f"MoCo cache ended with only {len(records)} usable series "
                f"(need at least 2). Check nodata / year ranges / soil sidecars."
            )

        # Shuffle so train/val split is not municipality-ordered
        order = rng.permutation(len(records))
        records = [records[i] for i in order]
        self.X_list = [r["x"] for r in records]
        if self._needs_soil:
            self.soil_list = [r["soil"] for r in records]
            np.save(self.cache, np.array(records, dtype=object), allow_pickle=True)
        else:
            # Legacy format: object array of [T, C] only (no soil).
            self.soil_list = None
            np.save(
                self.cache,
                np.array(self.X_list, dtype=object),
                allow_pickle=True,
            )
        print(f"  cached {len(self.X_list)} series → {self.cache}")

    def _cache_climate_dataset(self) -> None:
        """One MoCo series per municipality–year daily climate sidecar."""
        files = list_climate_sidecar_files(self.root, self.year_ranges)
        if not files:
            raise FileNotFoundError(
                f"No {{code}}_climate_daily.npy under {self.root} for year ranges "
                f"{self.year_ranges}. Build with "
                "data_download/build_muni_daily_climate_npy.py."
            )

        rng = np.random.default_rng(self.seed)
        n_take = (
            min(self.max_samples, len(files)) if self.max_samples > 0 else len(files)
        )
        if n_take < len(files):
            pick = rng.choice(len(files), size=n_take, replace=False)
            files = [files[i] for i in sorted(pick)]

        self.cache.parent.mkdir(parents=True, exist_ok=True)
        series: list[np.ndarray] = []
        for f in tqdm(files, desc="loading daily climate sidecars for MoCo"):
            cleaned = _clean_climate_row(load_climate_sidecar(f))
            if cleaned is None:
                continue
            series.append(cleaned)

        if len(series) < 2:
            raise RuntimeError(
                f"Climate MoCo cache ended with only {len(series)} usable series "
                f"(need at least 2). Check sidecars under {self.root}."
            )

        order = rng.permutation(len(series))
        self.X_list = [series[i] for i in order]
        self.soil_list = None
        np.save(
            self.cache,
            np.array(self.X_list, dtype=object),
            allow_pickle=True,
        )
        print(f"  cached {len(self.X_list)} climate series → {self.cache}")

    def __len__(self) -> int:
        return len(self.X_list)

    def __getitem__(self, index: int):
        X = self.X_list[index]
        soil = None if self.soil_list is None else self.soil_list[index]
        X, doy = self.transform(X, soil=soil)
        sample = {"x": X, "doy": doy}
        if self.dataaug is not None:
            q, k = self.dataaug(sample)
            return q, k
        return sample
