"""
Single source of truth for .npy pixel time-series → STNet inputs (training & inference).

Used by USCropsAggregatedNPY and any script that loads preprocessed municipality .npy files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .datautils import getWeight_batch
from .extra_scaler import (
    DEFAULT_INPUT_SCALER_PATH,
    InputScaler,
    scale_soil_channels,
    scale_soil_channels_legacy,
    scale_xavier_climate_extras,
    scale_xavier_rain_channels,
)
from .feature_layout import normalize_feature_layout, resolve_feature_layout
from .feature_recipes import assemble_recipe, extras_from_chunk
from .constants import NO_DATA_VALUE
from .daily_climate import (
    CLIMATE_MAX_SEQ_LEN,
    N_DAILY_CLIMATE,
    broadcast_climate,
    climate_to_tensors,
    pad_climate_to_length,
    season_doy_axis,
)

DOY_CHANNEL = 10
NUM_SPECTRAL_CHANNELS = 10
N_SOIL = 4


# Legacy hardcoded S2 stats (pre-train-split scaler / commit 05a1ea2).
# Default path loads mean/std from files/train_input_scaler.json once the scaler is loaded.
SPECTRAL_MEAN = np.array(
    [[0.147, 0.169, 0.186, 0.221, 0.273, 0.297, 0.308, 0.316, 0.256, 0.188]],
    dtype=np.float32,
)
SPECTRAL_STD = np.array(
    [0.227, 0.219, 0.222, 0.22, 0.2, 0.193, 0.192, 0.182, 0.123, 0.106],
    dtype=np.float32,
)


def scale_xavier_rain_channels_legacy(rain: np.ndarray) -> np.ndarray:
    """Hardcoded rain/dry divisors from pre-JSON-scaler training (05a1ea2)."""
    out = np.asarray(rain, dtype=np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out[..., 0] = np.clip(out[..., 0] / 2500.0, 0.0, 4.0)
    out[..., 1] = np.clip(out[..., 1] / 60.0, 0.0, 2.0)
    return out.astype(np.float32)


def scale_xavier_climate_extras_legacy(extra: np.ndarray) -> np.ndarray:
    """
    Hardcoded rain+climate divisors from pre-JSON-scaler training (05a1ea2).

    Channels: rain, dry streak, cum ETo, Rs, Tmax, Tmin.
    """
    out = np.asarray(extra, dtype=np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    if out.shape[-1] < 6:
        raise ValueError(
            f"Expected at least 6 Xavier climate extras (rain+ETo/Rs/T), got {out.shape[-1]}"
        )
    out[..., 0] = np.clip(out[..., 0] / 2500.0, 0.0, 4.0)
    out[..., 1] = np.clip(out[..., 1] / 60.0, 0.0, 2.0)
    out[..., 2] = np.clip(out[..., 2] / 1500.0, 0.0, 4.0)  # cum ETo (mm)
    out[..., 3] = np.clip(out[..., 3] / 5000.0, 0.0, 4.0)  # cum Rs (MJ/m^2)
    out[..., 4] = np.clip(out[..., 4] / 5000.0, 0.0, 4.0)  # cum Tmax (°C·day)
    out[..., 5] = np.clip(out[..., 5] / 5000.0, -1.0, 4.0)  # cum Tmin (°C·day)
    return out.astype(np.float32)


PixelTuple = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
BatchChunk = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class PixelTransform:
    """Pad / sample / interpolate pixel chunks to sequencelength (same logic as training)."""

    def __init__(
        self,
        sequencelength: int,
        feature_layout: str = "spectral",
        *,
        randomchoice: bool = False,
        interp: bool = False,
        seed: int = 27,
        extra_scaler: InputScaler | None = None,
        extra_scaler_path: str | Path | None = None,
        legacy_input_scaling: bool = False,
        deterministic_head: bool = False,
        climate_sequencelength: int | None = None,
    ):
        self.sequencelength = int(sequencelength)
        self.feature_layout = normalize_feature_layout(feature_layout)
        lay = resolve_feature_layout(self.feature_layout)
        self.input_feature_dim = int(lay["input_dim"])
        self._extra_channels_slice: tuple[int, int] | None = lay["extra_channels_slice"]
        self._soil_sidecar = bool(lay.get("soil_sidecar"))
        self._climate_sidecar = bool(lay.get("climate_sidecar"))
        self.climate_input_dim = int(lay.get("climate_input_dim") or 0)
        self.climate_sequencelength = int(
            climate_sequencelength
            if climate_sequencelength is not None
            else CLIMATE_MAX_SEQ_LEN
        )
        self._recipe = lay.get("recipe")
        self.rc = bool(randomchoice)
        self.interp = bool(interp)
        self.deterministic_head = bool(deterministic_head)
        self.getWeight_batch = getWeight_batch
        self.legacy_input_scaling = bool(legacy_input_scaling)
        self.mean = SPECTRAL_MEAN
        self.std = SPECTRAL_STD
        self._extra_scaler = extra_scaler
        self.extra_scaler_path = (
            Path(extra_scaler_path)
            if extra_scaler_path is not None
            else DEFAULT_INPUT_SCALER_PATH
        )
        if self.legacy_input_scaling:
            if self._recipe is not None:
                raise ValueError(
                    "legacy_input_scaling only supports slice layouts "
                    "(spectral / spectral_xavier / spectral_xavier_climate / "
                    "spectral_xavier_climate_soil), not recipes or dual layouts"
                )
            if self._climate_sidecar:
                raise ValueError(
                    "legacy_input_scaling does not support dual daily-climate layouts"
                )
            self._extra_scaler = None
        elif extra_scaler is not None:
            self._apply_spectral_stats(extra_scaler)

    def _apply_spectral_stats(self, scaler: InputScaler) -> None:
        self.mean = scaler.spectral_mean_row
        self.std = scaler.spectral_std

    def set_extra_scaler(self, scaler: InputScaler | None) -> None:
        if self.legacy_input_scaling:
            return
        self._extra_scaler = scaler
        if scaler is not None:
            self._apply_spectral_stats(scaler)
            if scaler.path is not None:
                self.extra_scaler_path = scaler.path

    def extra_scaler(self) -> InputScaler:
        if self.legacy_input_scaling:
            raise RuntimeError(
                "legacy_input_scaling is on: train_input_scaler.json is not used"
            )
        if self._extra_scaler is None:
            self._extra_scaler = InputScaler.require_load(self.extra_scaler_path)
            self._apply_spectral_stats(self._extra_scaler)
        return self._extra_scaler

    def _drop_empty_spectral_days(self, chunk_arr: np.ndarray) -> np.ndarray:
        """Keep timesteps where any pixel has a valid S2 observation."""
        spec = chunk_arr[:, :, :NUM_SPECTRAL_CHANNELS]
        day_valid = np.any(
            (spec != NO_DATA_VALUE) & (spec != 0) & np.isfinite(spec),
            axis=(0, 2),
        )
        if not np.any(day_valid):
            return chunk_arr[:, :0, :]
        if bool(np.all(day_valid)):
            return chunk_arr
        return np.ascontiguousarray(chunk_arr[:, day_valid, :])

    def transform_climate(
        self,
        climate: np.ndarray,
        n_pixels: int,
    ) -> BatchChunk:
        """Z-score and pad daily climate to climate_sequencelength."""
        if climate.ndim == 2:
            clim_tc = np.ascontiguousarray(climate, dtype=np.float32)
            if clim_tc.shape[-1] < N_DAILY_CLIMATE:
                raise ValueError(
                    f"climate has C={clim_tc.shape[-1]}, expected {N_DAILY_CLIMATE}"
                )
            clim_ntc = broadcast_climate(clim_tc[..., :N_DAILY_CLIMATE], n_pixels)
        elif climate.ndim == 3:
            arr = np.ascontiguousarray(climate, dtype=np.float32)
            if arr.shape[-1] < N_DAILY_CLIMATE:
                raise ValueError(
                    f"climate has C={arr.shape[-1]}, expected {N_DAILY_CLIMATE}"
                )
            arr = arr[..., :N_DAILY_CLIMATE]
            if arr.shape[0] == n_pixels:
                clim_ntc = arr
            elif arr.shape[0] == 1:
                clim_ntc = broadcast_climate(arr[0], n_pixels)
            else:
                with np.errstate(all="ignore"):
                    mean_tc = np.nanmean(
                        np.where(
                            (arr == NO_DATA_VALUE) | ~np.isfinite(arr),
                            np.nan,
                            arr,
                        ),
                        axis=0,
                    ).astype(np.float32)
                clim_ntc = broadcast_climate(mean_tc, n_pixels)
        else:
            raise ValueError(f"climate must be [T,C] or [N,T,C], got {climate.shape}")

        invalid = (clim_ntc == NO_DATA_VALUE) | ~np.isfinite(clim_ntc)
        clim_z = np.where(invalid, 0.0, clim_ntc).astype(np.float32)
        scaler = self.extra_scaler()
        clim_z = scaler.transform_daily_climate(clim_z)
        valid = ~np.all(invalid, axis=-1)
        weight = valid.astype(np.float64)
        doy = np.broadcast_to(
            season_doy_axis(clim_z.shape[1])[None, :],
            (n_pixels, clim_z.shape[1]),
        ).copy()
        x_pad, mask_bool, doy_pad, weight_pad = pad_climate_to_length(
            clim_z, doy, weight, self.climate_sequencelength
        )
        return climate_to_tensors(x_pad, mask_bool, doy_pad, weight_pad)

    def features_from_chunk(
        self,
        chunk_arr: np.ndarray,
        soil: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalized features (N,T,F), reflectance weights (N,T), DOY (N,T)."""
        _n, _t, c = chunk_arr.shape
        doy = chunk_arr[:, :, DOY_CHANNEL].astype(np.int32)
        x_spec = chunk_arr[:, :, :NUM_SPECTRAL_CHANNELS].astype(np.float32)
        # Treat preprocess nodata (-9999 / non-finite) like 0 before reflectance scale.
        # Leaving -9999 as-is yields ~-1 after *1e-4 and extreme z-scores.
        invalid = (x_spec == NO_DATA_VALUE) | ~np.isfinite(x_spec)
        x_spec = np.where(invalid, 0.0, x_spec) * 1e-4
        weight = self.getWeight_batch(x_spec)
        if self.legacy_input_scaling:
            x_spec_n = ((x_spec - self.mean) / self.std).astype(np.float32)
            scaler = None
        else:
            scaler = self.extra_scaler()
            x_spec_n = scaler.transform_spectral(x_spec)
        if self._recipe is not None:
            if self._soil_sidecar:
                raise ValueError(
                    f"Layout {self.feature_layout!r} uses soil_sidecar with a recipe; "
                    "soil is not supported for ma_* / mp_* recipes."
                )
            extras = extras_from_chunk(chunk_arr)
            recipe_scaler = (
                scaler
                if self._recipe.get("climate") not in (None, "none")
                else None
            )
            x = assemble_recipe(
                x_spec_n,
                x_spec,
                extras,
                doy,
                self._recipe,
                extra_scaler=recipe_scaler,
            )
            if x.shape[-1] != self.input_feature_dim:
                raise ValueError(
                    f"Recipe {self.feature_layout!r} produced {x.shape[-1]} features, "
                    f"expected input_dim={self.input_feature_dim}"
                )
            return x, weight, doy
        sl = self._extra_channels_slice
        if sl is not None:
            lo, hi = sl
            n_extra = hi - lo
            if c >= hi:
                extra = chunk_arr[:, :, lo:hi].astype(np.float32)
                extra = np.where(
                    (extra == NO_DATA_VALUE) | ~np.isfinite(extra), 0.0, extra
                )
                if (lo, hi) == (11, 13):
                    if self.legacy_input_scaling:
                        extra = scale_xavier_rain_channels_legacy(extra)
                    else:
                        extra = scale_xavier_rain_channels(extra, scaler=scaler)
                elif (lo, hi) == (11, 17):
                    if self.legacy_input_scaling:
                        extra = scale_xavier_climate_extras_legacy(extra)
                    else:
                        extra = scale_xavier_climate_extras(extra, scaler=scaler)
                else:
                    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                extra = np.zeros((_n, _t, n_extra), dtype=np.float32)
            x = np.concatenate([x_spec_n, extra], axis=-1)
        else:
            x = x_spec_n

        if self._soil_sidecar:
            if soil is None:
                raise ValueError(
                    f"Layout {self.feature_layout!r} requires a soil sidecar "
                    f"[N,{N_SOIL}] passed to features_from_chunk/transform_chunk"
                )
            soil_arr = np.asarray(soil, dtype=np.float32)
            if soil_arr.ndim != 2 or soil_arr.shape[0] != _n:
                raise ValueError(
                    f"soil must be [N,{N_SOIL}] with N={_n}, got shape {soil_arr.shape}"
                )
            if soil_arr.shape[1] < N_SOIL:
                raise ValueError(
                    f"soil must have >= {N_SOIL} channels, got {soil_arr.shape[1]}"
                )
            soil_arr = soil_arr[:, :N_SOIL]
            soil_arr = np.where(
                (soil_arr == NO_DATA_VALUE) | ~np.isfinite(soil_arr), 0.0, soil_arr
            )
            if self.legacy_input_scaling:
                soil_s = scale_soil_channels_legacy(soil_arr)
            else:
                soil_s = scale_soil_channels(soil_arr, scaler=scaler)
            soil_bt = np.broadcast_to(soil_s[:, None, :], (_n, _t, N_SOIL))
            x = np.concatenate([x, np.ascontiguousarray(soil_bt)], axis=-1)

        if x.shape[-1] != self.input_feature_dim:
            raise ValueError(
                f"Layout {self.feature_layout!r} produced {x.shape[-1]} features, "
                f"expected input_dim={self.input_feature_dim}"
            )
        return x, weight, doy

    def transform_chunk(
        self,
        chunk_arr: np.ndarray,
        soil: np.ndarray | None = None,
        climate: np.ndarray | None = None,
    ) -> BatchChunk:
        """Return (x, mask, doy, weight) or dual 8-tuple with climate tensors."""
        if chunk_arr.dtype != np.float32 or not chunk_arr.flags.c_contiguous:
            chunk_arr = np.ascontiguousarray(chunk_arr, dtype=np.float32)
        if self._climate_sidecar:
            chunk_arr = self._drop_empty_spectral_days(chunk_arr)
        n, t, _ = chunk_arr.shape
        if t == 0:
            fdim = self.input_feature_dim
            x = np.zeros((n, 0, fdim), dtype=np.float32)
            weight = np.zeros((n, 0), dtype=np.float64)
            doy = np.zeros((n, 0), dtype=np.int32)
        else:
            x, weight, doy = self.features_from_chunk(chunk_arr, soil=soil)
        fdim = self.input_feature_dim
        seq_len = self.sequencelength

        if self.interp:
            doy_pad = np.linspace(0, 366, seq_len).astype(np.int32)
            x_pad = np.zeros((n, seq_len, fdim), dtype=np.float32)
            for i in range(n):
                x_pad[i] = np.array(
                    [np.interp(doy_pad, doy[i], x[i, :, j]) for j in range(fdim)]
                ).T
            if self._recipe is None or self._recipe.get("spectral") == "zscore":
                spec_denorm = x_pad[:, :, :NUM_SPECTRAL_CHANNELS] * self.std + self.mean
                weight_pad = self.getWeight_batch(spec_denorm)
            else:
                weight_pad = np.ones((n, seq_len), dtype=np.float64)
                weight_pad /= float(seq_len)
            mask = np.ones((n, seq_len), dtype=np.int32)
            doy_pad_broadcast = np.tile(doy_pad.astype(np.int64), (n, 1))
        elif self.rc:
            replace = t >= seq_len
            idxs = np.random.choice(t, seq_len, replace=replace)
            idxs.sort()
            x_pad = x[:, idxs, :]
            doy_pad_broadcast = doy[:, idxs]
            weight_pad = weight[:, idxs]
            weight_pad /= weight_pad.sum(axis=1, keepdims=True)
            mask = np.ones((n, seq_len), dtype=np.int32)
        else:
            if t == seq_len:
                x_pad = x
                doy_pad_broadcast = doy
                weight_pad = weight
                weight_pad /= weight_pad.sum(axis=1, keepdims=True)
                mask = np.ones((n, seq_len), dtype=np.int32)
            elif t < seq_len:
                mask = np.zeros((n, seq_len), dtype=np.int32)
                mask[:, :t] = 1
                x_pad = np.zeros((n, seq_len, fdim), dtype=np.float32)
                x_pad[:, :t, :] = x
                doy_pad_broadcast = np.zeros((n, seq_len), dtype=np.int32)
                doy_pad_broadcast[:, :t] = doy
                weight_pad = np.zeros((n, seq_len), dtype=np.float64)
                weight_pad[:, :t] = weight
                wsum = weight_pad.sum(axis=1, keepdims=True)
                weight_pad /= np.where(wsum > 0, wsum, 1.0)
            elif self.deterministic_head:
                # Keep earliest sequencelength timesteps (heatmap / incomplete-series).
                x_pad = x[:, :seq_len, :]
                doy_pad_broadcast = doy[:, :seq_len]
                weight_pad = weight[:, :seq_len]
                weight_pad /= weight_pad.sum(axis=1, keepdims=True)
                mask = np.ones((n, seq_len), dtype=np.int32)
            else:
                idxs = np.random.choice(t, seq_len, replace=False)
                idxs.sort()
                x_pad = x[:, idxs, :]
                doy_pad_broadcast = doy[:, idxs]
                weight_pad = weight[:, idxs]
                weight_pad /= weight_pad.sum(axis=1, keepdims=True)
                mask = np.ones((n, seq_len), dtype=np.int32)

        x_pad = np.ascontiguousarray(x_pad.astype(np.float32))
        if self._soil_sidecar and soil is not None and x_pad.shape[-1] >= N_SOIL:
            # Soil is static: keep it on padded S2 steps so late fusion can read t=0.
            soil_arr = np.asarray(soil, dtype=np.float32)
            if soil_arr.ndim == 2 and soil_arr.shape[0] == n and soil_arr.shape[1] >= N_SOIL:
                soil_arr = soil_arr[:, :N_SOIL]
                soil_arr = np.where(
                    (soil_arr == NO_DATA_VALUE) | ~np.isfinite(soil_arr), 0.0, soil_arr
                )
                if self.legacy_input_scaling:
                    soil_s = scale_soil_channels_legacy(soil_arr)
                else:
                    soil_s = scale_soil_channels(soil_arr, scaler=self.extra_scaler())
                x_pad[:, :, -N_SOIL:] = soil_s[:, None, :]
        mask_bool = mask == 0
        doy_pad_broadcast = np.ascontiguousarray(doy_pad_broadcast.astype(np.int64))
        weight_pad = np.ascontiguousarray(weight_pad.astype(np.float32))

        s2 = (
            torch.from_numpy(x_pad),
            torch.from_numpy(mask_bool),
            torch.from_numpy(doy_pad_broadcast),
            torch.from_numpy(weight_pad),
        )
        if not self._climate_sidecar:
            return s2
        if climate is None:
            raise ValueError(
                f"Layout {self.feature_layout!r} requires a daily climate sidecar "
                "passed to transform_chunk"
            )
        clim = self.transform_climate(climate, n)
        return s2 + clim
