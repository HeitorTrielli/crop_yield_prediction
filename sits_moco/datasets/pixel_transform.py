"""
Single source of truth for .npy pixel time-series → STNet inputs (training & inference).

Used by USCropsAggregatedNPY and any script that loads preprocessed municipality .npy files.
"""

from __future__ import annotations

import numpy as np
import torch

from .datautils import getWeight_batch
from .feature_layout import normalize_feature_layout, resolve_feature_layout

DOY_CHANNEL = 10
NUM_SPECTRAL_CHANNELS = 10


def scale_xavier_rain_channels(rain: np.ndarray) -> np.ndarray:
    out = np.asarray(rain, dtype=np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out[..., 0] = np.clip(out[..., 0] / 2500.0, 0.0, 4.0)
    out[..., 1] = np.clip(out[..., 1] / 60.0, 0.0, 2.0)
    return out.astype(np.float32)


# Sentinel-2 reflectance normalization (training default)
SPECTRAL_MEAN = np.array(
    [[0.147, 0.169, 0.186, 0.221, 0.273, 0.297, 0.308, 0.316, 0.256, 0.188]],
    dtype=np.float32,
)
SPECTRAL_STD = np.array(
    [0.227, 0.219, 0.222, 0.22, 0.2, 0.193, 0.192, 0.182, 0.123, 0.106],
    dtype=np.float32,
)

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
    ):
        self.sequencelength = int(sequencelength)
        self.feature_layout = normalize_feature_layout(feature_layout)
        lay = resolve_feature_layout(self.feature_layout)
        self.input_feature_dim = int(lay["input_dim"])
        self._extra_channels_slice: tuple[int, int] | None = lay["extra_channels_slice"]
        self.rc = bool(randomchoice)
        self.interp = bool(interp)
        self.getWeight_batch = getWeight_batch
        self.mean = SPECTRAL_MEAN
        self.std = SPECTRAL_STD

    def features_from_chunk(
        self, chunk_arr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalized features (N,T,F), reflectance weights (N,T), DOY (N,T)."""
        _n, _t, c = chunk_arr.shape
        doy = chunk_arr[:, :, DOY_CHANNEL].astype(np.int32)
        x_spec = chunk_arr[:, :, :NUM_SPECTRAL_CHANNELS].astype(np.float32) * 1e-4
        weight = self.getWeight_batch(x_spec)
        x_spec_n = (x_spec - self.mean) / self.std
        sl = self._extra_channels_slice
        if sl is not None:
            lo, hi = sl
            n_extra = hi - lo
            if c >= hi:
                extra = chunk_arr[:, :, lo:hi].astype(np.float32)
                if (lo, hi) == (11, 13):
                    extra = scale_xavier_rain_channels(extra)
                else:
                    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                extra = np.zeros((_n, _t, n_extra), dtype=np.float32)
            x = np.concatenate([x_spec_n, extra], axis=-1)
        else:
            x = x_spec_n
        return x, weight, doy

    def transform_chunk(self, chunk_arr: np.ndarray) -> BatchChunk:
        """Return (x, mask, doy, weight) each [N, T, ...] — batched, no per-pixel list."""
        n, t, _ = chunk_arr.shape
        x, weight, doy = self.features_from_chunk(chunk_arr)
        fdim = self.input_feature_dim
        seq_len = self.sequencelength

        if self.interp:
            doy_pad = np.linspace(0, 366, seq_len).astype(np.int32)
            x_pad = np.zeros((n, seq_len, fdim), dtype=np.float32)
            for i in range(n):
                x_pad[i] = np.array(
                    [np.interp(doy_pad, doy[i], x[i, :, j]) for j in range(fdim)]
                ).T
            spec_denorm = x_pad[:, :, :NUM_SPECTRAL_CHANNELS] * self.std + self.mean
            weight_pad = self.getWeight_batch(spec_denorm)
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
            else:
                idxs = np.random.choice(t, seq_len, replace=False)
                idxs.sort()
                x_pad = x[:, idxs, :]
                doy_pad_broadcast = doy[:, idxs]
                weight_pad = weight[:, idxs]
                weight_pad /= weight_pad.sum(axis=1, keepdims=True)
                mask = np.ones((n, seq_len), dtype=np.int32)

        x_pad = np.ascontiguousarray(x_pad.astype(np.float32))
        mask_bool = mask == 0
        doy_pad_broadcast = np.ascontiguousarray(doy_pad_broadcast.astype(np.int64))
        weight_pad = np.ascontiguousarray(weight_pad.astype(np.float32))

        return (
            torch.from_numpy(x_pad),
            torch.from_numpy(mask_bool),
            torch.from_numpy(doy_pad_broadcast),
            torch.from_numpy(weight_pad),
        )
