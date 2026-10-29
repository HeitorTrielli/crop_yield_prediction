"""Daily Xavier climate sidecars for the dual-STNet path.

Sidecar: ``{code}_climate_daily.npy`` next to the municipal time-series .npy.
Shape ``[T, 5]`` (also accepts ``[1, T, 5]`` / ``[N, T, 5]``), raw daily values:

  0 pr (mm/day), 1 ETo (mm/day), 2 Rs (MJ/m^2/day), 3 Tmax (°C), 4 Tmin (°C)

``T`` is the inclusive Oct 1–Mar 31 window (182 days, 183 in leap harvest years).
Season-relative DOY is 1 = Oct 1, matching Sentinel cubes.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch

from .constants import NO_DATA_VALUE

N_DAILY_CLIMATE = 5
CLIMATE_MAX_SEQ_LEN = 183  # Oct 1–Mar 31 including a leap-year February
DAILY_CLIMATE_CHANNEL_NAMES: tuple[str, ...] = (
    "pr_mm",
    "eto_mm",
    "rs_mjm2",
    "tmax_c",
    "tmin_c",
)
# Extra columns derived from the raw daily sidecar. They are appended only
# to the climate trunk (after the five Xavier channels), never to Sentinel.
CLIMATE_TIMING_CHANNEL_NAMES: tuple[str, ...] = (
    "dry_spell_days",
    "gdd_cum_c",
    "wb_daily_mm",
)
DRY_SPELL_MM = 1.0
SOY_GDD_TBASE_C = 10.0
SOY_GDD_TMAX_CAP_C = 30.0
_CLIMATE_EXTRA_ALIASES: dict[str, str] = {
    "dry_spell": "dry_spell_days",
    "dry_spell_days": "dry_spell_days",
    "dryspell": "dry_spell_days",
    "gdd": "gdd_cum_c",
    "gdd_cum": "gdd_cum_c",
    "gdd_cum_c": "gdd_cum_c",
    "graus_dia": "gdd_cum_c",
    "growing_degree_days": "gdd_cum_c",
    "wb": "wb_daily_mm",
    "wb_daily": "wb_daily_mm",
    "wb_daily_mm": "wb_daily_mm",
    "water_balance": "wb_daily_mm",
    "rain_eto": "wb_daily_mm",
    "pr_eto": "wb_daily_mm",
}
CLIMATE_SIDECAR_SUFFIX = "_climate_daily.npy"

_YEAR_RANGE_RE = re.compile(r"^(\d{4})-(\d{4})$")


def climate_sidecar_path(npy_path: Path | str) -> Path:
    p = Path(npy_path)
    return p.with_name(f"{p.stem}{CLIMATE_SIDECAR_SUFFIX}")


def climate_window_dates(year_range: str) -> tuple[date, date]:
    """Inclusive [Oct 1 y1, Mar 31 y2] for folder label y1-y2."""
    m = _YEAR_RANGE_RE.match(year_range.strip())
    if not m:
        raise ValueError(f"Expected year-range like 2020-2021, got {year_range!r}")
    y1, y2 = int(m.group(1)), int(m.group(2))
    return date(y1, 10, 1), date(y2, 3, 31)


def climate_window_len(year_range: str) -> int:
    start, end = climate_window_dates(year_range)
    return (end - start).days + 1


def dates_inclusive(start: date, end: date) -> list[date]:
    out: list[date] = []
    d = start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def season_doy_axis(n_days: int) -> np.ndarray:
    """Season-relative DOY, 1 = Oct 1."""
    return np.arange(1, int(n_days) + 1, dtype=np.int32)


def harvest_year_to_year_range(harvest_year: int) -> str:
    y = int(harvest_year)
    return f"{y - 1}-{y}"


def load_climate_sidecar(path: Path | str) -> np.ndarray:
    """Return float32 ``[T, C]`` (C = 5). Accepts ``[T,C]``, ``[1,T,C]``, ``[N,T,C]``."""
    arr = np.asarray(np.load(path), dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            with np.errstate(invalid="ignore"):
                arr = np.nanmean(
                    np.where(
                        (arr == NO_DATA_VALUE) | ~np.isfinite(arr),
                        np.nan,
                        arr,
                    ),
                    axis=0,
                ).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Climate sidecar {path} must be [T,C] or [N,T,C], got {arr.shape}")
    if arr.shape[1] < N_DAILY_CLIMATE:
        raise ValueError(
            f"Climate sidecar {path} has C={arr.shape[1]}, expected >= {N_DAILY_CLIMATE}"
        )
    return np.ascontiguousarray(arr[:, :N_DAILY_CLIMATE], dtype=np.float32)


def broadcast_climate(climate_tc: np.ndarray, n_pixels: int) -> np.ndarray:
    """``[T, C]`` → ``[N, T, C]``."""
    clim = np.ascontiguousarray(climate_tc, dtype=np.float32)
    if clim.ndim != 2:
        raise ValueError(f"Expected [T, C] climate, got {clim.shape}")
    return np.broadcast_to(clim[None, :, :], (int(n_pixels), clim.shape[0], clim.shape[1])).copy()


def parse_climate_extra_features(spec: str | None) -> tuple[str, ...]:
    """Parse a comma list into climate-trunk extra column names.

    Accepted tokens: ``dry_spell``, ``gdd`` (soybean growing degree days),
    ``wb`` (daily rain minus ETo), or ``all``. Empty / ``none`` means no extras.
    """
    text = "" if spec is None else str(spec).strip()
    if not text or text.lower() in {"none", "off", "false"}:
        return ()
    out: list[str] = []
    for part in text.replace(";", ",").split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key == "all":
            for name in CLIMATE_TIMING_CHANNEL_NAMES:
                if name not in out:
                    out.append(name)
            continue
        name = _CLIMATE_EXTRA_ALIASES.get(key)
        if name is None:
            known = ", ".join(("dry_spell", "gdd", "wb", "all"))
            raise ValueError(
                f"Unknown climate extra {part.strip()!r}. Expected one of: {known}"
            )
        if name not in out:
            out.append(name)
    return tuple(out)


def climate_extra_spec_from_run_config(run_config: dict | None) -> str:
    """CLI string saved on the training run, or empty when the run has none."""
    cli = (run_config or {}).get("cli") or {}
    val = cli.get("climate_extra_features")
    return "" if val is None else str(val)


def climate_timing_features(raw_tc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derived daily climate columns from one raw series ``[T, 5]``.

    Returns ``(features, valid)`` with features ``[T, 3]`` in the order
    ``dry_spell_days, gdd_cum_c, wb_daily_mm`` and a boolean mask of the same
    shape. Invalid source days are finite on the cumulative columns (the
    running value is frozen) and marked False in ``valid`` so scaler moments
    can ignore them.

    * dry spell: consecutive days with rain < 1 mm. A missing rain day keeps
      the current streak and does not increment or reset it.
    * soybean growing degree days: cumulative
      ``max(0, (min(Tmax, 30) + Tmin) / 2 - 10)`` from the first day of the
      series (Oct 1). A missing temperature day does not add degree days.
    * water balance: that day's rain minus ETo, not a running sum.
    """
    x = np.asarray(raw_tc, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < N_DAILY_CLIMATE:
        raise ValueError(
            f"climate_timing_features expects [T, {N_DAILY_CLIMATE}], got {x.shape}"
        )
    pr = x[:, 0]
    eto = x[:, 1]
    tmax = x[:, 3]
    tmin = x[:, 4]
    pr_bad = (pr == NO_DATA_VALUE) | ~np.isfinite(pr)
    eto_bad = (eto == NO_DATA_VALUE) | ~np.isfinite(eto)
    t_bad = (
        (tmax == NO_DATA_VALUE)
        | ~np.isfinite(tmax)
        | (tmin == NO_DATA_VALUE)
        | ~np.isfinite(tmin)
    )
    n = int(x.shape[0])
    dry = np.zeros(n, dtype=np.float32)
    gdd = np.zeros(n, dtype=np.float32)
    wb = np.zeros(n, dtype=np.float32)
    dry_ok = np.zeros(n, dtype=bool)
    gdd_ok = np.zeros(n, dtype=bool)
    wb_ok = np.zeros(n, dtype=bool)
    streak = 0.0
    cum = 0.0
    for i in range(n):
        if pr_bad[i]:
            dry[i] = streak
        else:
            if float(pr[i]) < DRY_SPELL_MM:
                streak += 1.0
            else:
                streak = 0.0
            dry[i] = streak
            dry_ok[i] = True
        if t_bad[i]:
            gdd[i] = cum
        else:
            tmx = min(float(tmax[i]), SOY_GDD_TMAX_CAP_C)
            daily = max(0.0, 0.5 * (tmx + float(tmin[i])) - SOY_GDD_TBASE_C)
            cum += daily
            gdd[i] = cum
            gdd_ok[i] = True
        if pr_bad[i] or eto_bad[i]:
            wb[i] = 0.0
        else:
            wb[i] = float(pr[i]) - float(eto[i])
            wb_ok[i] = True
    feats = np.stack([dry, gdd, wb], axis=1).astype(np.float32, copy=False)
    valid = np.stack([dry_ok, gdd_ok, wb_ok], axis=1)
    return feats, valid


def climate_valid_mask(climate_ntc: np.ndarray) -> np.ndarray:
    """True where any daily channel is finite and not nodata. ``[N, T]``."""
    invalid = (climate_ntc == NO_DATA_VALUE) | ~np.isfinite(climate_ntc)
    return ~np.all(invalid, axis=-1)


def pad_climate_to_length(
    x: np.ndarray,
    doy: np.ndarray,
    weight: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pad/trim climate batch to ``seq_len``. Returns (x, mask_bool, doy, weight)."""
    n, t, c = x.shape
    seq_len = int(seq_len)
    if t == seq_len:
        x_pad = x
        doy_pad = doy
        weight_pad = weight
        mask = np.ones((n, seq_len), dtype=np.int32)
    elif t < seq_len:
        mask = np.zeros((n, seq_len), dtype=np.int32)
        mask[:, :t] = 1
        x_pad = np.zeros((n, seq_len, c), dtype=np.float32)
        x_pad[:, :t, :] = x
        doy_pad = np.zeros((n, seq_len), dtype=np.int32)
        doy_pad[:, :t] = doy
        weight_pad = np.zeros((n, seq_len), dtype=np.float64)
        weight_pad[:, :t] = weight
    else:
        x_pad = x[:, :seq_len, :]
        doy_pad = doy[:, :seq_len]
        weight_pad = weight[:, :seq_len]
        mask = np.ones((n, seq_len), dtype=np.int32)

    wsum = weight_pad.sum(axis=1, keepdims=True)
    weight_pad = weight_pad / np.where(wsum > 0, wsum, 1.0)
    mask_bool = mask == 0
    return (
        np.ascontiguousarray(x_pad.astype(np.float32)),
        mask_bool,
        np.ascontiguousarray(doy_pad.astype(np.int64)),
        np.ascontiguousarray(weight_pad.astype(np.float32)),
    )


def climate_to_tensors(
    x: np.ndarray,
    mask_bool: np.ndarray,
    doy: np.ndarray,
    weight: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(x),
        torch.from_numpy(mask_bool),
        torch.from_numpy(doy),
        torch.from_numpy(weight),
    )
