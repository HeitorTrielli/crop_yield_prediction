"""
Derived mega-pixel / pixel feature recipes.

Computed at train time from the existing 17-channel .npy
(10 S2 + DOY + 2 rain + 4 cum climate). No new files.

S2 band order (preprocess BANDNAMES / GEE):
  0 B2 blue, 1 B3 green, 2 B4 red, 3 B5 RE1, 4 B6 RE2,
  5 B7 RE3, 6 B8 NIR, 7 B8A, 8 B11 SWIR1, 9 B12 SWIR2

Xavier extras at channels 11:17:
  0 cum rain (mm), 1 dry-streak, 2 cum ETo, 3 cum Rs, 4 cum Tmax, 5 cum Tmin

Temperature "days over/under X" and GDD use interval-mean Tmax/Tmin
(Δcum °C·day / days between S2 dates). That is not a true daily count;
the .npy does not store daily T.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 = range(10)
_EPS = 1e-6


def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / (a + b + _EPS)


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a / (b + _EPS)


INDEX_FNS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "ndvi": lambda x: _nd(x[..., B8], x[..., B4]),
    "gndvi": lambda x: _nd(x[..., B8], x[..., B3]),
    "ndre": lambda x: _nd(x[..., B8A], x[..., B5]),
    "ndwi": lambda x: _nd(x[..., B3], x[..., B8]),
    "ndmi": lambda x: _nd(x[..., B8], x[..., B11]),
    "nbr": lambda x: _nd(x[..., B8], x[..., B12]),
    "evi": lambda x: 2.5
    * (x[..., B8] - x[..., B4])
    / (x[..., B8] + 6.0 * x[..., B4] - 7.5 * x[..., B2] + 1.0),
    "evi2": lambda x: 2.4 * (x[..., B8] - x[..., B4]) / (x[..., B8] + x[..., B4] + 1.0),
    "savi": lambda x: 1.5 * (x[..., B8] - x[..., B4]) / (x[..., B8] + x[..., B4] + 0.5),
    "wdrvi": lambda x: _nd(0.2 * x[..., B8], x[..., B4]),
    "cire": lambda x: _ratio(x[..., B7], x[..., B5]) - 1.0,
    "gci": lambda x: _ratio(x[..., B8], x[..., B3]) - 1.0,
    "msi": lambda x: _ratio(x[..., B11], x[..., B8]),
    "kndvi": lambda x: np.tanh(_nd(x[..., B8], x[..., B4]) ** 2),
}

INDEX_GROUPS: dict[str, tuple[str, ...]] = {
    "ndvi": ("ndvi",),
    "core": ("ndvi", "evi", "ndwi", "ndmi", "ndre"),
    "veg": ("ndvi", "gndvi", "evi", "savi", "ndre", "cire", "wdrvi"),
    "water": ("ndwi", "ndmi", "msi", "nbr"),
    "fullidx": (
        "ndvi",
        "gndvi",
        "evi",
        "evi2",
        "savi",
        "ndre",
        "cire",
        "wdrvi",
        "gci",
        "ndwi",
        "ndmi",
        "msi",
        "nbr",
        "kndvi",
    ),
    "tri": ("ndvi", "evi", "ndwi"),
}

SPECTRAL_GROUPS: dict[str, dict[str, Any]] = {
    "bands": {"spectral": "zscore", "indices": ()},
    "ndvi": {"spectral": "none", "indices": INDEX_GROUPS["ndvi"]},
    "core": {"spectral": "none", "indices": INDEX_GROUPS["core"]},
    "veg": {"spectral": "none", "indices": INDEX_GROUPS["veg"]},
    "water": {"spectral": "none", "indices": INDEX_GROUPS["water"]},
    "fullidx": {"spectral": "none", "indices": INDEX_GROUPS["fullidx"]},
    "tri": {"spectral": "none", "indices": INDEX_GROUPS["tri"]},
    "bands_ndvi": {"spectral": "zscore", "indices": INDEX_GROUPS["ndvi"]},
    "bands_core": {"spectral": "zscore", "indices": INDEX_GROUPS["core"]},
    "bands_fullidx": {"spectral": "zscore", "indices": INDEX_GROUPS["fullidx"]},
}

CLIMATE_WIDTH: dict[str, int] = {
    "none": 0,
    "rain": 2,
    "cum": 6,
    "delta": 6,
    "rate": 6,
    "wb_cum": 6,
    "wb_delta": 6,
    "hydro": 4,
    "thermo": 4,
    "log_cum": 6,
    "zscore": 6,
    "rain_eto": 3,
    # Interval-mean Tmax/Tmin proxies (between photos + season-to-date).
    "hot_days": 8,  # Tmax>28,30,32,35 × (interval, cum)
    "cold_days": 8,  # Tmin<5,10,15,18 × (interval, cum)
    "hot_cold": 8,  # hot 30/35 + cold 10/15 × (interval, cum)
    "gdd": 8,  # GDD 8/10/15 + chill-10, interval and season-to-date
    "stress": 8,  # heat 32/35 + cold 10 (int/cum) + dry streak + rain interval
}

HOT_THRESH_C: tuple[float, ...] = (28.0, 30.0, 32.0, 35.0)
COLD_THRESH_C: tuple[float, ...] = (5.0, 10.0, 15.0, 18.0)
GDD_BASES_C: tuple[float, ...] = (8.0, 10.0, 15.0)

CLIMATE_GROUPS: tuple[str, ...] = tuple(CLIMATE_WIDTH.keys())


def compute_indices(reflectance: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    """reflectance [N, T, 10] in 0–1. Returns [N, T, K] clipped."""
    if not names:
        n, t, _ = reflectance.shape
        return np.zeros((n, t, 0), dtype=np.float32)
    cols = []
    for name in names:
        fn = INDEX_FNS[name]
        cols.append(np.nan_to_num(fn(reflectance), nan=0.0, posinf=0.0, neginf=0.0))
    stacked = np.stack(cols, axis=-1).astype(np.float32)
    return np.clip(stacked, -2.0, 4.0)


def _diff_t(x: np.ndarray) -> np.ndarray:
    """Interval totals: first step keeps season-to-date, later steps are increments."""
    out = np.empty_like(x, dtype=np.float32)
    out[:, 0, ...] = x[:, 0, ...]
    out[:, 1:, ...] = x[:, 1:, ...] - x[:, :-1, ...]
    return out


def _doy_dt(doy: np.ndarray) -> np.ndarray:
    """Days spanned by each observation (handles New Year wrap; t0 = days since Oct 1)."""
    dt = np.ones(doy.shape, dtype=np.float32)
    delta = doy[:, 1:].astype(np.float32) - doy[:, :-1].astype(np.float32)
    delta = np.where(delta <= 0.0, delta + 365.0, delta)
    dt[:, 1:] = np.maximum(delta, 1.0)
    d0 = doy[:, 0].astype(np.float32)
    oct1 = 274.0
    since_oct = np.where(
        d0 >= 200.0,
        d0 - oct1 + 1.0,
        d0 + (365.0 - oct1 + 1.0),
    )
    dt[:, 0] = np.maximum(since_oct, 1.0)
    return dt


def _zscore_t(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return ((x - mu) / sd).astype(np.float32)


def _s_days_interval(v: np.ndarray) -> np.ndarray:
    return np.clip(v / 40.0, 0.0, 4.0).astype(np.float32)


def _s_days_cum(v: np.ndarray) -> np.ndarray:
    return np.clip(v / 150.0, 0.0, 4.0).astype(np.float32)


def _s_gdd_interval(v: np.ndarray) -> np.ndarray:
    return np.clip(v / 250.0, 0.0, 4.0).astype(np.float32)


def _s_gdd_cum(v: np.ndarray) -> np.ndarray:
    return np.clip(v / 2500.0, 0.0, 4.0).astype(np.float32)


def _pair_interval_cum(interval: np.ndarray) -> np.ndarray:
    """[N, T, C] interval counts → [N, T, 2C] (interval scaled, running sum scaled)."""
    cum = np.cumsum(interval, axis=1)
    return np.concatenate([_s_days_interval(interval), _s_days_cum(cum)], axis=-1)


def _days_over_tmax(tmax_mean: np.ndarray, dt: np.ndarray, thresh: float) -> np.ndarray:
    return np.where(tmax_mean > thresh, dt, 0.0).astype(np.float32)


def _days_under_tmin(tmin_mean: np.ndarray, dt: np.ndarray, thresh: float) -> np.ndarray:
    return np.where(tmin_mean < thresh, dt, 0.0).astype(np.float32)


def _gdd_interval(tmean_mean: np.ndarray, dt: np.ndarray, base: float) -> np.ndarray:
    return (np.maximum(tmean_mean - base, 0.0) * dt).astype(np.float32)


def _chill_interval(tmean_mean: np.ndarray, dt: np.ndarray, base: float) -> np.ndarray:
    return (np.maximum(base - tmean_mean, 0.0) * dt).astype(np.float32)


def extras_from_chunk(chunk_arr: np.ndarray) -> np.ndarray:
    """Always return [N, T, 6] rain+climate (zeros if the file is narrower)."""
    n, t, c = chunk_arr.shape
    out = np.zeros((n, t, 6), dtype=np.float32)
    if c >= 17:
        out[...] = chunk_arr[:, :, 11:17]
    elif c >= 13:
        out[:, :, :2] = chunk_arr[:, :, 11:13]
    invalid = (out == -9999) | ~np.isfinite(out)
    out = np.where(invalid, 0.0, out)
    return out.astype(np.float32)


def compute_climate(
    extras: np.ndarray,
    doy: np.ndarray,
    mode: str,
    extra_scaler=None,
) -> np.ndarray:
    """extras [N, T, 6] raw. Returns [N, T, C] scaled ~O(1)."""
    n, t, _ = extras.shape
    if mode == "none":
        return np.zeros((n, t, 0), dtype=np.float32)

    from .extra_scaler import InputScaler, scale_xavier_climate_extras, scale_xavier_rain_channels

    s: InputScaler | None = extra_scaler
    rain_s = None
    all_s = None
    if mode in {
        "rain",
        "cum",
        "delta",
        "rate",
        "wb_delta",
        "hydro",
        "log_cum",
        "stress",
    }:
        s = s or InputScaler.require_load()
        rain_s = scale_xavier_rain_channels(extras[:, :, :2], scaler=s)
    if mode == "cum":
        s = s or InputScaler.require_load()
        all_s = scale_xavier_climate_extras(extras, scaler=s)
    rain_cum = extras[:, :, 0:1]
    dry = rain_s[:, :, 1:2] if rain_s is not None else extras[:, :, 1:2]
    eto = extras[:, :, 2:3]
    rs = extras[:, :, 3:4]
    tmax = extras[:, :, 4:5]
    tmin = extras[:, :, 5:6]
    tmean = 0.5 * (tmax + tmin)
    dtr = tmax - tmin
    wb = rain_cum - eto

    def _need() -> InputScaler:
        nonlocal s
        if s is None:
            s = InputScaler.require_load()
        return s

    def s_rain(v: np.ndarray) -> np.ndarray:
        return _need().zscore_extra(v, 0)

    def s_eto(v: np.ndarray) -> np.ndarray:
        return _need().zscore_extra(v, 2)

    def s_rs(v: np.ndarray) -> np.ndarray:
        return _need().zscore_extra(v, 3)

    def s_tmax(v: np.ndarray) -> np.ndarray:
        return _need().zscore_extra(v, 4)

    def s_tmin(v: np.ndarray) -> np.ndarray:
        return _need().zscore_extra(v, 5)

    def s_wb(v: np.ndarray) -> np.ndarray:
        return _need().zscore_derived(v, "wb_mm")

    def s_dtr(v: np.ndarray) -> np.ndarray:
        return _need().zscore_derived(v, "dtr_cday")

    def s_tmean(v: np.ndarray) -> np.ndarray:
        return _need().zscore_derived(v, "tmean_cday")

    def s_d_rain(v: np.ndarray) -> np.ndarray:
        return _need().zscore_extra_delta(v, 0)

    def s_d_eto(v: np.ndarray) -> np.ndarray:
        return _need().zscore_extra_delta(v, 2)

    def s_d_rs(v: np.ndarray) -> np.ndarray:
        return _need().zscore_extra_delta(v, 3)

    def s_d_tmax(v: np.ndarray) -> np.ndarray:
        return _need().zscore_extra_delta(v, 4)

    def s_d_tmin(v: np.ndarray) -> np.ndarray:
        return _need().zscore_extra_delta(v, 5)

    rain_d = _diff_t(rain_cum)
    eto_d = _diff_t(eto)
    rs_d = _diff_t(rs)
    tmax_d = _diff_t(tmax)
    tmin_d = _diff_t(tmin)
    wb_d = rain_d - eto_d
    dt = _doy_dt(doy)[..., None]
    tmax_mean = tmax_d / dt
    tmin_mean = tmin_d / dt
    tmean_mean = 0.5 * (tmax_mean + tmin_mean)

    if mode == "rain":
        return rain_s
    if mode == "cum":
        return all_s
    if mode == "delta":
        return np.concatenate(
            [
                s_d_rain(rain_d),
                dry,
                s_d_eto(eto_d),
                s_d_rs(rs_d),
                s_d_tmax(tmax_d),
                s_d_tmin(tmin_d),
            ],
            axis=-1,
        )
    if mode == "rate":
        return np.concatenate(
            [
                np.clip(rain_d / dt / 15.0, -4.0, 4.0),
                dry,
                np.clip(eto_d / dt / 8.0, -4.0, 4.0),
                np.clip(rs_d / dt / 25.0, -4.0, 4.0),
                np.clip(tmax_d / dt / 40.0, -4.0, 4.0),
                np.clip(tmin_d / dt / 40.0, -4.0, 4.0),
            ],
            axis=-1,
        ).astype(np.float32)
    if mode == "wb_cum":
        return np.concatenate(
            [s_rain(rain_cum), s_eto(eto), s_wb(wb), s_rs(rs), s_tmean(tmean), s_dtr(dtr)],
            axis=-1,
        )
    if mode == "wb_delta":
        return np.concatenate(
            [
                s_d_rain(rain_d),
                dry,
                s_d_eto(eto_d),
                s_d_rain(wb_d),
                s_d_rs(rs_d),
                s_d_tmax(_diff_t(tmean)),
            ],
            axis=-1,
        )
    if mode == "hydro":
        return np.concatenate(
            [s_rain(rain_cum), dry, s_eto(eto), s_wb(wb)], axis=-1
        )
    if mode == "thermo":
        return np.concatenate([s_tmax(tmax), s_tmin(tmin), s_dtr(dtr), s_rs(rs)], axis=-1)
    if mode == "log_cum":
        return np.concatenate(
            [
                np.clip(np.log1p(np.maximum(rain_cum, 0.0)) / 8.0, 0.0, 2.0),
                dry,
                np.clip(np.log1p(np.maximum(eto, 0.0)) / 8.0, 0.0, 2.0),
                np.clip(np.log1p(np.maximum(rs, 0.0)) / 9.0, 0.0, 2.0),
                s_tmax(tmax),
                s_tmin(tmin),
            ],
            axis=-1,
        ).astype(np.float32)
    if mode == "zscore":
        return _zscore_t(extras)
    if mode == "rain_eto":
        return np.concatenate([s_rain(rain_cum), s_eto(eto), s_wb(wb)], axis=-1)
    if mode == "hot_days":
        cols = [_days_over_tmax(tmax_mean, dt, x) for x in HOT_THRESH_C]
        return _pair_interval_cum(np.concatenate(cols, axis=-1))
    if mode == "cold_days":
        cols = [_days_under_tmin(tmin_mean, dt, x) for x in COLD_THRESH_C]
        return _pair_interval_cum(np.concatenate(cols, axis=-1))
    if mode == "hot_cold":
        cols = [
            _days_over_tmax(tmax_mean, dt, 30.0),
            _days_over_tmax(tmax_mean, dt, 35.0),
            _days_under_tmin(tmin_mean, dt, 10.0),
            _days_under_tmin(tmin_mean, dt, 15.0),
        ]
        return _pair_interval_cum(np.concatenate(cols, axis=-1))
    if mode == "gdd":
        gdd_int = np.concatenate(
            [_gdd_interval(tmean_mean, dt, b) for b in GDD_BASES_C]
            + [_chill_interval(tmean_mean, dt, 10.0)],
            axis=-1,
        )
        gdd_cum = np.cumsum(gdd_int, axis=1)
        return np.concatenate(
            [_s_gdd_interval(gdd_int), _s_gdd_cum(gdd_cum)], axis=-1
        )
    if mode == "stress":
        hot_cold = _pair_interval_cum(
            np.concatenate(
                [
                    _days_over_tmax(tmax_mean, dt, 32.0),
                    _days_over_tmax(tmax_mean, dt, 35.0),
                    _days_under_tmin(tmin_mean, dt, 10.0),
                ],
                axis=-1,
            )
        )
        return np.concatenate(
            [hot_cold, dry, s_d_rain(rain_d)],
            axis=-1,
        )
    raise ValueError(f"Unknown climate mode {mode!r}")


def recipe_input_dim(spectral: str, indices: tuple[str, ...], climate: str) -> int:
    n = 10 if spectral == "zscore" else 0
    n += len(indices)
    n += int(CLIMATE_WIDTH[climate])
    return n


def layout_name(spectral_key: str, climate_key: str) -> str:
    return f"mp_{spectral_key}__{climate_key}"


def recipe_layout_records() -> list[dict[str, Any]]:
    records = []
    for skey, sspec in SPECTRAL_GROUPS.items():
        for ckey in CLIMATE_GROUPS:
            indices = tuple(sspec["indices"])
            spectral = str(sspec["spectral"])
            dim = recipe_input_dim(spectral, indices, ckey)
            need_extras = ckey != "none"
            records.append(
                {
                    "name": layout_name(skey, ckey),
                    "input_dim": dim,
                    "extra_channels_slice": (11, 17) if need_extras else None,
                    "recipe": {
                        "spectral": spectral,
                        "indices": indices,
                        "climate": ckey,
                    },
                    "description": (
                        f"Mega-pixel recipe spectral={skey} ({spectral}, "
                        f"idx={list(indices) or 'none'}) climate={ckey} → dim {dim}."
                    ),
                }
            )
    return records


def sweep_layout_names() -> list[str]:
    return [r["name"] for r in recipe_layout_records()]


def assemble_recipe(
    x_spec_n: np.ndarray,
    x_ref: np.ndarray,
    extras: np.ndarray,
    doy: np.ndarray,
    recipe: dict[str, Any],
    extra_scaler=None,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    if recipe.get("spectral") == "zscore":
        parts.append(x_spec_n)
    idx_names = tuple(recipe.get("indices") or ())
    if idx_names:
        parts.append(compute_indices(x_ref, idx_names))
    climate = str(recipe.get("climate") or "none")
    clim = compute_climate(extras, doy, climate, extra_scaler=extra_scaler)
    if clim.shape[-1] > 0:
        parts.append(clim)
    if not parts:
        raise ValueError(f"Empty feature recipe: {recipe!r}")
    return np.concatenate(parts, axis=-1).astype(np.float32)
