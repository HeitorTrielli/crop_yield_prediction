"""
Train-split StandardScaler for every npy input channel.

Fit once on the training municipalities (spatial mean per date, so large
municipalities do not dominate). Persist mean/std/var and reuse for
val/test/inference. No clipping.

Channels (reflectance 0–1 after ×1e-4, extras in raw npy units):
  spectral: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
  extras:   rain_cum, dry_streak, eto_cum, rs_cum, tmax_cum, tmin_cum

Derived extras (same dates, for recipe encodings that are not npy columns):
  wb = rain − ETo, dtr = Tmax − Tmin, tmean = 0.5 × (Tmax + Tmin)
  extras_delta: interval increments of the 6 extras (same rule as feature_recipes._diff_t)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_SCALER_PATH = REPO_ROOT / "files" / "train_input_scaler.json"
DEFAULT_EXTRA_SCALER_PATH = DEFAULT_INPUT_SCALER_PATH

NO_DATA_VALUE = -9999
STD_FLOOR = 1e-6
N_SPECTRAL = 10
N_EXTRA = 6

SPECTRAL_CHANNEL_NAMES: tuple[str, ...] = (
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
)
EXTRA_CHANNEL_NAMES: tuple[str, ...] = (
    "rain_cum_mm",
    "dry_streak_days",
    "eto_cum_mm",
    "rs_cum_mjm2",
    "tmax_cum_cday",
    "tmin_cum_cday",
)
DERIVED_CHANNEL_NAMES: tuple[str, ...] = ("wb_mm", "dtr_cday", "tmean_cday")
CHANNEL_NAMES = EXTRA_CHANNEL_NAMES  # ExtraScaler / older call sites

_LOAD_CACHE: dict[str, "InputScaler"] = {}


def layout_needs_extra_scaler(feature_layout: str) -> bool:
    """Kept for call sites; every layout now z-scores S2 from the train scaler."""
    return True


def raw_spectral_nan(chunk: np.ndarray) -> np.ndarray:
    """Extract [N, T, 10] reflectance; nodata / 0 / non-finite → NaN (for fitting)."""
    if chunk.ndim != 3:
        raise ValueError(f"Expected [N, T, C] npy, got shape {chunk.shape}")
    spec = chunk[:, :, :N_SPECTRAL].astype(np.float32, copy=False)
    invalid = (spec == NO_DATA_VALUE) | (spec == 0) | ~np.isfinite(spec)
    refl = np.where(invalid, np.nan, spec * 1e-4)
    return refl.astype(np.float32)


def raw_extras_nan(chunk: np.ndarray) -> np.ndarray:
    """Extract [N, T, 6] rain+climate; nodata / non-finite → NaN (for fitting)."""
    if chunk.ndim != 3:
        raise ValueError(f"Expected [N, T, C] npy, got shape {chunk.shape}")
    n, t, c = chunk.shape
    out = np.full((n, t, N_EXTRA), np.nan, dtype=np.float32)
    if c >= 17:
        out[...] = chunk[:, :, 11:17].astype(np.float32, copy=False)
    elif c >= 13:
        out[:, :, :2] = chunk[:, :, 11:13].astype(np.float32, copy=False)
    invalid = (out == NO_DATA_VALUE) | ~np.isfinite(out)
    return np.where(invalid, np.nan, out).astype(np.float32)


def _moments(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Population mean/std/var along axis 0, ignoring NaN. std floor applied later."""
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0, ddof=0)
    var = np.nanvar(mat, axis=0, ddof=0)
    n_per = np.isfinite(mat).sum(axis=0).astype(np.int64)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std >= STD_FLOOR), std, 1.0)
    var = np.where(np.isfinite(var), var, 0.0)
    return mean, std, var, n_per


def _diff_t_2d(x: np.ndarray) -> np.ndarray:
    """Same interval rule as feature_recipes._diff_t, for [T, C]."""
    out = np.empty_like(x, dtype=np.float32)
    out[0] = x[0]
    out[1:] = x[1:] - x[:-1]
    return out


def _block(
    names: tuple[str, ...],
    mean: np.ndarray,
    std: np.ndarray,
    var: np.ndarray,
    n_per: np.ndarray,
) -> dict[str, Any]:
    return {
        "channels": list(names),
        "mean": [float(x) for x in np.asarray(mean).reshape(-1)],
        "std": [float(x) for x in np.asarray(std).reshape(-1)],
        "var": [float(x) for x in np.asarray(var).reshape(-1)],
        "n_frames_per_channel": [int(x) for x in np.asarray(n_per).reshape(-1)],
    }


class InputScaler:
    """Train-set (x − mean) / std for spectral, extras, extra-deltas, and derived extras."""

    def __init__(
        self,
        *,
        spectral_mean: np.ndarray,
        spectral_std: np.ndarray,
        extra_mean: np.ndarray,
        extra_std: np.ndarray,
        spectral_var: np.ndarray | None = None,
        extra_var: np.ndarray | None = None,
        extra_delta_mean: np.ndarray | None = None,
        extra_delta_std: np.ndarray | None = None,
        extra_delta_var: np.ndarray | None = None,
        derived_mean: dict[str, float] | None = None,
        derived_std: dict[str, float] | None = None,
        derived_var: dict[str, float] | None = None,
        n_frames: int | dict[str, int] = 0,
        path: Path | str | None = None,
        meta: dict[str, Any] | None = None,
    ):
        self.spectral_mean = _vec(spectral_mean, N_SPECTRAL, "spectral mean")
        self.spectral_std = _std_vec(spectral_std, N_SPECTRAL, "spectral std")
        self.spectral_var = _var_or_sq(spectral_var, self.spectral_std, N_SPECTRAL)

        self.mean = _vec(extra_mean, N_EXTRA, "extra mean")
        self.std = _std_vec(extra_std, N_EXTRA, "extra std")
        self.var = _var_or_sq(extra_var, self.std, N_EXTRA)

        if extra_delta_mean is None:
            extra_delta_mean = np.zeros(N_EXTRA, dtype=np.float32)
            extra_delta_std = np.ones(N_EXTRA, dtype=np.float32)
            extra_delta_var = np.ones(N_EXTRA, dtype=np.float32)
        self.extra_delta_mean = _vec(extra_delta_mean, N_EXTRA, "extra-delta mean")
        self.extra_delta_std = _std_vec(extra_delta_std, N_EXTRA, "extra-delta std")
        self.extra_delta_var = _var_or_sq(
            extra_delta_var, self.extra_delta_std, N_EXTRA
        )

        d_mean = dict(derived_mean or {})
        d_std = dict(derived_std or {})
        d_var = dict(derived_var or {})
        for name in DERIVED_CHANNEL_NAMES:
            d_mean.setdefault(name, 0.0)
            d_std.setdefault(name, 1.0)
            d_var.setdefault(name, 1.0)
            if d_std[name] < STD_FLOOR:
                d_std[name] = 1.0
        self.derived_mean = {k: float(d_mean[k]) for k in DERIVED_CHANNEL_NAMES}
        self.derived_std = {k: float(d_std[k]) for k in DERIVED_CHANNEL_NAMES}
        self.derived_var = {k: float(d_var[k]) for k in DERIVED_CHANNEL_NAMES}

        self.n_frames = n_frames
        self.path = Path(path) if path is not None else None
        self.meta = dict(meta or {})

    @property
    def spectral_mean_row(self) -> np.ndarray:
        return self.spectral_mean.reshape(1, -1)

    def transform_spectral(self, spec: np.ndarray) -> np.ndarray:
        """Z-score reflectance [..., 10]. No clip."""
        out = np.asarray(spec, dtype=np.float32)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        if int(out.shape[-1]) != N_SPECTRAL:
            raise ValueError(
                f"Expected last dim {N_SPECTRAL} (S2 bands), got {out.shape[-1]}"
            )
        return ((out - self.spectral_mean) / self.spectral_std).astype(np.float32)

    def transform(self, extra: np.ndarray) -> np.ndarray:
        """Z-score extras [..., 2] rain or [..., 6] rain+climate. No clip."""
        out = np.asarray(extra, dtype=np.float32)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        n = int(out.shape[-1])
        if n == 2:
            return ((out - self.mean[:2]) / self.std[:2]).astype(np.float32)
        if n == N_EXTRA:
            return ((out - self.mean) / self.std).astype(np.float32)
        raise ValueError(
            f"Expected last dim 2 (rain) or {N_EXTRA} (rain+climate), got {n}"
        )

    def zscore_extra(self, x: np.ndarray, channel: int | str) -> np.ndarray:
        idx = EXTRA_CHANNEL_NAMES.index(channel) if isinstance(channel, str) else int(channel)
        out = np.asarray(x, dtype=np.float32)
        return ((out - self.mean[idx]) / self.std[idx]).astype(np.float32)

    def zscore_extra_delta(self, x: np.ndarray, channel: int | str) -> np.ndarray:
        idx = EXTRA_CHANNEL_NAMES.index(channel) if isinstance(channel, str) else int(channel)
        out = np.asarray(x, dtype=np.float32)
        return ((out - self.extra_delta_mean[idx]) / self.extra_delta_std[idx]).astype(
            np.float32
        )

    def zscore_derived(self, x: np.ndarray, name: str) -> np.ndarray:
        if name not in self.derived_mean:
            raise KeyError(f"Unknown derived channel {name!r}")
        out = np.asarray(x, dtype=np.float32)
        return ((out - self.derived_mean[name]) / self.derived_std[name]).astype(
            np.float32
        )

    def to_dict(self) -> dict[str, Any]:
        n_frames = self.n_frames
        if isinstance(n_frames, dict):
            n_json: Any = {k: int(v) for k, v in n_frames.items()}
        else:
            n_json = int(n_frames)
        payload = {
            "version": 2,
            "aggregation": "spatial_mean_per_muni_date",
            "std_floor": STD_FLOOR,
            "n_frames": n_json,
            "spectral": _block(
                SPECTRAL_CHANNEL_NAMES,
                self.spectral_mean,
                self.spectral_std,
                self.spectral_var,
                np.zeros(N_SPECTRAL, dtype=int),
            ),
            "extras": _block(
                EXTRA_CHANNEL_NAMES,
                self.mean,
                self.std,
                self.var,
                np.zeros(N_EXTRA, dtype=int),
            ),
            "extras_delta": _block(
                tuple(f"{n}_delta" for n in EXTRA_CHANNEL_NAMES),
                self.extra_delta_mean,
                self.extra_delta_std,
                self.extra_delta_var,
                np.zeros(N_EXTRA, dtype=int),
            ),
            "derived": {
                name: {
                    "mean": self.derived_mean[name],
                    "std": self.derived_std[name],
                    "var": self.derived_var[name],
                }
                for name in DERIVED_CHANNEL_NAMES
            },
            "units": {
                "spectral": "reflectance_0_1_after_1e-4",
                "extras": "raw_npy_units",
            },
        }
        # Prefer per-channel counts stored in meta when present.
        meta = dict(self.meta)
        for key, block in (
            ("spectral_n_frames_per_channel", "spectral"),
            ("extras_n_frames_per_channel", "extras"),
            ("extras_delta_n_frames_per_channel", "extras_delta"),
        ):
            if key in meta:
                payload[block]["n_frames_per_channel"] = meta.pop(key)
        payload.update(meta)
        return payload

    def save(self, path: Path | str | None = None) -> Path:
        dest = Path(path) if path is not None else self.path
        if dest is None:
            dest = DEFAULT_INPUT_SCALER_PATH
        dest = dest.expanduser().resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        self.path = dest
        _LOAD_CACHE.pop(str(dest), None)
        return dest

    @classmethod
    def load(cls, path: Path | str | None = None) -> InputScaler:
        src = Path(path) if path is not None else DEFAULT_INPUT_SCALER_PATH
        src = src.expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(_missing_scaler_msg(src))
        payload = json.loads(src.read_text(encoding="utf-8"))
        version = int(payload.get("version", 1))
        if version < 2 or "spectral" not in payload:
            raise ValueError(
                f"{src} is an extras-only scaler (version {version}). "
                "Re-fit all channels with: python preprocessing/fit_xavier_extra_scaler.py"
            )
        spec = payload["spectral"]
        extra = payload["extras"]
        delta = payload.get("extras_delta") or {}
        derived = payload.get("derived") or {}
        reserved = {
            "version",
            "aggregation",
            "std_floor",
            "n_frames",
            "spectral",
            "extras",
            "extras_delta",
            "derived",
            "units",
            "channels",
            "mean",
            "std",
            "var",
        }
        meta = {k: v for k, v in payload.items() if k not in reserved}
        d_mean = {k: float(derived[k]["mean"]) for k in derived if "mean" in derived[k]}
        d_std = {k: float(derived[k]["std"]) for k in derived if "std" in derived[k]}
        d_var = {k: float(derived[k]["var"]) for k in derived if "var" in derived[k]}
        return cls(
            spectral_mean=spec["mean"],
            spectral_std=spec["std"],
            spectral_var=spec.get("var"),
            extra_mean=extra["mean"],
            extra_std=extra["std"],
            extra_var=extra.get("var"),
            extra_delta_mean=delta.get("mean"),
            extra_delta_std=delta.get("std"),
            extra_delta_var=delta.get("var"),
            derived_mean=d_mean,
            derived_std=d_std,
            derived_var=d_var,
            n_frames=payload.get("n_frames", 0),
            path=src,
            meta=meta,
        )

    @classmethod
    def try_load(cls, path: Path | str | None = None) -> InputScaler | None:
        src = Path(path) if path is not None else DEFAULT_INPUT_SCALER_PATH
        src = src.expanduser().resolve()
        if not src.is_file():
            return None
        return cls.load(src)

    @classmethod
    def require_load(cls, path: Path | str | None = None) -> InputScaler:
        src = Path(path) if path is not None else DEFAULT_INPUT_SCALER_PATH
        key = str(src.expanduser().resolve())
        cached = _LOAD_CACHE.get(key)
        if cached is not None:
            return cached
        loaded = cls.try_load(path)
        if loaded is None:
            raise FileNotFoundError(_missing_scaler_msg(Path(key)))
        _LOAD_CACHE[key] = loaded
        return loaded

    def summary_lines(self) -> list[str]:
        loc = str(self.path) if self.path is not None else "(in-memory)"
        lines = [f"Train input scaler: {loc}  n_frames={self.n_frames}"]
        lines.append("  spectral (reflectance)")
        for name, mu, sd, va in zip(
            SPECTRAL_CHANNEL_NAMES, self.spectral_mean, self.spectral_std, self.spectral_var
        ):
            lines.append(f"    {name:18s}  mean={mu:12.6f}  std={sd:12.6f}  var={va:12.6f}")
        lines.append("  extras (raw npy)")
        for name, mu, sd, va in zip(EXTRA_CHANNEL_NAMES, self.mean, self.std, self.var):
            lines.append(f"    {name:18s}  mean={mu:12.4f}  std={sd:12.4f}  var={va:12.4f}")
        lines.append("  extras_delta")
        for name, mu, sd, va in zip(
            EXTRA_CHANNEL_NAMES,
            self.extra_delta_mean,
            self.extra_delta_std,
            self.extra_delta_var,
        ):
            lines.append(
                f"    {name+'_d':18s}  mean={mu:12.4f}  std={sd:12.4f}  var={va:12.4f}"
            )
        lines.append("  derived")
        for name in DERIVED_CHANNEL_NAMES:
            lines.append(
                f"    {name:18s}  mean={self.derived_mean[name]:12.4f}  "
                f"std={self.derived_std[name]:12.4f}  var={self.derived_var[name]:12.4f}"
            )
        return lines


ExtraScaler = InputScaler


def scale_xavier_rain_channels(
    rain: np.ndarray, scaler: InputScaler | None = None
) -> np.ndarray:
    """Z-score rain extras with the train-set scaler. No clip."""
    scaler = scaler or InputScaler.require_load()
    return scaler.transform(rain)


def scale_xavier_climate_extras(
    extra: np.ndarray, scaler: InputScaler | None = None
) -> np.ndarray:
    """Z-score rain+climate extras with the train-set scaler. No clip."""
    scaler = scaler or InputScaler.require_load()
    return scaler.transform(extra)


def fit_extra_scaler_from_dataset(dataset, *, progress: bool = True) -> InputScaler:
    """
    Fit mean/std on the train dataset for S2 + extras.

    Each municipality–year contributes its spatial mean series, so a large
    municipality counts as one series, not N pixels.
    """
    keys = list(getattr(dataset, "municipality_list", []))
    if not keys:
        raise RuntimeError("Cannot fit train input scaler: train dataset is empty")

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    iterator = keys
    if progress and tqdm is not None:
        iterator = tqdm(keys, desc="Fit train input scaler", unit="muni")

    spec_frames: list[np.ndarray] = []
    extra_frames: list[np.ndarray] = []
    delta_frames: list[np.ndarray] = []
    n_series = 0
    n_missing = 0
    for key in iterator:
        if isinstance(key, tuple):
            code, year = key
        else:
            code = key
            year = dataset.municipality_years.get(code, getattr(dataset, "year", None))
        path = dataset._resolve_npy_path(code, year)
        if path is None or not path.is_file():
            n_missing += 1
            continue
        try:
            data = np.load(path, mmap_mode="r")
        except OSError:
            n_missing += 1
            continue
        if data.ndim != 3 or data.shape[2] < 11:
            n_missing += 1
            continue
        arr = np.asarray(data)
        spec = raw_spectral_nan(arr)
        with np.errstate(all="ignore"):
            spatial_s = np.nanmean(spec, axis=0)
        if np.isfinite(spatial_s).any():
            spec_frames.append(spatial_s.astype(np.float32, copy=False))
        if arr.shape[2] >= 13:
            extras = raw_extras_nan(arr)
            with np.errstate(all="ignore"):
                spatial_e = np.nanmean(extras, axis=0)
            if np.isfinite(spatial_e).any():
                extra_frames.append(spatial_e.astype(np.float32, copy=False))
                delta_frames.append(_diff_t_2d(spatial_e))
        n_series += 1

    if not spec_frames:
        raise RuntimeError(
            "Cannot fit train input scaler: no train .npy files with spectral bands"
        )

    s_mean, s_std, s_var, s_n = _moments(np.concatenate(spec_frames, axis=0))
    if extra_frames:
        extra_mat = np.concatenate(extra_frames, axis=0)
        e_mean, e_std, e_var, e_n = _moments(extra_mat)
        d_mean, d_std, d_var, d_n = _moments(np.concatenate(delta_frames, axis=0))
        rain = extra_mat[:, 0]
        eto = extra_mat[:, 2]
        tmax = extra_mat[:, 4]
        tmin = extra_mat[:, 5]
        derived_mat = np.stack(
            [rain - eto, tmax - tmin, 0.5 * (tmax + tmin)],
            axis=1,
        )
        der_mean, der_std, der_var, _der_n = _moments(derived_mat)
        derived_mean = {
            name: float(der_mean[i]) for i, name in enumerate(DERIVED_CHANNEL_NAMES)
        }
        derived_std = {
            name: float(der_std[i]) for i, name in enumerate(DERIVED_CHANNEL_NAMES)
        }
        derived_var = {
            name: float(der_var[i]) for i, name in enumerate(DERIVED_CHANNEL_NAMES)
        }
    else:
        e_mean = np.zeros(N_EXTRA)
        e_std = np.ones(N_EXTRA)
        e_var = np.ones(N_EXTRA)
        e_n = np.zeros(N_EXTRA, dtype=np.int64)
        d_mean = np.zeros(N_EXTRA)
        d_std = np.ones(N_EXTRA)
        d_var = np.ones(N_EXTRA)
        d_n = np.zeros(N_EXTRA, dtype=np.int64)
        derived_mean = derived_std = derived_var = None

    harvest_years: list[int] = []
    if keys and isinstance(keys[0], tuple):
        harvest_years = sorted({int(y) for _, y in keys})

    n_frames = {
        "spectral": int(s_n.max()) if s_n.size else 0,
        "extras": int(e_n.max()) if e_n.size else 0,
        "extras_delta": int(d_n.max()) if d_n.size else 0,
    }
    meta = {
        "datapath": str(Path(dataset.root).expanduser().resolve()),
        "n_series": int(n_series),
        "n_missing": int(n_missing),
        "spectral_n_frames_per_channel": [int(x) for x in s_n],
        "extras_n_frames_per_channel": [int(x) for x in e_n],
        "extras_delta_n_frames_per_channel": [int(x) for x in d_n],
        "harvest_years": harvest_years,
        "feature_layout": getattr(dataset, "feature_layout", None),
    }
    return InputScaler(
        spectral_mean=s_mean,
        spectral_std=s_std,
        spectral_var=s_var,
        extra_mean=e_mean,
        extra_std=e_std,
        extra_var=e_var,
        extra_delta_mean=d_mean,
        extra_delta_std=d_std,
        extra_delta_var=d_var,
        derived_mean=derived_mean,
        derived_std=derived_std,
        derived_var=derived_var,
        n_frames=n_frames,
        meta=meta,
    )


def load_or_fit_extra_scaler(
    dataset,
    path: Path | str | None = None,
    *,
    refit: bool = False,
    progress: bool = True,
    verbose: bool = True,
) -> InputScaler:
    dest = Path(path) if path is not None else DEFAULT_INPUT_SCALER_PATH
    dest = dest.expanduser()
    if dest.is_file() and not refit:
        scaler = InputScaler.load(dest)
        if verbose:
            print(f"Loaded train input scaler from {dest}")
            for line in scaler.summary_lines()[1:]:
                print(line)
            current_root = str(Path(dataset.root).expanduser().resolve())
            saved_root = scaler.meta.get("datapath")
            if saved_root and Path(saved_root).resolve() != Path(current_root).resolve():
                print(
                    f"  Note: scaler was fit on {saved_root}; current train root is {current_root}. "
                    "Pass --refit-extra-scaler to recompute."
                )
        return scaler

    if verbose:
        action = "Refitting" if refit and dest.is_file() else "Fitting"
        print(f"{action} train input scaler (S2 + extras) on the train split → {dest}")
    scaler = fit_extra_scaler_from_dataset(dataset, progress=progress)
    scaler.save(dest)
    if verbose:
        print(f"Saved train input scaler to {dest}")
        for line in scaler.summary_lines()[1:]:
            print(line)
    return scaler


def apply_extra_scaler_to_dataset(dataset, scaler: InputScaler) -> None:
    dataset.pixel_transform.set_extra_scaler(scaler)


def _vec(values: np.ndarray | list, n: int, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size != n:
        raise ValueError(f"{label} expects {n} values, got {arr.size}")
    return arr


def _std_vec(values: np.ndarray | list, n: int, label: str) -> np.ndarray:
    arr = _vec(values, n, label)
    return np.where(arr < STD_FLOOR, 1.0, arr).astype(np.float32)


def _var_or_sq(
    var: np.ndarray | list | None, std: np.ndarray, n: int
) -> np.ndarray:
    if var is None:
        return (std.astype(np.float64) ** 2).astype(np.float32)
    return _vec(var, n, "var")


def _missing_scaler_msg(path: Path) -> str:
    return (
        f"Train input scaler not found at {path}. "
        "Train once (it is fit on the train split and saved) or run "
        "python preprocessing/fit_xavier_extra_scaler.py"
    )
