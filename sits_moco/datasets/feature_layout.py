"""
Explicit per-pixel feature layouts for municipality .npy → STNet regression.

Layouts are named strings (CLI / config), not inferred from files. Extend the registry
when adding channels (e.g. evapotranspiration) after updating preprocess and scaling.
"""

from __future__ import annotations

from typing import Any

# name -> input_dim for STNet MLP, optional [start, end) slice of .npy channels to append after spectral+DOY block
# Channels 0..9 = spectral, 10 = DOY; extras live at 11+ (see preprocessing/preprocess_daily_to_npy).
_LAYOUT: dict[str, dict[str, Any]] = {
    "spectral": {
        "input_dim": 10,
        "extra_channels_slice": None,
        "description": "10 Sentinel-2 bands only (normalized); DOY is separate → positional encoding.",
    },
    "spectral_xavier": {
        "input_dim": 12,
        "extra_channels_slice": (11, 13),
        "description": "10 bands + 2 Xavier rain channels (11:13), scaled; DOY → PE.",
    },
    "spectral_xavier_climate": {
        "input_dim": 16,
        "extra_channels_slice": (11, 17),
        "description": (
            "10 bands + 2 rain (11:13) + cum ETo/Rs/Tmax/Tmin (13:17), scaled; DOY → PE. "
            "Requires 17-channel daily .npy."
        ),
    },
}


def feature_layout_choices() -> tuple[str, ...]:
    return tuple(sorted(_LAYOUT.keys()))


def normalize_feature_layout(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    aliases = {
        "s2": "spectral",
        "s2_only": "spectral",
        "xavier": "spectral_xavier",
        "xavier_rain": "spectral_xavier",
        "s2_xavier": "spectral_xavier",
        "xavier_climate": "spectral_xavier_climate",
        "s2_xavier_climate": "spectral_xavier_climate",
        "spectral_xavier_clim": "spectral_xavier_climate",
    }
    key = aliases.get(key, key)
    if key not in _LAYOUT:
        raise ValueError(
            f"Unknown feature layout {name!r}. Choose one of: {', '.join(feature_layout_choices())}"
        )
    return key


def resolve_feature_layout(name: str) -> dict[str, Any]:
    """Return a copy of the layout record for the canonical name."""
    k = normalize_feature_layout(name)
    return dict(_LAYOUT[k], name=k)


def feature_layout_input_dim(name: str) -> int:
    return int(resolve_feature_layout(name)["input_dim"])


def feature_layout_extra_slice(name: str) -> tuple[int, int] | None:
    """If set, .npy must have at least ``end`` channels; slice ``[lo:hi)`` is stacked after spectral."""
    sl = resolve_feature_layout(name)["extra_channels_slice"]
    if sl is None:
        return None
    lo, hi = sl
    return (int(lo), int(hi))
