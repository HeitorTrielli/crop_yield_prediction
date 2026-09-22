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
        "description": (
            "10 Sentinel-2 bands only, z-scored from the train split "
            "(files/train_input_scaler.json); DOY is separate → positional encoding."
        ),
    },
    "spectral_xavier": {
        "input_dim": 12,
        "extra_channels_slice": (11, 13),
        "description": (
            "10 bands + 2 Xavier rain channels (11:13), z-scored from the train split "
            "(files/train_input_scaler.json); DOY → PE."
        ),
    },
    "spectral_xavier_climate": {
        "input_dim": 16,
        "extra_channels_slice": (11, 17),
        "description": (
            "10 bands + 2 rain (11:13) + cum ETo/Rs/Tmax/Tmin (13:17), z-scored from "
            "the train split (files/train_input_scaler.json); DOY → PE. "
            "Requires 17-channel daily .npy."
        ),
    },
    "spectral_xavier_climate_soil": {
        "input_dim": 20,
        "extra_channels_slice": (11, 17),
        "soil_sidecar": True,
        "description": (
            "spectral_xavier_climate + 4 MapBiomas Solo channels "
            "(clay/silt/sand % + SOC t/ha) from {code}_soil.npy sidecars; "
            "joined at transform time (17-ch .npy unchanged). DOY → PE. "
            "Use --soil-fusion early|late to pass soil through the transformer "
            "or only into the decoder after temporal pooling."
        ),
    },
}


def _register_recipe_layouts() -> None:
    """Add ma_* derived layouts (municipal-aggregate sweep); alias legacy mp_* names."""
    from .feature_recipes import recipe_layout_records

    for rec in recipe_layout_records():
        name = rec["name"]
        if name in _LAYOUT or name in _ALIASES:
            raise ValueError(f"recipe layout name collision: {name}")
        _LAYOUT[name] = {
            "input_dim": int(rec["input_dim"]),
            "extra_channels_slice": rec["extra_channels_slice"],
            "recipe": dict(rec["recipe"]),
            "description": rec["description"],
        }
        legacy = rec.get("legacy_name")
        if legacy and legacy != name:
            if legacy in _LAYOUT or legacy in _ALIASES:
                raise ValueError(f"legacy recipe alias collision: {legacy}")
            _ALIASES[legacy] = name


_ALIASES: dict[str, str] = {
    "s2": "spectral",
    "s2_only": "spectral",
    "xavier": "spectral_xavier",
    "xavier_rain": "spectral_xavier",
    "s2_xavier": "spectral_xavier",
    "xavier_climate": "spectral_xavier_climate",
    "s2_xavier_climate": "spectral_xavier_climate",
    "spectral_xavier_clim": "spectral_xavier_climate",
    "spectral_xavier_full": "spectral_xavier_climate",
    "xavier_full": "spectral_xavier_climate",
    "full": "spectral_xavier_climate",
    "xavier_climate_soil": "spectral_xavier_climate_soil",
    "s2_xavier_climate_soil": "spectral_xavier_climate_soil",
    "full_soil": "spectral_xavier_climate_soil",
    "spectral_xavier_full_soil": "spectral_xavier_climate_soil",
}


_register_recipe_layouts()


def feature_layout_choices() -> tuple[str, ...]:
    return tuple(sorted(_LAYOUT.keys()))


def feature_layout_cli_choices() -> tuple[str, ...]:
    """Canonical layout names plus CLI aliases (for argparse choices)."""
    return tuple(sorted(set(_LAYOUT) | set(_ALIASES)))


def normalize_feature_layout(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    key = _ALIASES.get(key, key)
    if key not in _LAYOUT:
        known = feature_layout_cli_choices()
        preview = ", ".join(known[:8])
        extra = f" (+{len(known) - 8} more)" if len(known) > 8 else ""
        raise ValueError(
            f"Unknown feature layout {name!r}. Choose one of: {preview}{extra}"
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


def feature_layout_needs_soil_sidecar(name: str) -> bool:
    """True when layout expects {code}_soil.npy joined at transform time."""
    return bool(resolve_feature_layout(name).get("soil_sidecar"))
