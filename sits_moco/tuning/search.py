"""Trial sampling for grid and random search."""

from __future__ import annotations

import itertools
import random
from typing import Any, Iterator

import numpy as np


def _sample_one(spec: dict, rng: random.Random) -> Any:
    ptype = spec.get("type", "choice")
    if ptype == "choice":
        return rng.choice(spec["values"])
    if ptype == "int_uniform":
        return rng.randint(int(spec["low"]), int(spec["high"]))
    if ptype == "float_uniform":
        low, high = float(spec["low"]), float(spec["high"])
        return rng.uniform(low, high)
    if ptype == "log_uniform":
        low, high = float(spec["low"]), float(spec["high"])
        if low <= 0 or high <= 0:
            raise ValueError("log_uniform requires positive low and high")
        return 10 ** rng.uniform(np.log10(low), np.log10(high))
    raise ValueError(f"Unknown parameter type {ptype!r}")


def _grid_values(spec: dict) -> list[Any]:
    ptype = spec.get("type", "choice")
    if ptype == "choice":
        return list(spec["values"])
    if ptype == "int_uniform":
        return list(range(int(spec["low"]), int(spec["high"]) + 1))
    if ptype in ("float_uniform", "log_uniform"):
        if "values" in spec:
            return list(spec["values"])
        n = int(spec.get("n_grid", 3))
        low, high = float(spec["low"]), float(spec["high"])
        if ptype == "log_uniform":
            if low <= 0 or high <= 0:
                raise ValueError("log_uniform grid requires positive bounds")
            pts = np.logspace(np.log10(low), np.log10(high), num=n)
        else:
            pts = np.linspace(low, high, num=n)
        return [float(x) for x in pts]
    raise ValueError(f"Unsupported grid parameter type {ptype!r}")


def _as_year_list(value: Any) -> list[int]:
    if value is None or value is False:
        return []
    if isinstance(value, (int, str)):
        return [int(value)]
    return [int(y) for y in value]


def _year_loo_folds_for_layout(search: dict, *, needs_climate: bool) -> list[dict[str, Any]]:
    years = _as_year_list(search.get("year_loo"))
    skip_missing_climate = set(
        _as_year_list(search.get("skip_holdout_years_missing_climate"))
    )
    scaler_template = str(
        search.get("extra_scaler_template")
        or "files/train_input_scaler_loo{holdout_year}.json"
    )
    skip_reason = (
        "npy for this holdout year is rain-only (C=13); layout needs climate extras"
    )
    folds: list[dict[str, Any]] = []
    for year in years:
        fold: dict[str, Any] = {
            "holdout_year": year,
            "extra_scaler": scaler_template.format(holdout_year=year, year=year),
        }
        if needs_climate and year in skip_missing_climate:
            fold["skip"] = True
            fold["skip_reason"] = skip_reason
        folds.append(fold)
    return folds


def _feature_sweep_trials(search: dict) -> list[dict[str, Any]]:
    """One trial per mp_* layout. ``year_loo`` nests holdout years inside the trial."""
    from datasets.feature_recipes import recipe_layout_records

    records = recipe_layout_records()
    years = _as_year_list(search.get("year_loo"))
    if not years:
        return [{"feature_layout": rec["name"]} for rec in records]

    trials: list[dict[str, Any]] = []
    for rec in records:
        trials.append(
            {
                "feature_layout": rec["name"],
                "year_loo_folds": _year_loo_folds_for_layout(
                    search,
                    needs_climate=rec.get("extra_channels_slice") is not None,
                ),
            }
        )
    return trials


def generate_trials(search: dict) -> list[dict[str, Any]]:
    """
    Return a list of parameter dicts to merge onto study base config.

    search keys: strategy, n_trials, parameters, trials, seed,
    year_loo (feature_sweep only)
    """
    strategy = search["strategy"]
    parameters: dict = search.get("parameters") or {}

    if strategy == "list":
        trials = [dict(t) for t in (search.get("trials") or [])]
        if not trials:
            raise ValueError("list search requires a non-empty search.trials list")
        return _filter_invalid_trials(trials, search)

    if strategy == "feature_sweep":
        return _filter_invalid_trials(_feature_sweep_trials(search), search)

    if not parameters:
        return [{}]

    if strategy == "grid":
        keys = sorted(parameters.keys())
        value_lists = [_grid_values(parameters[k]) for k in keys]
        trials = []
        for combo in itertools.product(*value_lists):
            trials.append(dict(zip(keys, combo)))
        return _filter_invalid_trials(trials, search)

    if strategy == "random":
        n_trials = int(search["n_trials"])
        rng = random.Random(int(search.get("seed", 42)))
        trials = []
        seen: set[tuple] = set()
        max_attempts = n_trials * 100
        attempts = 0
        while len(trials) < n_trials and attempts < max_attempts:
            attempts += 1
            sample = {k: _sample_one(spec, rng) for k, spec in parameters.items()}
            if not _trial_is_valid(sample, search):
                continue
            key = tuple(sorted((k, repr(v)) for k, v in sample.items()))
            if key in seen:
                continue
            seen.add(key)
            trials.append(sample)
        if len(trials) < n_trials:
            raise RuntimeError(
                f"Could only sample {len(trials)} unique random trials "
                f"(requested {n_trials})"
            )
        return trials

    raise ValueError(f"Unknown strategy {strategy!r}")


def _trial_is_valid(sample: dict, search: dict) -> bool:
    """Drop invalid architecture combos (e.g. n_head must divide d_model)."""
    d_model = sample.get("model_d_model")
    if d_model is None:
        d_model = (search.get("base") or {}).get("model_d_model")
    n_head = sample.get("model_n_head")
    if d_model is not None and n_head is not None:
        d_model = int(d_model)
        n_head = int(n_head)
        if d_model % n_head != 0:
            return False
    return True


def _filter_invalid_trials(trials: list[dict], search: dict) -> list[dict]:
    valid = [t for t in trials if _trial_is_valid(t, search)]
    dropped = len(trials) - len(valid)
    if dropped:
        print(f"  (dropped {dropped} invalid architecture combo(s), e.g. d_model % n_head != 0)")
    return valid


def iter_trials(search: dict) -> Iterator[dict[str, Any]]:
    yield from generate_trials(search)
