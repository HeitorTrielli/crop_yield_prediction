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


def generate_trials(search: dict) -> list[dict[str, Any]]:
    """
    Return a list of parameter dicts to merge onto study base config.

    search keys: strategy, n_trials, parameters, trials, seed
    """
    strategy = search["strategy"]
    parameters: dict = search.get("parameters") or {}

    if strategy == "list":
        trials = [dict(t) for t in (search.get("trials") or [])]
        if not trials:
            raise ValueError("list search requires a non-empty search.trials list")
        return _filter_invalid_trials(trials, search)

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
