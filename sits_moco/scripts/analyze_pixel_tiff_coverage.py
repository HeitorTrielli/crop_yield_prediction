#!/usr/bin/env python3
"""
Per-pixel temporal coverage from daily municipal TIFFs (band-1 only, no full stack).

Core logic lives in ``datasets.pixel_coverage`` (reusable for training / MoCo later).
This script is the CLI for batch QA and JSON/CSV export.

Usage:
  python scripts/analyze_pixel_tiff_coverage.py
  python scripts/analyze_pixel_tiff_coverage.py --exclude 2018-2019 --workers 6
  python scripts/analyze_pixel_tiff_coverage.py --limit-munis 20  # smoke test
  python scripts/analyze_pixel_tiff_coverage.py --season 2022-2023
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datasets.pixel_coverage import (  # noqa: E402
    DAY_BIN_LABELS,
    MONTH_LABELS,
    MuniPixelStats,
    analyze_municipality,
    parse_exclude,
    season_directories,
)


def _log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _worker(args: tuple[str, str, str]) -> MuniPixelStats:
    season, code, muni_dir_str = args
    return analyze_municipality(season, code, muni_dir_str)


def aggregate(rows: list[MuniPixelStats]) -> dict:
    by_season: dict[str, list[MuniPixelStats]] = {}
    for r in rows:
        by_season.setdefault(r.season, []).append(r)

    seasons_out = {}
    for season, items in sorted(by_season.items()):
        day_hist = np.zeros(len(DAY_BIN_LABELS), dtype=np.int64)
        month_present = Counter()
        totals = Counter()
        worst_early = []
        worst_late = []
        worst_sparse = []
        errors = []
        n_munis = len(items)
        n_empty = 0

        for r in items:
            if r.error:
                errors.append({"code": r.code, "error": r.error})
            if r.empty or r.n_kept == 0:
                n_empty += 1
                continue
            day_hist += np.asarray(r.day_hist, dtype=np.int64)
            for k, v in r.month_present_pixels.items():
                month_present[k] += v
            totals["n_kept"] += r.n_kept
            totals["n_early_only"] += r.n_early_only
            totals["n_late_only"] += r.n_late_only
            totals["n_both"] += r.n_both
            totals["n_leq_2_months"] += r.n_leq_2_months
            totals["n_leq_3_months"] += r.n_leq_3_months
            totals["n_full_6"] += r.n_full_6
            totals["n_no_oct_nov"] += r.n_no_oct_nov
            totals["sum_mean_days_weighted"] += r.mean_valid_days * r.n_kept

            if r.n_kept > 0:
                frac_early = r.n_early_only / r.n_kept
                frac_late = r.n_late_only / r.n_kept
                frac_sparse = r.n_leq_2_months / r.n_kept
                if frac_early >= 0.05:
                    worst_early.append(
                        {
                            "code": r.code,
                            "n_kept": r.n_kept,
                            "n_early_only": r.n_early_only,
                            "frac": round(frac_early, 4),
                            "median_days": r.median_valid_days,
                        }
                    )
                if frac_late >= 0.05:
                    worst_late.append(
                        {
                            "code": r.code,
                            "n_kept": r.n_kept,
                            "n_late_only": r.n_late_only,
                            "frac": round(frac_late, 4),
                            "median_days": r.median_valid_days,
                        }
                    )
                if frac_sparse >= 0.05:
                    worst_sparse.append(
                        {
                            "code": r.code,
                            "n_kept": r.n_kept,
                            "n_leq_2_months": r.n_leq_2_months,
                            "frac": round(frac_sparse, 4),
                            "median_days": r.median_valid_days,
                        }
                    )

        n_kept = int(totals["n_kept"])
        def pct(k: str) -> float:
            return round(100.0 * totals[k] / n_kept, 3) if n_kept else 0.0

        seasons_out[season] = {
            "n_munis": n_munis,
            "n_empty_or_no_kept": n_empty,
            "n_kept_pixels": n_kept,
            "pct_early_only": pct("n_early_only"),
            "pct_late_only": pct("n_late_only"),
            "pct_both_halves": pct("n_both"),
            "pct_leq_2_months": pct("n_leq_2_months"),
            "pct_leq_3_months": pct("n_leq_3_months"),
            "pct_full_6_months": pct("n_full_6"),
            "pct_no_oct_nov": pct("n_no_oct_nov"),
            "n_early_only": int(totals["n_early_only"]),
            "n_late_only": int(totals["n_late_only"]),
            "n_both": int(totals["n_both"]),
            "n_leq_2_months": int(totals["n_leq_2_months"]),
            "n_leq_3_months": int(totals["n_leq_3_months"]),
            "n_full_6": int(totals["n_full_6"]),
            "n_no_oct_nov": int(totals["n_no_oct_nov"]),
            "mean_valid_days": round(totals["sum_mean_days_weighted"] / n_kept, 2)
            if n_kept
            else 0.0,
            "day_hist_labels": DAY_BIN_LABELS,
            "day_hist": day_hist.astype(int).tolist(),
            "month_present_pixels": {k: int(month_present[k]) for _, k in MONTH_LABELS},
            "worst_early_munis": sorted(worst_early, key=lambda x: -x["frac"])[:15],
            "worst_late_munis": sorted(worst_late, key=lambda x: -x["frac"])[:15],
            "worst_sparse_munis": sorted(worst_sparse, key=lambda x: -x["frac"])[:15],
            "errors": errors[:20],
            "n_errors": len(errors),
        }

    overall_kept = sum(s["n_kept_pixels"] for s in seasons_out.values())
    overall = {
        "n_kept_pixels": overall_kept,
        "n_early_only": sum(s["n_early_only"] for s in seasons_out.values()),
        "n_late_only": sum(s["n_late_only"] for s in seasons_out.values()),
        "n_both": sum(s["n_both"] for s in seasons_out.values()),
        "n_leq_2_months": sum(s["n_leq_2_months"] for s in seasons_out.values()),
        "n_leq_3_months": sum(s["n_leq_3_months"] for s in seasons_out.values()),
        "n_full_6": sum(s["n_full_6"] for s in seasons_out.values()),
        "n_no_oct_nov": sum(s["n_no_oct_nov"] for s in seasons_out.values()),
    }
    if overall_kept:
        for k in (
            "n_early_only",
            "n_late_only",
            "n_both",
            "n_leq_2_months",
            "n_leq_3_months",
            "n_full_6",
            "n_no_oct_nov",
        ):
            overall[f"pct_{k[2:]}"] = round(100.0 * overall[k] / overall_kept, 3)

    day_hist_all = np.zeros(len(DAY_BIN_LABELS), dtype=np.int64)
    for s in seasons_out.values():
        day_hist_all += np.asarray(s["day_hist"], dtype=np.int64)

    return {
        "day_hist_labels": DAY_BIN_LABELS,
        "day_hist_all": day_hist_all.astype(int).tolist(),
        "overall": overall,
        "seasons": seasons_out,
        "muni_rows": [asdict(r) for r in rows],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("files/daily_tiff"))
    p.add_argument(
        "--exclude",
        nargs="*",
        default=["2018-2019"],
        help="Season folders to skip (default: 2018-2019)",
    )
    p.add_argument("--season", type=str, default=None, help="Only this season")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--limit-munis",
        type=int,
        default=0,
        help="If >0, only first N munis per season (smoke test)",
    )
    p.add_argument(
        "--json",
        type=Path,
        default=Path("files/pixel_tiff_coverage_summary.json"),
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("files/pixel_tiff_coverage_by_muni.csv"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    exclude = parse_exclude(args.exclude or [])
    seasons = season_directories(root)
    if args.season:
        seasons = [p for p in seasons if p.name == args.season]
    seasons = [p for p in seasons if p.name not in exclude]
    if not seasons:
        raise SystemExit("No seasons left to scan")

    jobs: list[tuple[str, str, str]] = []
    for sd in seasons:
        munis = sorted(p for p in sd.iterdir() if p.is_dir())
        if args.limit_munis > 0:
            munis = munis[: args.limit_munis]
        for m in munis:
            jobs.append((sd.name, m.name, str(m)))

    _log(
        f"Scanning {len(jobs)} municipality-seasons across "
        f"{[s.name for s in seasons]} with {args.workers} workers"
    )
    t0 = time.time()
    rows: list[MuniPixelStats] = []

    if args.workers <= 1:
        for i, job in enumerate(jobs, 1):
            rows.append(_worker(job))
            if i % 25 == 0 or i == len(jobs):
                _log(f"  {i}/{len(jobs)} done ({time.time() - t0:.0f}s)")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_worker, job): job for job in jobs}
            done = 0
            for fut in as_completed(futs):
                rows.append(fut.result())
                done += 1
                if done % 25 == 0 or done == len(jobs):
                    _log(f"  {done}/{len(jobs)} done ({time.time() - t0:.0f}s)")

    rows.sort(key=lambda r: (r.season, r.code))
    summary = aggregate(rows)
    summary["meta"] = {
        "root": str(root.resolve()),
        "exclude": sorted(exclude),
        "seasons": [s.name for s in seasons],
        "n_jobs": len(jobs),
        "workers": args.workers,
        "elapsed_s": round(time.time() - t0, 1),
        "note": (
            "Per-pixel validity from daily TIFF band 1; kept if valid days > 2 "
            "(same as preprocess_daily_to_npy). Early=Oct-Dec, Late=Jan-Mar."
        ),
    }

    # Slim JSON for viz (drop huge muni_rows detail from main summary optional)
    out_slim = {
        "meta": summary["meta"],
        "day_hist_labels": summary["day_hist_labels"],
        "day_hist_all": summary["day_hist_all"],
        "overall": summary["overall"],
        "seasons": {
            k: {kk: vv for kk, vv in v.items() if kk != "errors"}
            for k, v in summary["seasons"].items()
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out_slim, indent=2), encoding="utf-8")
    _log(f"Wrote {args.json}")

    # CSV per muni
    import csv

    fields = [
        "season",
        "code",
        "n_tiffs",
        "n_kept",
        "n_early_only",
        "n_late_only",
        "n_both",
        "n_leq_2_months",
        "n_leq_3_months",
        "n_full_6",
        "n_no_oct_nov",
        "mean_valid_days",
        "median_valid_days",
        "p10_valid_days",
        "p90_valid_days",
        "max_valid_days",
        "empty",
        "error",
    ]
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: asdict(r).get(k) for k in fields})
    _log(f"Wrote {args.csv}")

    # Console highlight
    print()
    ov = summary["overall"]
    _log(
        f"OVERALL kept_pixels={ov['n_kept_pixels']:,} "
        f"early_only={ov['n_early_only']:,} ({ov.get('pct_early_only', 0)}%) "
        f"late_only={ov['n_late_only']:,} ({ov.get('pct_late_only', 0)}%) "
        f"leq2mo={ov['n_leq_2_months']:,} ({ov.get('pct_leq_2_months', 0)}%) "
        f"full6={ov['n_full_6']:,} ({ov.get('pct_full_6', 0)}%)"
    )
    for season, s in summary["seasons"].items():
        _log(
            f"  {season}: kept={s['n_kept_pixels']:,} "
            f"early={s['pct_early_only']}% late={s['pct_late_only']}% "
            f"leq2={s['pct_leq_2_months']}% full6={s['pct_full_6_months']}% "
            f"mean_days={s['mean_valid_days']}"
        )


if __name__ == "__main__":
    main()
