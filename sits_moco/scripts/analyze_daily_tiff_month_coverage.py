#!/usr/bin/env python3
"""
Municipality-level temporal coverage from daily_tiff filenames only (no raster I/O).

Layout: files/daily_tiff/<season>/<code>/{code}_YYYY_MM_DD.tiff
Season months for PR soy: Oct..Mar (calendar 10,11,12,1,2,3).

Usage:
  python scripts/analyze_daily_tiff_month_coverage.py
  python scripts/analyze_daily_tiff_month_coverage.py --root files/daily_tiff --json out.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

SEASON_MONTHS = [10, 11, 12, 1, 2, 3]
EARLY = {10, 11, 12}
LATE = {1, 2, 3}
MONTH_LABEL = {
    10: "Oct",
    11: "Nov",
    12: "Dec",
    1: "Jan",
    2: "Feb",
    3: "Mar",
}
FNAME_RE = re.compile(r"^(\d+)_(\d{4})_(\d{2})_(\d{2})\.")


def _pct(sorted_vals: list[int], p: float) -> int | None:
    if not sorted_vals:
        return None
    i = int(round((p / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[i]


def season_month_key(months: set[int]) -> tuple[int, ...]:
    return tuple(sorted(months & set(SEASON_MONTHS), key=lambda m: (0 if m >= 10 else 1, m)))


def analyze_season(season_dir: Path) -> dict:
    munis = sorted(p for p in season_dir.iterdir() if p.is_dir())
    cats: Counter[str] = Counter()
    n_months_hist: Counter[int] = Counter()
    n_imgs: list[int] = []
    per_month_img: Counter[int] = Counter()
    month_present: Counter[int] = Counter()
    month_only: Counter[int] = Counter()
    month_combo: Counter[tuple[int, ...]] = Counter()
    early_only: list[dict] = []
    late_only: list[dict] = []
    sparse: list[dict] = []
    no_oct_nov: list[dict] = []
    empty: list[str] = []

    for mdir in munis:
        months: set[int] = set()
        month_counts: Counter[int] = Counter()
        n = 0
        for f in list(mdir.glob("*.tiff")) + list(mdir.glob("*.tif")):
            mo = FNAME_RE.match(f.name)
            if not mo:
                continue
            m = int(mo.group(3))
            months.add(m)
            month_counts[m] += 1
            per_month_img[m] += 1
            n += 1

        code = mdir.name
        if n == 0:
            cats["empty"] += 1
            empty.append(code)
            continue

        n_imgs.append(n)
        sm = set(season_month_key(months))
        n_months_hist[len(sm)] += 1
        for m in sm:
            month_present[m] += 1
        month_combo[season_month_key(months)] += 1
        if len(sm) == 1:
            month_only[next(iter(sm))] += 1

        has_early = bool(sm & EARLY)
        has_late = bool(sm & LATE)
        row = {
            "code": code,
            "n_images": n,
            "months": [MONTH_LABEL[m] for m in season_month_key(months)],
            "month_counts": {MONTH_LABEL[m]: month_counts[m] for m in season_month_key(months)},
        }

        if has_early and not has_late:
            cats["early_only"] += 1
            early_only.append(row)
        elif has_late and not has_early:
            cats["late_only"] += 1
            late_only.append(row)
        elif has_early and has_late:
            cats["both_halves"] += 1
        else:
            cats["other"] += 1

        if len(sm) <= 2:
            cats["leq_2_months"] += 1
            sparse.append(row)
        if len(sm) <= 3:
            cats["leq_3_months"] += 1
        if 1 not in sm:
            cats["missing_jan"] += 1
        if not (sm & {10, 11}):
            cats["no_oct_nov"] += 1
            no_oct_nov.append(row)
        if sm == set(SEASON_MONTHS):
            cats["full_6_months"] += 1
        # first-3-only / last-3-only in the stronger sense (only those months)
        if sm and sm.issubset(EARLY):
            cats["subset_oct_dec_only"] += 1
        if sm and sm.issubset(LATE):
            cats["subset_jan_mar_only"] += 1

    n_imgs_sorted = sorted(n_imgs)
    return {
        "season": season_dir.name,
        "n_munis": len(munis),
        "n_with_imgs": len(n_imgs),
        "total_imgs": int(sum(n_imgs)),
        "cats": dict(cats),
        "n_months_hist": {str(k): v for k, v in sorted(n_months_hist.items())},
        "munis_with_month": {MONTH_LABEL[m]: month_present[m] for m in SEASON_MONTHS},
        "munis_only_month": {MONTH_LABEL[m]: month_only[m] for m in SEASON_MONTHS if month_only[m]},
        "imgs_per_month": {MONTH_LABEL[m]: per_month_img[m] for m in SEASON_MONTHS},
        "img_count": {
            "min": n_imgs_sorted[0] if n_imgs_sorted else 0,
            "p10": _pct(n_imgs_sorted, 10),
            "p25": _pct(n_imgs_sorted, 25),
            "p50": _pct(n_imgs_sorted, 50),
            "p75": _pct(n_imgs_sorted, 75),
            "p90": _pct(n_imgs_sorted, 90),
            "max": n_imgs_sorted[-1] if n_imgs_sorted else 0,
            "mean": round(sum(n_imgs) / len(n_imgs), 1) if n_imgs else 0,
        },
        "top_combos": [
            {"months": [MONTH_LABEL[m] for m in k], "n": v}
            for k, v in sorted(month_combo.items(), key=lambda x: -x[1])[:15]
        ],
        "early_only": early_only,
        "late_only": late_only,
        "sparse_leq_2_months": sparse,
        "no_oct_nov": no_oct_nov,
        "empty": empty,
    }


def print_report(s: dict) -> None:
    print("=" * 70)
    print(s["season"])
    print(
        f"  munis={s['n_munis']} with_imgs={s['n_with_imgs']} "
        f"empty={s['cats'].get('empty', 0)} total_imgs={s['total_imgs']}"
    )
    print(f"  categories: {s['cats']}")
    print(f"  #season-months covered: {s['n_months_hist']}")
    ic = s["img_count"]
    print(
        f"  imgs/muni: min={ic['min']} p10={ic['p10']} p50={ic['p50']} "
        f"p90={ic['p90']} max={ic['max']} mean={ic['mean']}"
    )
    print(f"  munis with >=1 img in month: {s['munis_with_month']}")
    print(f"  total imgs by month: {s['imgs_per_month']}")
    if s["munis_only_month"]:
        print(f"  munis with ONLY that single month: {s['munis_only_month']}")
    print("  top month combos:")
    for row in s["top_combos"][:8]:
        print(f"    {row['n']:4d}  {'+'.join(row['months'])}")
    if s["early_only"]:
        print(f"  early_only (Oct-Dec, no Jan-Mar): {len(s['early_only'])}")
        for r in s["early_only"][:5]:
            print(f"    {r['code']} n={r['n_images']} months={r['months']}")
    if s["late_only"]:
        print(f"  late_only (Jan-Mar, no Oct-Dec): {len(s['late_only'])}")
        for r in s["late_only"][:5]:
            print(f"    {r['code']} n={r['n_images']} months={r['months']}")
    if s["sparse_leq_2_months"]:
        print(f"  sparse (<=2 months): {len(s['sparse_leq_2_months'])}")
        for r in s["sparse_leq_2_months"][:5]:
            print(f"    {r['code']} n={r['n_images']} months={r['months']}")
    n_no = len(s["no_oct_nov"])
    if n_no:
        print(f"  no Oct/Nov: {n_no}")
        for r in s["no_oct_nov"][:5]:
            print(f"    {r['code']} n={r['n_images']} months={r['months']}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("files/daily_tiff"))
    p.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to write full JSON summary",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    seasons = sorted(
        p for p in root.iterdir() if p.is_dir() and re.match(r"^\d{4}-\d{4}$", p.name)
    )
    if not seasons:
        raise SystemExit(f"No season folders under {root}")

    all_stats = {}
    for sd in seasons:
        s = analyze_season(sd)
        all_stats[sd.name] = s
        print_report(s)

    print("=" * 70)
    print("CROSS-SEASON HIGHLIGHTS")
    for name, s in all_stats.items():
        c = s["cats"]
        print(
            f"  {name}: full6={c.get('full_6_months', 0)} "
            f"early_only={c.get('early_only', 0)} "
            f"late_only={c.get('late_only', 0)} "
            f"leq3mo={c.get('leq_3_months', 0)} "
            f"no_oct_nov={c.get('no_oct_nov', 0)} "
            f"empty={c.get('empty', 0)}"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
