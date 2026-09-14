#!/usr/bin/env python3
"""
Compute per-pixel temporal coverage from daily municipal TIFFs and filter survival.

Core logic lives in ``datasets.pixel_coverage`` (reusable for training / MoCo later).
This script adds survival tables, plots, and optional keep-mask export.

Examples:
  python scripts/compute_pixel_temporal_coverage.py --workers 6
  python scripts/compute_pixel_temporal_coverage.py --season 2022-2023 --limit-munis 10
  python scripts/compute_pixel_temporal_coverage.py --from-summary files/pixel_tiff_coverage_summary.json
  python scripts/compute_pixel_temporal_coverage.py --write-masks --min-valid-days 8 --min-season-months 4
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datasets.pixel_coverage import (  # noqa: E402
    MAX_DAYS_HIST,
    analyze_municipality_histograms,
    keep_mask_path,
    parse_exclude,
    season_directories,
    survival_from_day_hist,
    survival_from_joint,
    survival_from_month_hist,
)

MAX_DAYS = MAX_DAYS_HIST
DEFAULT_MIN_DAYS = 8
DEFAULT_MIN_MONTHS = 4


def _log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _worker(
    args: tuple[str, str, str, int, int, bool, str | None],
) -> dict:
    season, code, muni_dir_s, min_days, min_months, write_masks, mask_dir_s = args
    km = (
        keep_mask_path(mask_dir_s, season, code)
        if write_masks and mask_dir_s
        else None
    )
    return analyze_municipality_histograms(
        season,
        code,
        muni_dir_s,
        min_valid_days=min_days,
        min_season_months=min_months,
        keep_mask_path=km,
    )


def plot_results(
    out_dir: Path,
    *,
    day_hist: np.ndarray,
    month_hist: np.ndarray,
    joint: np.ndarray,
    seasons_day: dict[str, np.ndarray],
    n_kept: int,
) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Day-count histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(3, min(41, MAX_DAYS + 1))
    ax.bar(xs, day_hist[xs] / 1e6, width=0.9, color="#4C78A8")
    ax.set_xlabel("Valid images per pixel")
    ax.set_ylabel("Pixels (millions)")
    ax.set_title("Per-pixel valid-day count (kept pixels, days > 2)")
    ax.axvline(8, color="#E45756", ls="--", lw=1.2, label="min_days=8")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pixel_valid_days_hist.png", dpi=160)
    plt.close(fig)

    # 2) Month-span histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ms = np.arange(1, 7)
    ax.bar(ms, month_hist[ms] / 1e6, color="#54A24B")
    ax.set_xlabel("Distinct season months with ≥1 image (Oct–Mar)")
    ax.set_ylabel("Pixels (millions)")
    ax.set_title("Per-pixel season-month coverage")
    ax.axvline(3.5, color="#E45756", ls="--", lw=1.2, label="min_months=4")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pixel_season_months_hist.png", dpi=160)
    plt.close(fig)

    # 3) Survival vs min_days
    day_thresholds = list(range(3, 31))
    surv_d = [100.0 * survival_from_day_hist(day_hist, t) / n_kept for t in day_thresholds]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(day_thresholds, surv_d, color="#4C78A8", lw=2)
    ax.axvline(8, color="#E45756", ls="--", label="suggested min_days=8")
    ax.set_xlabel("Minimum valid days")
    ax.set_ylabel("Pixels kept (%)")
    ax.set_title("Filter survival vs min valid days")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "survival_min_valid_days.png", dpi=160)
    plt.close(fig)

    # 4) Survival vs min_months
    month_thresholds = list(range(1, 7))
    surv_m = [
        100.0 * survival_from_month_hist(month_hist, t) / n_kept for t in month_thresholds
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(month_thresholds, surv_m, color="#54A24B", lw=2, marker="o")
    ax.axvline(4, color="#E45756", ls="--", label="suggested min_months=4")
    ax.set_xlabel("Minimum season months")
    ax.set_ylabel("Pixels kept (%)")
    ax.set_title("Filter survival vs min season months")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "survival_min_season_months.png", dpi=160)
    plt.close(fig)

    # 5) Joint survival heatmap
    day_ts = [3, 5, 8, 10, 12, 15, 20]
    month_ts = [2, 3, 4, 5, 6]
    grid = np.zeros((len(month_ts), len(day_ts)), dtype=np.float64)
    for i, m in enumerate(month_ts):
        for j, d in enumerate(day_ts):
            grid[i, j] = 100.0 * survival_from_joint(joint, d, m) / n_kept
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=100)
    ax.set_xticks(range(len(day_ts)))
    ax.set_xticklabels([str(d) for d in day_ts])
    ax.set_yticks(range(len(month_ts)))
    ax.set_yticklabels([str(m) for m in month_ts])
    ax.set_xlabel("min_valid_days")
    ax.set_ylabel("min_season_months")
    ax.set_title("Joint filter survival (% pixels kept)")
    for i in range(len(month_ts)):
        for j in range(len(day_ts)):
            ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, ax=ax, label="% kept")
    fig.tight_layout()
    fig.savefig(out_dir / "survival_joint_heatmap.png", dpi=160)
    plt.close(fig)

    # 6) Mean-ish day hist by season (stacked comparison of CDF-ish survival)
    if seasons_day:
        fig, ax = plt.subplots(figsize=(8, 4))
        for season, dh in sorted(seasons_day.items()):
            nk = int(dh.sum())
            if nk == 0:
                continue
            surv = [100.0 * survival_from_day_hist(dh, t) / nk for t in day_thresholds]
            ax.plot(day_thresholds, surv, lw=1.8, label=season)
        ax.axvline(8, color="k", ls="--", lw=1)
        ax.set_xlabel("Minimum valid days")
        ax.set_ylabel("Pixels kept (%)")
        ax.set_title("Day-count filter survival by season")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "survival_min_valid_days_by_season.png", dpi=160)
        plt.close(fig)


def write_survival_tables(
    out_dir: Path,
    joint: np.ndarray,
    day_hist: np.ndarray,
    month_hist: np.ndarray,
    n_kept: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "survival_min_valid_days.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["min_valid_days", "n_kept", "pct_kept"])
        for t in range(3, 31):
            n = survival_from_day_hist(day_hist, t)
            w.writerow([t, n, round(100.0 * n / n_kept, 4)])

    with (out_dir / "survival_min_season_months.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(["min_season_months", "n_kept", "pct_kept"])
        for t in range(1, 7):
            n = survival_from_month_hist(month_hist, t)
            w.writerow([t, n, round(100.0 * n / n_kept, 4)])

    day_ts = [3, 5, 8, 10, 12, 15, 20]
    month_ts = [2, 3, 4, 5, 6]
    with (out_dir / "survival_joint.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["min_valid_days", "min_season_months", "n_kept", "pct_kept"])
        for d in day_ts:
            for m in month_ts:
                n = survival_from_joint(joint, d, m)
                w.writerow([d, m, n, round(100.0 * n / n_kept, 4)])


def compute_from_summary_json(path: Path, out_dir: Path) -> None:
    """Approximate day-count survival only (no joint / month hist) from prior summary."""
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = data["day_hist_labels"]
    counts = np.asarray(data["day_hist_all"], dtype=np.int64)
    # Rebuild a fine day_hist by placing coarse bins at bin starts (approx).
    # Better: expand using bin edges used in analyze_pixel_tiff_coverage.
    edges = [3, 5, 8, 11, 15, 20, 25, 30, 40, 10_000]
    day_hist = np.zeros(MAX_DAYS + 1, dtype=np.int64)
    for i, c in enumerate(counts):
        lo = edges[i]
        hi = min(edges[i + 1] - 1, MAX_DAYS)
        # Spread count uniformly across bin (approx for survival curves).
        width = max(hi - lo + 1, 1)
        each = c // width
        rem = c - each * width
        day_hist[lo : hi + 1] += each
        day_hist[lo] += rem

    n_kept = int(day_hist.sum())
    # Month hist unknown — synthesize from overall keys if present
    month_hist = np.zeros(7, dtype=np.int64)
    ov = data.get("overall", {})
    # We only know leq2, leq3, full6 — leave month hist empty-ish
    if ov:
        # cannot reconstruct exactly; skip month plots detail
        pass

    joint = np.zeros((MAX_DAYS + 1, 7), dtype=np.int64)
    # Put all mass in month=6 as placeholder? Better skip joint from summary.
    seasons_day = {}
    for season, s in data.get("seasons", {}).items():
        dh = np.zeros(MAX_DAYS + 1, dtype=np.int64)
        for i, c in enumerate(s.get("day_hist", [])):
            lo = edges[i]
            hi = min(edges[i + 1] - 1, MAX_DAYS)
            width = max(hi - lo + 1, 1)
            each = int(c) // width
            rem = int(c) - each * width
            dh[lo : hi + 1] += each
            dh[lo] += rem
        seasons_day[season] = dh

    out_dir.mkdir(parents=True, exist_ok=True)
    # Day survival table is reliable enough with uniform-within-bin approx
    with (out_dir / "survival_min_valid_days_from_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(["min_valid_days", "n_kept_approx", "pct_kept_approx", "note"])
        for t in range(3, 31):
            # Prefer exact from coarse bins when threshold aligns with edges
            n = survival_from_day_hist(day_hist, t)
            w.writerow([t, n, round(100.0 * n / n_kept, 4), "approx from coarse bins"])

    # Exact survival at bin edges from original coarse hist
    with (out_dir / "survival_min_valid_days_exact_bins.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(["min_valid_days", "n_kept", "pct_kept"])
        total = int(sum(counts))
        cum_drop = 0
        # labels correspond to edges bins
        thresholds_exact = [3, 5, 8, 11, 15, 20, 25, 30, 40]
        for i, t in enumerate(thresholds_exact):
            n = total - cum_drop
            w.writerow([t, n, round(100.0 * n / total, 4)])
            if i < len(counts):
                cum_drop += int(counts[i])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    th = [3, 5, 8, 11, 15, 20, 25, 30, 40]
    surv = []
    cum_drop = 0
    total = int(sum(counts))
    for i, t in enumerate(th):
        surv.append(100.0 * (total - cum_drop) / total)
        if i < len(counts):
            cum_drop += int(counts[i])
    ax.plot(th, surv, marker="o", color="#4C78A8", lw=2)
    ax.axvline(8, color="#E45756", ls="--", label="min_days=8")
    ax.set_xlabel("Minimum valid days")
    ax.set_ylabel("Pixels kept (%)")
    ax.set_title("Filter survival vs min valid days (from summary bins)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "survival_min_valid_days.png", dpi=160)
    plt.close(fig)

    # Coarse hist plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, np.asarray(counts) / 1e6, color="#4C78A8")
    ax.set_xlabel("Valid images per pixel (binned)")
    ax.set_ylabel("Pixels (millions)")
    ax.set_title("Per-pixel valid-day count")
    fig.tight_layout()
    fig.savefig(out_dir / "pixel_valid_days_hist.png", dpi=160)
    plt.close(fig)

    summary = {
        "mode": "from_summary",
        "source": str(path),
        "n_kept": total,
        "bin_labels": labels,
        "bin_counts": [int(x) for x in counts],
        "survival_at_min_days_8": round(100.0 * (total - int(counts[0]) - int(counts[1])) / total, 4),
        "note": (
            "Day survival at bin edges is exact. Joint day×month filters require a "
            "full scan (omit --from-summary)."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log(f"Wrote day-count survival from summary -> {out_dir}")
    _log(
        f"  kept={total:,}; survive min_days>=8 ~ "
        f"{summary['survival_at_min_days_8']}% "
        f"(drops 3-4 and 5-7 bins)"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("files/daily_tiff"))
    p.add_argument(
        "--exclude",
        nargs="*",
        default=["2018-2019"],
        help="Season folders to skip (default: 2018-2019)",
    )
    p.add_argument("--season", type=str, default=None)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit-munis", type=int, default=0)
    p.add_argument("--min-valid-days", type=int, default=DEFAULT_MIN_DAYS)
    p.add_argument("--min-season-months", type=int, default=DEFAULT_MIN_MONTHS)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/pixel_temporal_coverage"),
    )
    p.add_argument(
        "--from-summary",
        type=Path,
        default=None,
        help="Skip TIFF scan; build day-count survival from analyze_pixel_tiff_coverage JSON",
    )
    p.add_argument(
        "--write-masks",
        action="store_true",
        help="Save {out_dir}/masks/{season}/{code}_keep.npy bool masks",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_summary is not None:
        compute_from_summary_json(args.from_summary, out_dir)
        return

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

    mask_dir = str(out_dir / "masks") if args.write_masks else None
    jobs: list[tuple] = []
    for sd in seasons:
        munis = sorted(p for p in sd.iterdir() if p.is_dir())
        if args.limit_munis > 0:
            munis = munis[: args.limit_munis]
        for m in munis:
            jobs.append(
                (
                    sd.name,
                    m.name,
                    str(m),
                    args.min_valid_days,
                    args.min_season_months,
                    args.write_masks,
                    mask_dir,
                )
            )

    _log(
        f"Scanning {len(jobs)} muni-seasons | min_days>={args.min_valid_days} "
        f"min_months>={args.min_season_months} | workers={args.workers}"
    )
    t0 = time.time()
    rows: list[dict] = []

    if args.workers <= 1:
        for i, job in enumerate(jobs, 1):
            rows.append(_worker(job))
            if i % 25 == 0 or i == len(jobs):
                _log(f"  {i}/{len(jobs)} ({time.time() - t0:.0f}s)")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_worker, job): job for job in jobs}
            done = 0
            for fut in as_completed(futs):
                rows.append(fut.result())
                done += 1
                if done % 25 == 0 or done == len(jobs):
                    _log(f"  {done}/{len(jobs)} ({time.time() - t0:.0f}s)")

    day_hist = np.zeros(MAX_DAYS + 1, dtype=np.int64)
    month_hist = np.zeros(7, dtype=np.int64)
    joint = np.zeros((MAX_DAYS + 1, 7), dtype=np.int64)
    seasons_day: dict[str, np.ndarray] = {}
    n_kept = 0
    n_pass = 0
    n_early = 0
    n_late = 0

    muni_csv = out_dir / "by_municipality.csv"
    with muni_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "season",
                "code",
                "n_tiffs",
                "n_kept",
                "n_pass",
                "pct_pass",
                "mean_days",
                "median_days",
                "early_only",
                "late_only",
                "error",
            ],
        )
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["season"], x["code"])):
            if r.get("error"):
                w.writerow(
                    {
                        "season": r["season"],
                        "code": r["code"],
                        "n_tiffs": r["n_tiffs"],
                        "n_kept": 0,
                        "n_pass": 0,
                        "pct_pass": 0,
                        "mean_days": "",
                        "median_days": "",
                        "early_only": 0,
                        "late_only": 0,
                        "error": r["error"],
                    }
                )
                continue
            if r["empty"] or r["n_kept"] == 0:
                w.writerow(
                    {
                        "season": r["season"],
                        "code": r["code"],
                        "n_tiffs": r["n_tiffs"],
                        "n_kept": 0,
                        "n_pass": 0,
                        "pct_pass": 0,
                        "mean_days": "",
                        "median_days": "",
                        "early_only": 0,
                        "late_only": 0,
                        "error": "",
                    }
                )
                continue

            day_hist += r["day_hist"]
            month_hist += r["month_hist"]
            joint += r["joint_hist"]
            seasons_day.setdefault(
                r["season"], np.zeros(MAX_DAYS + 1, dtype=np.int64)
            )
            seasons_day[r["season"]] += r["day_hist"]
            n_kept += r["n_kept"]
            n_pass += r["n_pass"]
            n_early += r["early_only"]
            n_late += r["late_only"]
            w.writerow(
                {
                    "season": r["season"],
                    "code": r["code"],
                    "n_tiffs": r["n_tiffs"],
                    "n_kept": r["n_kept"],
                    "n_pass": r["n_pass"],
                    "pct_pass": round(100.0 * r["n_pass"] / r["n_kept"], 4),
                    "mean_days": round(r.get("mean_days", 0), 3),
                    "median_days": r.get("median_days", ""),
                    "early_only": r["early_only"],
                    "late_only": r["late_only"],
                    "error": "",
                }
            )

    if n_kept == 0:
        raise SystemExit("No kept pixels found")

    write_survival_tables(out_dir, joint, day_hist, month_hist, n_kept)
    plot_results(
        out_dir,
        day_hist=day_hist,
        month_hist=month_hist,
        joint=joint,
        seasons_day=seasons_day,
        n_kept=n_kept,
    )

    pct_pass = 100.0 * n_pass / n_kept
    summary = {
        "meta": {
            "root": str(root.resolve()),
            "exclude": sorted(exclude),
            "seasons": [s.name for s in seasons],
            "min_valid_days": args.min_valid_days,
            "min_season_months": args.min_season_months,
            "elapsed_s": round(time.time() - t0, 1),
            "n_jobs": len(jobs),
        },
        "n_kept_pixels": int(n_kept),
        "n_pass_filter": int(n_pass),
        "pct_pass_filter": round(pct_pass, 4),
        "pct_early_only": round(100.0 * n_early / n_kept, 4),
        "pct_late_only": round(100.0 * n_late / n_kept, 4),
        "day_hist": day_hist.tolist(),
        "month_hist": month_hist.tolist(),
        "survival_examples": {
            f"days>={d}_months>={m}": {
                "n": survival_from_joint(joint, d, m),
                "pct": round(100.0 * survival_from_joint(joint, d, m) / n_kept, 4),
            }
            for d, m in ((8, 1), (8, 4), (10, 4), (12, 4), (8, 5))
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _log(f"Wrote results -> {out_dir}")
    _log(
        f"OVERALL kept={n_kept:,} | "
        f"pass(days>={args.min_valid_days}, months>={args.min_season_months})="
        f"{n_pass:,} ({pct_pass:.2f}%)"
    )
    for key, val in summary["survival_examples"].items():
        _log(f"  {key}: {val['pct']}%")


if __name__ == "__main__":
    main()
