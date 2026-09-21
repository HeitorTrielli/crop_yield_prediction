"""Compare the temporal footprint of each season's .npy files.

If one harvest year carries no within-year yield signal, the usual cause is that
its imagery covers a different part of the crop calendar than the other seasons.
This prints, per season, when observations actually fall relative to DOY 1 =
Oct 1 and where the NDVI peak sits.

    python debug_season_alignment.py --datapath /home/emap/sits_moco_data/npy_muni_mean
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

DOY_CHANNEL = 10
NO_DATA = -9999
RED, NIR = 2, 6
# Season months 1..6 starting Oct 1; DOY here is season-relative (day 1 = Oct 1).
MONTH_EDGES = [(1, 31, "Oct"), (32, 61, "Nov"), (62, 92, "Dec"),
               (93, 123, "Jan"), (124, 151, "Feb"), (152, 182, "Mar")]


def season_dirs(root: Path) -> list[tuple[int, Path]]:
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        a, _, b = d.name.partition("-")
        if a.isdigit() and b.isdigit():
            out.append((int(b), d))
        elif d.name.isdigit() and len(d.name) == 4:
            out.append((int(d.name), d))
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datapath", required=True, type=Path)
    ap.add_argument("--per-season", type=int, default=60, help="municipalities sampled")
    ap.add_argument("--max-pixels", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for year, d in season_dirs(args.datapath):
        munis = [m for m in sorted(d.iterdir()) if m.is_dir()]
        if not munis:
            continue
        pick = rng.choice(len(munis), min(args.per_season, len(munis)), replace=False)
        for i in pick:
            m = munis[int(i)]
            f = m / f"{m.name}.npy"
            if not f.is_file():
                continue
            mm = np.load(f, mmap_mode="r")
            n = int(mm.shape[0])
            sel = slice(None) if n <= args.max_pixels else np.sort(
                rng.choice(n, args.max_pixels, replace=False)
            )
            a = np.asarray(mm[sel], dtype=np.float32)
            doy = a[:, :, DOY_CHANNEL].astype(int)
            spec = a[:, :, :10]
            bad = (spec == NO_DATA) | ~np.isfinite(spec)
            refl = np.where(bad, np.nan, spec) * 1e-4
            with np.errstate(invalid="ignore"):
                ndvi = (refl[:, :, NIR] - refl[:, :, RED]) / (
                    refl[:, :, NIR] + refl[:, :, RED] + 1e-8
                )
            nd = np.nanmean(ndvi, axis=0)  # (T,) municipality mean per date
            dv = doy[0]
            ok = np.isfinite(nd)
            if ok.sum() == 0:
                continue
            rec = {
                "year": year,
                "muni": m.name,
                "T": int(a.shape[1]),
                "doy_min": int(dv.min()),
                "doy_max": int(dv.max()),
                "doy_span": int(dv.max() - dv.min()),
                "doy_at_ndvi_peak": int(dv[ok][np.argmax(nd[ok])]),
                "ndvi_peak": float(np.nanmax(nd[ok])),
            }
            for lo, hi, name in MONTH_EDGES:
                rec[name] = int(((dv >= lo) & (dv <= hi)).sum())
            rows.append(rec)

    df = pl.DataFrame(rows)
    print(f"\nsampled {df.height} municipality-seasons from {args.datapath}\n")
    print("=== TEMPORAL FOOTPRINT PER SEASON (DOY 1 = Oct 1) ===")
    print(
        df.group_by("year")
        .agg(
            pl.len().alias("n"),
            pl.col("doy_min").median().alias("doy_min"),
            pl.col("doy_max").median().alias("doy_max"),
            pl.col("doy_span").median().alias("span"),
            pl.col("T").median().alias("T_med"),
            pl.col("doy_at_ndvi_peak").median().alias("peak_doy"),
            pl.col("ndvi_peak").mean().round(3).alias("ndvi_peak"),
        )
        .sort("year")
    )

    print("\n=== OBSERVATIONS PER SEASON MONTH (median per municipality) ===")
    print(
        df.group_by("year")
        .agg([pl.col(n).median().alias(n) for _, _, n in MONTH_EDGES])
        .sort("year")
    )
    print(
        "\nSoy in Parana is sown Oct-Nov and harvested Feb-Mar; the yield-bearing\n"
        "signal is the Jan-Feb pod-fill window. A season with few Jan/Feb frames,\n"
        "or an NDVI peak at a different DOY than its neighbours, is misaligned."
    )


if __name__ == "__main__":
    main()
