#!/usr/bin/env python3
"""Compare zonal CSV vs npy_muni_mean / pixel-mean for one muni/season."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]


def season_doy(d: date, season_start: date) -> int:
    return (d - season_start).days + 1


def compare(label: str, series: np.ndarray, z: pd.DataFrame, season_start: date) -> None:
    """series: [T, C] municipal mean time series."""
    npy_doy = series[:, 10]
    # Map season_doy -> timestep index (first match)
    doy_to_i: dict[int, int] = {}
    for i, v in enumerate(npy_doy):
        if np.isfinite(v):
            doy_to_i[int(round(float(v)))] = i

    rows = []
    for _, r in z.iterrows():
        d = r["date"]
        sd = season_doy(d, season_start)
        if sd not in doy_to_i:
            print(f"  no npy timestep for {d} (season_doy={sd})")
            continue
        i = doy_to_i[sd]
        for bi, b in enumerate(BANDS):
            zv = float(r[b]) if pd.notna(r[b]) else np.nan
            nv = float(series[i, bi])
            if not (np.isfinite(zv) and np.isfinite(nv)):
                continue
            rows.append(
                {
                    "date": d.isoformat(),
                    "band": b,
                    "zonal": zv,
                    "npy": nv,
                    "abs_diff": abs(zv - nv),
                    "rel_diff": abs(zv - nv) / max(abs(nv), 1e-6),
                }
            )

    df = pd.DataFrame(rows)
    print(f"\n=== {label} ===")
    print(f"matched band-date cells: {len(df)}")
    if df.empty:
        print("No overlap")
        print(f"  npy season DOYs: {sorted(doy_to_i)}")
        print(
            "  zonal season DOYs:",
            [season_doy(d, season_start) for d in z["date"]],
        )
        return

    print(
        f"abs_diff: mean={df['abs_diff'].mean():.4f} "
        f"median={df['abs_diff'].median():.4f} max={df['abs_diff'].max():.4f}"
    )
    print(
        f"rel_diff: mean={df['rel_diff'].mean():.4%} "
        f"median={df['rel_diff'].median():.4%} max={df['rel_diff'].max():.4%}"
    )
    # Correlation overall
    corr = np.corrcoef(df["zonal"], df["npy"])[0, 1]
    print(f"Pearson r (all bands pooled): {corr:.6f}")

    # Per-date mean abs rel over bands
    by_date = (
        df.groupby("date")
        .agg(mean_rel=("rel_diff", "mean"), mean_abs=("abs_diff", "mean"))
        .reset_index()
    )
    print(by_date.to_string(index=False))

    sample = df[df["band"].isin(["B4", "B8"])].sort_values(["date", "band"])
    print("\nB4/B8 detail:")
    print(sample.to_string(index=False))


def main() -> None:
    code = "4100103"
    year_end = 2024
    year_range = f"{year_end - 1}-{year_end}"
    season_start = date(year_end - 1, 10, 1)

    zonal = Path(
        "/mnt/c/Users/ADM/Desktop/Produtividade/crop_yield_prediction/sits_moco/"
        f"files/zonal_mean/{year_range}/{code}/{code}_zonal.csv"
    )
    npy_mean = Path.home() / f"sits_moco_data/npy_muni_mean/{year_range}/{code}/{code}.npy"
    npy_pix = Path.home() / f"sits_moco_data/npy/{year_range}/{code}/{code}.npy"

    z = pd.read_csv(zonal)
    z["date"] = pd.to_datetime(z["date"]).dt.date
    print(f"zonal: {len(z)} dates {[d.isoformat() for d in z['date']]}")

    a_mean = np.load(npy_mean)
    print(f"npy_muni_mean shape={a_mean.shape} DOYs={a_mean[0, :, 10]}")
    compare("npy_muni_mean", a_mean[0], z, season_start)

    a_pix = np.load(npy_pix)
    print(f"\npixel npy shape={a_pix.shape}")
    pix_mean = np.nanmean(a_pix, axis=0)
    compare("pixel nanmean", pix_mean, z, season_start)

    # Scale check: if npy looks like 0-1 reflectance
    print(
        f"\nvalue scale check: zonal B2 mean={z['B2'].mean():.2f}, "
        f"npy B2 mean={np.nanmean(a_mean[0, :, 0]):.2f}"
    )


if __name__ == "__main__":
    main()
