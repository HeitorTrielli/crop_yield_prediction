"""Correlate MapBiomas Solo municipal means with IBGE PAM soy yield.

Uses all-land municipal zonal means (not soy-masked) of clay/silt/sand (0–30 cm)
and year-matched SOC stock vs coverage-filtered PAM yield_t_ha.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
SOLO = REPO / "files" / "mapbiomas_solo"
SHP_ROOT = REPO / "files" / "municipal_shapefiles"
YIELD_CSV = REPO / "files" / "pam_soy_pr_2019_2025_coverage_0.8_1.2.csv"
OUT_DIR = REPO / "results" / "debug_inputs"
SOIL_CACHE = OUT_DIR / "muni_soil_features.csv"
CORR_JSON = OUT_DIR / "soil_yield_correlation.json"
PANEL_CSV = OUT_DIR / "soil_yield_panel.csv"

NODATA = -9999.0
SOC_YEARS = list(range(2019, 2025))
FEATURES = ["clay_pct", "silt_pct", "sand_pct", "soc_t_ha", "clay_sand_ratio"]


def _mean_masked(src, geoms, band: int = 1) -> tuple[float, int]:
    out, _ = mask(src, geoms, crop=True, filled=True, nodata=NODATA, indexes=band)
    # rasterio.mask returns (1, H, W) for a list of indexes, but (H, W) for a
    # single int index — do not index [0] in the 2D case (that drops to a row).
    if getattr(out, "ndim", 0) == 3:
        a = out[0].astype(np.float64)
    else:
        a = np.asarray(out, dtype=np.float64)
    a = np.where(a == NODATA, np.nan, a)
    n = int(np.isfinite(a).sum())
    if n == 0:
        return float("nan"), 0
    return float(np.nanmean(a)), n


def _muni_code(gdf: gpd.GeoDataFrame, shp: Path) -> str:
    if "code_muni" in gdf.columns:
        code = str(gdf["code_muni"].iloc[0])
    else:
        code = shp.parent.name.split("_")[0]
    return f"{int(float(code)):07d}"


def build_soil_table(*, overwrite: bool = False) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if SOIL_CACHE.exists() and not overwrite:
        print(f"loading cache {SOIL_CACHE}")
        return pd.read_csv(SOIL_CACHE)

    shps = sorted(SHP_ROOT.rglob("*.shp"))
    print(f"{len(shps)} municipality shapefiles")

    # MapBiomas Solo clips often lack a clean CRS authority; treat as SIRGAS 2000.
    soil_crs = "EPSG:4674"

    rows: list[dict] = []
    tex_path = SOLO / "pr_texture_0_30cm.tif"
    with rasterio.open(tex_path) as tex:
        for i, shp in enumerate(shps, 1):
            gdf = gpd.read_file(shp)
            if gdf.crs is None:
                gdf = gdf.set_crs(soil_crs)
            code = _muni_code(gdf, shp)
            name = (
                str(gdf["name_muni"].iloc[0]) if "name_muni" in gdf.columns else code
            )
            geoms = list(gdf.to_crs(soil_crs).geometry)
            clay, n = _mean_masked(tex, geoms, band=1)
            silt, _ = _mean_masked(tex, geoms, band=2)
            sand, _ = _mean_masked(tex, geoms, band=3)
            rows.append(
                {
                    "municipality_code": int(code),
                    "municipality_name": name,
                    "clay_pct": clay,
                    "silt_pct": silt,
                    "sand_pct": sand,
                    "n_soil_pixels": n,
                }
            )
            if i % 50 == 0 or i == len(shps):
                print(f"  texture {i}/{len(shps)}")

    df = pd.DataFrame(rows)
    for y in SOC_YEARS:
        path = SOLO / f"pr_soc_0_30cm_{y}.tif"
        print(f"SOC {y}...")
        means: list[tuple[int, float]] = []
        with rasterio.open(path) as src:
            for i, shp in enumerate(shps, 1):
                gdf = gpd.read_file(shp)
                if gdf.crs is None:
                    gdf = gdf.set_crs(soil_crs)
                code = int(_muni_code(gdf, shp))
                geoms = list(gdf.to_crs(soil_crs).geometry)
                m, _ = _mean_masked(src, geoms, band=1)
                means.append((code, m))
                if i % 100 == 0 or i == len(shps):
                    print(f"  {y}: {i}/{len(shps)}")
        df = df.merge(
            pd.DataFrame(means, columns=["municipality_code", f"soc_{y}"]),
            on="municipality_code",
            how="left",
        )

    df.to_csv(SOIL_CACHE, index=False)
    print(f"wrote {SOIL_CACHE}")
    return df


def corr_pair(x: np.ndarray, y: np.ndarray) -> dict:
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 5:
        return {
            "n": int(m.sum()),
            "pearson_r": None,
            "pearson_p": None,
            "spearman_r": None,
            "spearman_p": None,
            "r2": None,
        }
    pr = stats.pearsonr(x[m], y[m])
    sr = stats.spearmanr(x[m], y[m])
    return {
        "n": int(m.sum()),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_r": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
        "r2": float(pr.statistic**2),
    }


def ols_r2(X: np.ndarray, y: np.ndarray) -> dict:
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[m]
    y = y[m]
    n, k = X.shape
    if n <= k + 1:
        return {"n": int(n), "r2": None, "adj_r2": None}
    Xd = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    pred = Xd @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)
    return {"n": int(n), "r2": float(r2), "adj_r2": float(adj), "coefs": beta.tolist()}


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--overwrite-soil-cache",
        action="store_true",
        help="Recompute municipal soil zonal means",
    )
    args = p.parse_args()
    soil = build_soil_table(overwrite=args.overwrite_soil_cache)
    yield_df = pd.read_csv(YIELD_CSV)
    yield_df["municipality_code"] = yield_df["municipality_code"].astype(int)

    panel = yield_df.merge(soil, on="municipality_code", how="inner")
    soc_vals = []
    for _, r in panel.iterrows():
        col = f"soc_{int(r['year'])}"
        soc_vals.append(float(r[col]) if col in panel.columns else float("nan"))
    panel["soc_t_ha"] = soc_vals
    panel["clay_sand_ratio"] = panel["clay_pct"] / panel["sand_pct"].clip(lower=1e-6)
    panel.to_csv(PANEL_CSV, index=False)

    y = panel["yield_t_ha"].to_numpy(dtype=float)
    univariate = {
        f: corr_pair(panel[f].to_numpy(dtype=float), y) for f in FEATURES
    }

    panel2 = panel.copy()
    panel2["yield_dm"] = panel2["yield_t_ha"] - panel2.groupby("year")[
        "yield_t_ha"
    ].transform("mean")
    for f in FEATURES:
        panel2[f"{f}_dm"] = panel2[f] - panel2.groupby("year")[f].transform("mean")
    univariate_year_dm = {
        f: corr_pair(
            panel2[f"{f}_dm"].to_numpy(dtype=float),
            panel2["yield_dm"].to_numpy(dtype=float),
        )
        for f in FEATURES
    }

    by_year = {}
    for yr, g in panel.groupby("year"):
        by_year[str(int(yr))] = {
            f: corr_pair(
                g[f].to_numpy(dtype=float), g["yield_t_ha"].to_numpy(dtype=float)
            )
            for f in FEATURES
        }

    muni = (
        panel.groupby("municipality_code")
        .agg(
            yield_mean=("yield_t_ha", "mean"),
            clay_pct=("clay_pct", "first"),
            silt_pct=("silt_pct", "first"),
            sand_pct=("sand_pct", "first"),
            soc_mean=("soc_t_ha", "mean"),
            clay_sand_ratio=("clay_sand_ratio", "first"),
            n_years=("year", "count"),
        )
        .reset_index()
    )
    cross = {
        "clay_pct": corr_pair(muni["clay_pct"].to_numpy(), muni["yield_mean"].to_numpy()),
        "silt_pct": corr_pair(muni["silt_pct"].to_numpy(), muni["yield_mean"].to_numpy()),
        "sand_pct": corr_pair(muni["sand_pct"].to_numpy(), muni["yield_mean"].to_numpy()),
        "soc_mean": corr_pair(muni["soc_mean"].to_numpy(), muni["yield_mean"].to_numpy()),
        "clay_sand_ratio": corr_pair(
            muni["clay_sand_ratio"].to_numpy(), muni["yield_mean"].to_numpy()
        ),
    }

    multi = {
        "clay_silt": ols_r2(
            panel[["clay_pct", "silt_pct"]].to_numpy(dtype=float), y
        ),
        "clay_silt_sand": ols_r2(
            panel[["clay_pct", "silt_pct", "sand_pct"]].to_numpy(dtype=float), y
        ),
        "clay_silt_soc": ols_r2(
            panel[["clay_pct", "silt_pct", "soc_t_ha"]].to_numpy(dtype=float), y
        ),
        "all_four": ols_r2(
            panel[["clay_pct", "silt_pct", "sand_pct", "soc_t_ha"]].to_numpy(
                dtype=float
            ),
            y,
        ),
    }
    multi_year_dm = {
        "clay_silt_soc_year_demeaned": ols_r2(
            panel2[["clay_pct_dm", "silt_pct_dm", "soc_t_ha_dm"]].to_numpy(dtype=float),
            panel2["yield_dm"].to_numpy(dtype=float),
        )
    }

    within = []
    for _, g in panel.groupby("municipality_code"):
        if len(g) < 4:
            continue
        c = corr_pair(
            g["soc_t_ha"].to_numpy(dtype=float), g["yield_t_ha"].to_numpy(dtype=float)
        )
        if c["pearson_r"] is not None:
            within.append(c["pearson_r"])
    within_soc = {
        "n_munis": len(within),
        "mean_pearson_r": float(np.mean(within)) if within else None,
        "median_pearson_r": float(np.median(within)) if within else None,
        "frac_positive": float(np.mean(np.asarray(within) > 0)) if within else None,
    }

    summary = {
        "scope": (
            "Paraná municipalities; MapBiomas Solo C3 municipal means "
            "(all land, not soy-masked) vs IBGE PAM soy yield_t_ha; "
            "coverage-filtered CSV 0.8–1.2"
        ),
        "descriptive": {
            "n_panel_rows": int(len(panel)),
            "n_municipalities": int(panel["municipality_code"].nunique()),
            "years": sorted(int(x) for x in panel["year"].unique()),
            "yield_mean": float(panel["yield_t_ha"].mean()),
            "yield_std": float(panel["yield_t_ha"].std()),
            "clay_mean": float(panel["clay_pct"].mean()),
            "sand_mean": float(panel["sand_pct"].mean()),
            "soc_mean": float(panel["soc_t_ha"].mean()),
        },
        "univariate_pooled": univariate,
        "univariate_year_demeaned": univariate_year_dm,
        "cross_section_muni_means": cross,
        "by_year": by_year,
        "multivariate_ols": multi,
        "multivariate_year_demeaned": multi_year_dm,
        "within_muni_soc_vs_yield": within_soc,
    }
    with open(CORR_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {CORR_JSON}")
    print(f"Wrote {PANEL_CSV}")


if __name__ == "__main__":
    main()
