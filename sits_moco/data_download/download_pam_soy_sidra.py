#!/usr/bin/env python3
"""Download PAM soybean data from IBGE SIDRA for Paraná municipalities.

Usage:
  python data_download/download_pam_soy_sidra.py
  python add_train_valid_test_split.py --holdout-years 2021
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SIDRA_BASE = "https://apisidra.ibge.gov.br/values"
TABLE_TEMPORARY_CROPS = 1612
SOYBEAN_PRODUCT = 2713
VAR_AREA_PLANTED_HA = 109
VAR_PRODUCTION_T = 214
VAR_YIELD_KG_HA = 112
PR_STATE_CODE = 41


def build_sidra_url(
    *,
    years: str,
    state_code: int = PR_STATE_CODE,
) -> str:
    parts = [
        f"t/{TABLE_TEMPORARY_CROPS}",
        f"n6/in n3 {state_code}",
        f"v/{VAR_AREA_PLANTED_HA},{VAR_PRODUCTION_T},{VAR_YIELD_KG_HA}",
        f"p/{years}",
        f"c81/{SOYBEAN_PRODUCT}",
        "h/n",
    ]
    return f"{SIDRA_BASE}/" + "/".join(urllib.parse.quote(part, safe="/") for part in parts)


def fetch_sidra(url: str, *, timeout: int = 120) -> list[dict]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def sidra_to_dataframe(records: list[dict]) -> pd.DataFrame:
    base_columns = [
        "municipality_code",
        "municipality_name",
        "year",
        "area_planted_ha",
        "production_t",
        "yield_t_ha",
    ]
    if not records:
        return pd.DataFrame(columns=base_columns)

    df = pd.DataFrame(records)
    df = df.loc[df["V"] != "-"].copy()
    df["value"] = pd.to_numeric(df["V"], errors="coerce")

    wide = (
        df.pivot_table(
            index=["D1C", "D1N", "D3C"],
            columns="D2C",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename(
            columns={
                "D1C": "municipality_code",
                "D1N": "municipality_name",
                "D3C": "year",
                str(VAR_AREA_PLANTED_HA): "area_planted_ha",
                str(VAR_PRODUCTION_T): "production_t",
                str(VAR_YIELD_KG_HA): "yield_kg_ha",
            }
        )
    )
    wide.columns.name = None
    wide["municipality_code"] = wide["municipality_code"].astype(int)
    wide["year"] = wide["year"].astype(int)
    if "yield_kg_ha" in wide.columns:
        wide["yield_t_ha"] = wide["yield_kg_ha"] / 1000.0
    if "area_planted_ha" not in wide.columns:
        wide["area_planted_ha"] = np.nan

    derived = wide["area_planted_ha"].isna() & wide["production_t"].notna() & (
        wide["yield_t_ha"] > 0
    )
    wide.loc[derived, "area_planted_ha"] = (
        wide.loc[derived, "production_t"] / wide.loc[derived, "yield_t_ha"]
    )

    return wide[base_columns].sort_values(["municipality_code", "year"])


def preserve_existing_columns(
    df_new: pd.DataFrame,
    existing_path: Path,
    *,
    key_cols: tuple[str, ...] = ("municipality_code", "year"),
    preserve_cols: tuple[str, ...] = ("split",),
) -> pd.DataFrame:
    """Keep selected columns from an existing CSV when re-downloading SIDRA data."""
    if not existing_path.is_file():
        return df_new
    old = pd.read_csv(existing_path)
    merge_cols = [c for c in preserve_cols if c in old.columns]
    if not merge_cols:
        return df_new
    keys = list(key_cols)
    extra = old[keys + merge_cols].drop_duplicates(subset=keys)
    out = df_new.merge(extra, on=keys, how="left")
    missing = out[merge_cols[0]].isna().sum()
    if missing:
        print(
            f"Note: {missing} new SIDRA rows have no preserved {merge_cols[0]} "
            f"(re-run add_train_valid_test_split.py if needed)."
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download PAM soybean production for Paraná from IBGE SIDRA."
    )
    parser.add_argument(
        "--years",
        default="2019-2025",
        help="SIDRA period spec (default: 2019-2025)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("files/pam_soy_pr_2019_2025.csv"),
        help="Output CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    url = build_sidra_url(years=args.years)
    print(f"Fetching: {url}")
    records = fetch_sidra(url)
    df = sidra_to_dataframe(records)
    df = preserve_existing_columns(df, args.output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    years = sorted(df["year"].unique()) if not df.empty else []
    print(f"Saved {len(df)} rows to {args.output}")
    if years:
        print(f"Years available: {years[0]}–{years[-1]}")
    if args.years.endswith("2025") and (not years or years[-1] < 2025):
        print("Note: PAM 2025 is not published yet in SIDRA; only released years are included.")
    if "split" not in df.columns:
        print("Run add_train_valid_test_split.py before training (adds train/valid/test split).")


if __name__ == "__main__":
  main()
