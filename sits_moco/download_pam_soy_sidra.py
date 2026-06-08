#!/usr/bin/env python3
"""Download PAM soybean data from IBGE SIDRA for Paraná municipalities."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

SIDRA_BASE = "https://apisidra.ibge.gov.br/values"
TABLE_TEMPORARY_CROPS = 1612
SOYBEAN_PRODUCT = 2713
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
        f"v/{VAR_PRODUCTION_T},{VAR_YIELD_KG_HA}",
        f"p/{years}",
        f"c81/{SOYBEAN_PRODUCT}",
        "h/n",
    ]
    return f"{SIDRA_BASE}/" + "/".join(urllib.parse.quote(part, safe="/") for part in parts)


def fetch_sidra(url: str, *, timeout: int = 120) -> list[dict]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def sidra_to_dataframe(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=[
                "municipality_code",
                "municipality_name",
                "year",
                "production_t",
                "yield_t_ha",
            ]
        )

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
                str(VAR_PRODUCTION_T): "production_t",
                str(VAR_YIELD_KG_HA): "yield_kg_ha",
            }
        )
    )
    wide.columns.name = None
    wide["municipality_code"] = wide["municipality_code"].astype(int)
    wide["year"] = wide["year"].astype(int)
    wide["yield_t_ha"] = wide["yield_kg_ha"] / 1000.0
    return wide[
        [
            "municipality_code",
            "municipality_name",
            "year",
            "production_t",
            "yield_t_ha",
        ]
    ].sort_values(["municipality_code", "year"])


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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    years = sorted(df["year"].unique()) if not df.empty else []
    print(f"Saved {len(df)} rows to {args.output}")
    if years:
        print(f"Years available: {years[0]}–{years[-1]}")
    if args.years.endswith("2025") and (not years or years[-1] < 2025):
        print("Note: PAM 2025 is not published yet in SIDRA; only released years are included.")


if __name__ == "__main__":
  main()
