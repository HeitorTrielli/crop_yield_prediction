#!/usr/bin/env python3
"""Download PAM soybean data from IBGE SIDRA for selected UF municipalities.

Default years are 2019-2025 (SIDRA drops years that are not published yet).

Usage:
  python data_download/download_pam_soy_sidra.py
  python data_download/download_pam_soy_sidra.py --states all --years 2020-2024
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
UF_BY_CODE = {
    11: "RO",
    12: "AC",
    13: "AM",
    14: "RR",
    15: "PA",
    16: "AP",
    17: "TO",
    21: "MA",
    22: "PI",
    23: "CE",
    24: "RN",
    25: "PB",
    26: "PE",
    27: "AL",
    28: "SE",
    29: "BA",
    31: "MG",
    32: "ES",
    33: "RJ",
    35: "SP",
    41: "PR",
    42: "SC",
    43: "RS",
    50: "MS",
    51: "MT",
    52: "GO",
    53: "DF",
}
UF_ALIASES = {
    "all": 0,
    "br": 0,
    "brazil": 0,
    "brasil": 0,
    "ro": 11,
    "rondonia": 11,
    "rondônia": 11,
    "ac": 12,
    "acre": 12,
    "am": 13,
    "amazonas": 13,
    "rr": 14,
    "roraima": 14,
    "pa": 15,
    "para": 15,
    "pará": 15,
    "ap": 16,
    "amapa": 16,
    "amapá": 16,
    "to": 17,
    "tocantins": 17,
    "ma": 21,
    "maranhao": 21,
    "maranhão": 21,
    "pi": 22,
    "piaui": 22,
    "piauí": 22,
    "ce": 23,
    "ceara": 23,
    "ceará": 23,
    "rn": 24,
    "rio grande do norte": 24,
    "pb": 25,
    "paraiba": 25,
    "paraíba": 25,
    "pe": 26,
    "pernambuco": 26,
    "al": 27,
    "alagoas": 27,
    "se": 28,
    "sergipe": 28,
    "ba": 29,
    "bahia": 29,
    "mg": 31,
    "minas gerais": 31,
    "es": 32,
    "espirito santo": 32,
    "espírito santo": 32,
    "rj": 33,
    "rio de janeiro": 33,
    "sp": 35,
    "são paulo": 35,
    "sao paulo": 35,
    "pr": 41,
    "paraná": 41,
    "parana": 41,
    "sc": 42,
    "santa catarina": 42,
    "rs": 43,
    "rio grande do sul": 43,
    "ms": 50,
    "mato grosso do sul": 50,
    "mt": 51,
    "mato grosso": 51,
    "go": 52,
    "goias": 52,
    "goiás": 52,
    "df": 53,
    "distrito federal": 53,
}


def parse_state_codes(spec: str) -> list[int]:
    """Parse 'pr,sc,rs,mt', 'all', or IBGE UF codes into a unique ordered list."""
    codes: list[int] = []
    seen: set[int] = set()
    for raw in spec.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token in ("all", "br", "brazil", "brasil"):
            return sorted(UF_BY_CODE)
        if token.isdigit():
            code = int(token)
        elif token in UF_ALIASES:
            code = UF_ALIASES[token]
            if code == 0:
                return sorted(UF_BY_CODE)
        else:
            raise ValueError(
                f"Unknown UF {raw!r}; expected one of {sorted(set(UF_ALIASES))} "
                f"or IBGE codes {sorted(UF_BY_CODE)}"
            )
        if code not in UF_BY_CODE:
            raise ValueError(
                f"UF code {code} is not in the soy download map {sorted(UF_BY_CODE)}"
            )
        if code not in seen:
            seen.add(code)
            codes.append(code)
    if not codes:
        raise ValueError("--states must list at least one UF")
    return codes


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
        "uf_code",
        "uf",
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
                VAR_AREA_PLANTED_HA: "area_planted_ha",
                VAR_PRODUCTION_T: "production_t",
                VAR_YIELD_KG_HA: "yield_kg_ha",
            }
        )
    )
    wide.columns.name = None
    for col, dtype in (
        ("area_planted_ha", "float64"),
        ("production_t", "float64"),
        ("yield_kg_ha", "float64"),
    ):
        if col not in wide.columns:
            wide[col] = np.nan
        else:
            wide[col] = pd.to_numeric(wide[col], errors="coerce")
    wide["municipality_code"] = wide["municipality_code"].astype(int)
    wide["year"] = wide["year"].astype(int)
    wide["uf_code"] = (wide["municipality_code"] // 100000).astype(int)
    wide["uf"] = wide["uf_code"].map(lambda c: UF_BY_CODE.get(int(c), f"{int(c):02d}"))
    wide["yield_t_ha"] = wide["yield_kg_ha"] / 1000.0

    derived = wide["area_planted_ha"].isna() & wide["production_t"].notna() & (
        wide["yield_t_ha"] > 0
    )
    wide.loc[derived, "area_planted_ha"] = (
        wide.loc[derived, "production_t"] / wide.loc[derived, "yield_t_ha"]
    )

    return wide[base_columns].sort_values(["uf_code", "municipality_code", "year"])


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
        description="Download PAM soybean production from IBGE SIDRA."
    )
    parser.add_argument(
        "--years",
        default="2019-2025",
        help="SIDRA period spec (default: 2019-2025; unpublished years are omitted)",
    )
    parser.add_argument(
        "--states",
        default="pr",
        help="Comma-separated UFs (pr,sc,rs,mt) or 'all' for every UF (default: pr)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: files/pam_soy_pr_2019_2025.csv, or a multi-UF name)",
    )
    return parser.parse_args()


def default_output_path(state_codes: list[int], years: str) -> Path:
    years_tag = years.replace(",", "_").replace("-", "_")
    if state_codes == [PR_STATE_CODE] and years == "2019-2025":
        return Path("files/pam_soy_pr_2019_2025.csv")
    if state_codes == sorted(UF_BY_CODE):
        return Path(f"files/pam_soy_br_{years_tag}.csv")
    tag = "_".join(UF_BY_CODE[c].lower() for c in state_codes)
    return Path(f"files/pam_soy_{tag}_{years_tag}.csv")


def fetch_states(years: str, state_codes: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for code in state_codes:
        url = build_sidra_url(years=years, state_code=code)
        print(f"Fetching {UF_BY_CODE.get(code, code)} ({code}): {url}")
        try:
            records = fetch_sidra(url)
            part = sidra_to_dataframe(records)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue
        print(f"  {len(part)} rows")
        frames.append(part)
    if not frames:
        return sidra_to_dataframe([])
    return pd.concat(frames, ignore_index=True).sort_values(
        ["uf_code", "municipality_code", "year"]
    )


def main() -> None:
    args = parse_args()
    state_codes = parse_state_codes(args.states)
    args.output = args.output or default_output_path(state_codes, args.years)
    df = fetch_states(args.years, state_codes)
    df = preserve_existing_columns(df, args.output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    years = sorted(df["year"].unique()) if not df.empty else []
    print(f"Saved {len(df)} rows to {args.output}")
    if not df.empty and "uf" in df.columns:
        for uf, n in df.groupby("uf").size().items():
            print(f"  {uf}: {n} rows")
    if years:
        print(f"Years available: {years[0]}–{years[-1]} ({years})")
    if args.years.endswith("2025") and (not years or max(years) < 2025):
        print("Note: PAM 2025 is not published yet in SIDRA; only released years are included.")
    if "split" not in df.columns:
        print("Run add_train_valid_test_split.py before training (adds train/valid/test split).")


if __name__ == "__main__":
  main()
