#!/usr/bin/env python3
"""
Download Brazilian municipal boundary shapefiles via geobr (IBGE).

Writes one folder per municipality in the layout expected by GEE downloaders:

  {output}/{code}/{code}.shp

By default downloads ALL municipalities in Brazil (all 27 UFs) in one geobr
fetch (code_muni="all"), then splits to per-municipality folders.

Usage:
  python data_download/download_municipal_shapefiles.py
  python data_download/download_municipal_shapefiles.py --states pr,sc,rs,mt
  python data_download/download_municipal_shapefiles.py --output files/shapefiles --year 2020
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    import geopandas as gpd
    from geobr import read_municipality
except ImportError:
    raise SystemExit("Install geobr and geopandas: pip install geobr geopandas")

# IBGE UF code -> sigla (all 27 units including DF)
UF_BY_CODE: dict[int, str] = {
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
UF_ALIASES: dict[str, int] = {
    sigla.lower(): code for code, sigla in UF_BY_CODE.items()
}

RETRYABLE = (
    ConnectionError,
    TimeoutError,
    OSError,
)


def parse_state_codes(spec: str | None) -> list[int] | None:
    """Parse 'pr,sc,rs' / IBGE codes into UF codes. None = all states."""
    if spec is None or not spec.strip():
        return None
    codes: list[int] = []
    seen: set[int] = set()
    for raw in spec.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token.isdigit():
            code = int(token)
        elif token in UF_ALIASES:
            code = UF_ALIASES[token]
        else:
            raise ValueError(
                f"Unknown UF {raw!r}; expected siglas like PR,SP or IBGE codes "
                f"{sorted(UF_BY_CODE)}"
            )
        if code not in UF_BY_CODE:
            raise ValueError(f"UF code {code} is not a valid Brazilian state/DF code")
        if code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("files/shapefiles"),
        help="Root directory (default: files/shapefiles)",
    )
    p.add_argument(
        "--year",
        type=int,
        default=2020,
        help="IBGE territorial mesh year for geobr (default: 2020)",
    )
    p.add_argument(
        "--states",
        type=str,
        default=None,
        help="Comma-separated UFs to download (e.g. pr,sc,rs,mt). Default: all Brazil",
    )
    p.add_argument(
        "--simplified",
        action="store_true",
        default=True,
        help="Use simplified geometry (default: True)",
    )
    p.add_argument(
        "--no-simplified",
        action="store_false",
        dest="simplified",
        help="Use full-resolution geometry",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite shapefiles even if {code}/{code}.shp already exists",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=8,
        help="Retries on geobr network errors (default: 8)",
    )
    p.add_argument(
        "--retry-wait",
        type=float,
        default=5.0,
        help="Initial retry wait seconds; doubles each attempt (default: 5)",
    )
    return p.parse_args()


def municipality_code_column(gdf: gpd.GeoDataFrame) -> str:
    for col in ("code_muni", "code_mun", "CD_MUN", "cod_mun"):
        if col in gdf.columns:
            return col
    raise SystemExit(
        f"Could not find municipality code column. Columns: {list(gdf.columns)}"
    )


def read_municipality_retry(
    code_muni: int | str,
    year: int,
    simplified: bool,
    retries: int,
    retry_wait: float,
    label: str,
) -> gpd.GeoDataFrame:
    """Call geobr.read_municipality with exponential backoff on network errors."""
    wait = retry_wait
    last_err: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            gdf = read_municipality(
                code_muni=code_muni,
                year=year,
                simplified=simplified,
            )
            if gdf is None or len(gdf) == 0:
                raise RuntimeError(f"geobr returned no municipalities for {label}")
            return gdf
        except RETRYABLE as e:
            last_err = e
            # geobr wraps requests errors as ConnectionError with a generic message
            if attempt >= retries:
                break
            print(
                f"  [{label}] network error (attempt {attempt}/{retries}): {e}\n"
                f"           retrying in {wait:.0f}s...",
                file=sys.stderr,
            )
            time.sleep(wait)
            wait = min(wait * 2, 120.0)
        except Exception as e:
            # Non-network: still retry a couple times if message looks like connectivity
            msg = str(e).lower()
            last_err = e
            transient = any(
                t in msg
                for t in (
                    "connection",
                    "timed out",
                    "timeout",
                    "remotely closed",
                    "remote end closed",
                    "temporarily unavailable",
                    "failed to download",
                )
            )
            if not transient or attempt >= retries:
                raise
            print(
                f"  [{label}] transient error (attempt {attempt}/{retries}): {e}\n"
                f"           retrying in {wait:.0f}s...",
                file=sys.stderr,
            )
            time.sleep(wait)
            wait = min(wait * 2, 120.0)
    raise SystemExit(
        f"Failed to download {label} after {retries} attempts: {last_err}"
    )


# ESRI Shapefile attribute names are capped at 10 chars; rename before write
# so pyogrio does not spam truncation warnings on every municipality.
SHP_COL_RENAME: dict[str, str] = {
    "abbrev_state": "abbrev_sta",
    "code_region": "code_regio",
    "name_region": "name_regio",
}


def prepare_shp_row(row: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Shorten attribute names for .shp (≤10 chars)."""
    out = row.rename(
        columns={k: v for k, v in SHP_COL_RENAME.items() if k in row.columns}
    )
    keep = [c for c in out.columns if c == "geometry" or len(c) <= 10]
    return out[keep]


def write_one(row: gpd.GeoDataFrame, code: str, out_root: Path, overwrite: bool) -> str:
    """Write a single-municipality GeoDataFrame. Returns status: wrote|skip|fail."""
    mun_dir = out_root / code
    shp_path = mun_dir / f"{code}.shp"
    if shp_path.is_file() and not overwrite:
        return "skip"
    mun_dir.mkdir(parents=True, exist_ok=True)
    try:
        prepare_shp_row(row).to_file(shp_path)
    except Exception as e:
        print(f"  FAIL {code}: {e}", file=sys.stderr)
        return "fail"
    return "wrote"


def write_gdf(
    gdf: gpd.GeoDataFrame,
    out_root: Path,
    overwrite: bool,
    label: str,
) -> tuple[int, int, int]:
    code_col = municipality_code_column(gdf)
    # Rename once on the full frame so per-row writes stay quiet
    gdf = gdf.rename(
        columns={k: v for k, v in SHP_COL_RENAME.items() if k in gdf.columns}
    )
    wrote = skipped = failed = 0
    n = len(gdf)
    for i in range(n):
        row = gdf.iloc[[i]]
        code = str(int(row.iloc[0][code_col]))
        status = write_one(row, code, out_root, overwrite)
        if status == "wrote":
            wrote += 1
        elif status == "skip":
            skipped += 1
        else:
            failed += 1
        done = wrote + skipped + failed
        if done % 100 == 0 or done == n:
            print(
                f"  [{label}] {done}/{n} "
                f"(wrote={wrote}, skip={skipped}, fail={failed})"
            )
    return wrote, skipped, failed


def main() -> None:
    args = parse_args()
    try:
        state_codes = parse_state_codes(args.states)
    except ValueError as e:
        raise SystemExit(str(e)) from e

    all_states = state_codes is None
    if state_codes is None:
        state_codes = sorted(UF_BY_CODE.keys())

    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    labels = ", ".join(UF_BY_CODE[c] for c in state_codes)
    print(
        f"Downloading municipal shapefiles → {out_root}\n"
        f"  year={args.year}, simplified={args.simplified}, "
        f"overwrite={args.overwrite}, retries={args.retries}\n"
        f"  states ({len(state_codes)}): {labels}"
    )

    total_w = total_s = total_f = 0

    if all_states:
        # One geobr request for the whole country (fewer fragile metadata fetches).
        print("[BR] fetching all municipalities (year={})...".format(args.year))
        gdf = read_municipality_retry(
            code_muni="all",
            year=args.year,
            simplified=args.simplified,
            retries=args.retries,
            retry_wait=args.retry_wait,
            label="BR",
        )
        print(f"[BR] received {len(gdf)} municipalities; writing folders...")
        total_w, total_s, total_f = write_gdf(gdf, out_root, args.overwrite, "BR")
    else:
        for code_state in state_codes:
            sigla = UF_BY_CODE[code_state]
            print(f"[{sigla}] fetching municipalities (year={args.year})...")
            gdf = read_municipality_retry(
                code_muni=code_state,
                year=args.year,
                simplified=args.simplified,
                retries=args.retries,
                retry_wait=args.retry_wait,
                label=sigla,
            )
            w, s, f = write_gdf(gdf, out_root, args.overwrite, sigla)
            total_w += w
            total_s += s
            total_f += f

    print(
        f"Done. wrote={total_w}, skipped={total_s}, failed={total_f}, "
        f"output={out_root}"
    )
    if total_f:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
