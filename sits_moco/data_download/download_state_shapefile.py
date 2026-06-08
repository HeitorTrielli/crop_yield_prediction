#!/usr/bin/env python3
"""
Download official Brazilian state boundary shapefile using geobr (IBGE data).

Mirrors the approach of municipal_shapefiles.ipynb but for state-level boundaries.
Use this to get a single state (e.g. Paraná) for running data_download/download_soy_gee_drive.py
with a state shapefile, then clip_tiffs_to_shapefiles.py to split by municipalities.

Usage:
  python data_download/download_state_shapefile.py --state PR --output files/state_shapefile
  python data_download/download_state_shapefile.py --state 41 --year 2020 --output files/parana_state.shp
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import geopandas as gpd
    from geobr import read_state
except ImportError:
    raise SystemExit("Install geobr and geopandas: pip install geobr geopandas")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download Brazilian state boundary shapefile via geobr (IBGE)."
    )
    p.add_argument(
        "--state",
        type=str,
        required=True,
        help="State code: 2-letter abbreviation (e.g. PR, SP) or IBGE code (e.g. 41 for Paraná)",
    )
    p.add_argument(
        "--year",
        type=int,
        default=2020,
        help="Reference year for the territorial mesh (default: 2020)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("files/state_shapefile"),
        help="Output path: .shp file or directory (default: files/state_shapefile). "
        "If directory, writes state_<code>.shp inside it.",
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    state_arg = args.state.strip().upper()
    # Allow numeric string (e.g. "41") or integer-like
    if state_arg.isdigit():
        code_state = int(state_arg)
    else:
        code_state = state_arg  # "PR", "SP", etc.

    gdf = read_state(code_state=code_state, year=args.year, simplified=args.simplified)
    if gdf is None or len(gdf) == 0:
        raise SystemExit(
            f"No state data returned for state={args.state}, year={args.year}"
        )

    out = Path(args.output).resolve()
    if out.suffix.lower() in (".shp", ".gpkg"):
        out_path = out
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        out.mkdir(parents=True, exist_ok=True)
        # Default filename: state_PR.shp or state_41.shp
        name = str(code_state) if isinstance(code_state, int) else code_state
        out_path = out / f"state_{name}.shp"

    gdf.to_file(out_path)
    print(f"Downloaded state shapefile: {out_path}")
    print(f"  State(s): {len(gdf)}, CRS: {gdf.crs}")


if __name__ == "__main__":
    main()
