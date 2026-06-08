#!/usr/bin/env python3
"""
Run STNet talhão-level benchmark predictions (sum of pixel forecasts inside each plot).

Wraps predict_yield_talhoes.py:
  - loads polygons from benchmark/talhoes_baseline.geojson (or .gpkg)
  - predicts each valid pixel from municipal .npy + daily TIFF georef
  - sums pixels inside each talhão polygon → forecast (tons)
  - compares to AgroIA baseline (max prod in that harvest cycle)

Example (Seed6007, all ~570 talhões, harvests 2019–2024):

  python scripts/run_benchmark_talhoes_predictions.py --harvest-years 2019 2020 2021 2022 2023 2024

Single season (2021 harvest):

  python scripts/run_benchmark_talhoes_predictions.py --year-range 2020-2021

  # optional: restrict to one or more municipalities:
  python scripts/run_benchmark_talhoes_predictions.py --municipalities 4109401
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from run_paths import model_best_in_run, predictions_dir, run_dir_from_path


def harvest_to_year_range(harvest_year: int) -> str:
    return f"{harvest_year - 1}-{harvest_year}"


def parse_args() -> argparse.Namespace:
    run_name = "Yield_STNetRegression_Pad_Hy_2020_2022_2023_Seed6007"
    default_run = _ROOT / "results" / run_name

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=f"model_best.pth (default: {default_run}/model_best.pth or training/)",
    )
    p.add_argument(
        "--datapath",
        type=Path,
        default=_ROOT / "files" / "npy",
        help="NPY root (uses {datapath}/{year-range}/{{muni}}/{{muni}}.npy)",
    )
    p.add_argument(
        "--tiffpath",
        type=Path,
        default=_ROOT / "files" / "daily_tiff",
    )
    p.add_argument(
        "--year-range",
        type=str,
        default=None,
        help="Single season folder, e.g. 2020-2021 (2021 harvest). Ignored if --harvest-years is set.",
    )
    p.add_argument(
        "--harvest-years",
        type=int,
        nargs="*",
        default=None,
        metavar="YEAR",
        help=(
            "One or more harvest years (end year of season folder). "
            "E.g. 2019 2020 2021 2022 2023 2024 runs seasons 2018-2019 … 2023-2024."
        ),
    )
    p.add_argument(
        "--sequencelength",
        type=int,
        default=45,
    )
    p.add_argument(
        "--municipalities",
        type=str,
        nargs="*",
        default=None,
        help="Optional IBGE filter; default = every talhão in the baseline shapes file",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV for a single run only (default: {run_dir}/predictions/talhoes_forecasts_{year-range}.csv)",
    )
    p.add_argument(
        "--talhoes",
        type=Path,
        default=None,
        help="Talhões .geojson or .gpkg (default: benchmark/talhoes_baseline.geojson)",
    )
    p.add_argument(
        "--baseline-csv",
        type=Path,
        default=_ROOT / "benchmark" / "produtividade_AgroIA_baseline.csv",
    )
    p.add_argument("-d", "--device", type=str, default=None)
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip harvest seasons whose output CSV already exists",
    )
    p.add_argument(
        "--full-municipality-predict",
        action="store_true",
        help="Infer every valid municipal pixel (legacy, slow)",
    )
    p.add_argument(
        "--reproject-tiffs",
        action="store_true",
        help="Slow valid-pixel mask (warp each daily TIFF)",
    )
    p.add_argument(
        "--no-mask-cache",
        action="store_true",
        help="Do not read/write {muni}.keep_mask.npy",
    )
    return p.parse_args()


def run_one(
    *,
    checkpoint: Path,
    datapath: Path,
    tiffpath: Path,
    year_range: str,
    sequencelength: int,
    talhoes: Path,
    baseline_csv: Path,
    output_csv: Path,
    municipalities: list[str] | None,
    device: str | None,
    full_municipality_predict: bool,
    reproject_tiffs: bool,
    no_mask_cache: bool,
) -> None:
    predict_py = _ROOT / "predict_yield_talhoes.py"
    cmd = [
        sys.executable,
        str(predict_py),
        "--checkpoint",
        str(checkpoint.resolve()),
        "--datapath",
        str(datapath.resolve()),
        "--tiffpath",
        str(tiffpath.resolve()),
        "--year-range",
        year_range,
        "--sequencelength",
        str(sequencelength),
        "--talhoes-gpkg",
        str(talhoes.resolve()),
        "--output-csv",
        str(output_csv.resolve()),
    ]
    if baseline_csv.is_file():
        cmd.extend(["--baseline-csv", str(baseline_csv.resolve())])
    if device:
        cmd.extend(["--device", device])
    if municipalities:
        cmd.extend(["--municipalities", *municipalities])
    if full_municipality_predict:
        cmd.append("--full-municipality-predict")
    if reproject_tiffs:
        cmd.append("--reproject-tiffs")
    if no_mask_cache:
        cmd.append("--no-mask-cache")

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=_ROOT)
    print(f"Done. Output: {output_csv}")


def main() -> None:
    args = parse_args()
    default_run = _ROOT / "results" / "Yield_STNetRegression_Pad_Hy_2020_2022_2023_Seed6007"

    if args.checkpoint is None:
        args.checkpoint = model_best_in_run(default_run)
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    run_dir = run_dir_from_path(args.checkpoint)

    if args.talhoes is None:
        gpkg = _ROOT / "benchmark" / "talhoes_baseline.gpkg"
        geojson = _ROOT / "benchmark" / "talhoes_baseline.geojson"
        args.talhoes = gpkg if gpkg.is_file() else geojson

    if args.harvest_years:
        year_ranges = [harvest_to_year_range(y) for y in args.harvest_years]
    elif args.year_range:
        year_ranges = [args.year_range]
    else:
        year_ranges = ["2020-2021"]

    if len(year_ranges) > 1 and args.output_csv is not None:
        raise SystemExit("--output-csv cannot be used with multiple --harvest-years")

    outputs: list[Path] = []
    for year_range in year_ranges:
        if args.output_csv is not None:
            output_csv = args.output_csv
        else:
            tag = year_range.replace("-", "_")
            output_csv = (
                predictions_dir(run_dir, create=True) / f"talhoes_forecasts_{tag}.csv"
            )
        if args.skip_existing and output_csv.is_file():
            print(f"Skipping {year_range} (output exists: {output_csv})")
            outputs.append(output_csv)
            continue
        run_one(
            checkpoint=args.checkpoint,
            datapath=args.datapath,
            tiffpath=args.tiffpath,
            year_range=year_range,
            sequencelength=args.sequencelength,
            talhoes=args.talhoes,
            baseline_csv=args.baseline_csv,
            output_csv=output_csv,
            municipalities=args.municipalities,
            device=args.device,
            full_municipality_predict=args.full_municipality_predict,
            reproject_tiffs=args.reproject_tiffs,
            no_mask_cache=args.no_mask_cache,
        )
        outputs.append(output_csv)

    if len(outputs) > 1:
        print(f"\nAll {len(outputs)} harvest seasons complete:")
        for path in outputs:
            print(f"  {path}")


if __name__ == "__main__":
    main()
