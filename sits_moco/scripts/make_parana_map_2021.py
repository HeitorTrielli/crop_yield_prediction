#!/usr/bin/env python3
"""
Run predict_yield (Seed6003 checkpoint, spectral Pad config) for harvest 2021, then plot Paraná error map.

Training alignment:
  - sequencelength 45 (default in main_yield_regression_polars.py for this project)
  - no --rc, no --interp (matches folder tag Pad)
  - spectral only: checkpoint has input_dim 10 (use --feature-layout spectral)
  - seed 6003 matches run name

Requires local data:
  - --datapath must be the season directory whose .npy files correspond to harvest 2021
    (often named like 2020-2021 with folders per municipality).
  - files/municipality_production_with_codes.csv with year column and production.

Dependencies for the map step: pip install geopandas geobr matplotlib

Usage:
  python scripts/make_parana_map_2021.py --datapath files/npy/2020-2021
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from run_paths import figures_dir, model_best_in_run, predictions_dir, run_dir_from_path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--datapath",
        type=Path,
        required=True,
        help="Season root: contains <municipality>/<municipality>.npy for 2021 harvest",
    )
    p.add_argument(
        "--yield-csv",
        type=Path,
        default=None,
        help="Defaults to files/municipality_production_with_codes.csv under repo root",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="model_best.pth (default: Seed6003 run under results/, new or legacy layout)",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Forecasts CSV (default: {run_dir}/predictions/yield_forecasts_2021_pr.csv)",
    )
    p.add_argument(
        "--output-map",
        type=Path,
        default=None,
        help="Map PNG (default: {run_dir}/figures/map_pr_error_2021.png)",
    )
    p.add_argument("--sequencelength", type=int, default=45)
    p.add_argument("--seed", type=int, default=6003, help="Match training seed for run name parity")
    p.add_argument(
        "--skip-predict",
        action="store_true",
        help="Only build map from existing --output-csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    yield_csv = (
        Path(args.yield_csv)
        if args.yield_csv
        else root / "files/municipality_production_with_codes.csv"
    )

    if args.checkpoint is None:
        run_guess = root / "results/Yield_STNetRegression_Pad_Hy_2020_2022_2023_2024_Seed6003"
        args.checkpoint = model_best_in_run(run_guess)
    else:
        args.checkpoint = Path(args.checkpoint)

    run_dir = run_dir_from_path(args.checkpoint)
    if args.output_csv is None:
        args.output_csv = predictions_dir(run_dir, create=True) / "yield_forecasts_2021_pr.csv"
    if args.output_map is None:
        args.output_map = figures_dir(run_dir, create=True) / "map_pr_error_2021.png"

    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if not yield_csv.is_file():
        raise SystemExit(f"Yield CSV not found: {yield_csv}")

    predict_py = root / "predict_yield.py"
    plot_py = root / "plot_municipal_yield_map.py"

    if not args.skip_predict:
        if not args.datapath.is_dir():
            raise SystemExit(f"--datapath is not a directory: {args.datapath}")

        cmd = [
            sys.executable,
            str(predict_py),
            "--checkpoint",
            str(args.checkpoint),
            "--datapath",
            str(args.datapath.resolve()),
            "--yield-csv",
            str(yield_csv.resolve()),
            "--yield-year",
            "2021",
            "--restrict-to-yield-year",
            "--sequencelength",
            str(args.sequencelength),
            "--seed",
            str(args.seed),
            "--feature-layout",
            "spectral",
            "--output-csv",
            str(args.output_csv.resolve()),
        ]
        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd, cwd=root)

    if not args.output_csv.is_file():
        raise SystemExit(f"No forecasts CSV at {args.output_csv}")

    if not plot_py.is_file():
        raise SystemExit(f"Missing {plot_py}")

    cmd_map = [
        sys.executable,
        str(plot_py),
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--forecasts-csv",
        str(args.output_csv.resolve()),
        "--state",
        "PR",
        "--column",
        "error",
        "--output",
        str(args.output_map.resolve()),
        "--title",
        "Paraná — erro de previsão (t) vs produção 2021 — Seed6003",
    ]
    print("Running:", " ".join(cmd_map))
    subprocess.check_call(cmd_map, cwd=root)
    print(f"Done.\n  CSV: {args.output_csv}\n  Map: {args.output_map}")


if __name__ == "__main__":
    main()
