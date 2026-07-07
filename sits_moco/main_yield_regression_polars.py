"""
Main training script for municipality-level yield prediction using aggregated regression.
Polars-optimized version for faster index loading.

Pixel-level predictions are aggregated per municipality (sum for total, mean for productivity).
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.optim
from torch.utils.data import DataLoader

# Import Polars version of dataset
from datasets.feature_layout import feature_layout_choices
from datasets.uscrops_aggregated_npy_polars import USCropsAggregatedNPY
from models.weight_init import weight_init_regression
from run_paths import (
    ensure_run_layout,
    experiment_dir,
    load_training_sessions,
    run_dir_for_experiment,
    trainlog_path,
)
from utils import (
    AverageMeter,
    adjust_learning_rate,
    get_ntrainparams,
    recursive_todevice,
    save,
)
from utils_aggregated import (
    TARGET_SPECS,
    AggregatedMSELoss,
    aggregated_collate_fn,
    regression_metrics,
    resolve_target,
    stnet_regression_input_dim_from_state_dict,
    test_epoch_aggregated,
    train_epoch_aggregated,
)

from training.early_stop import (
    early_stop_state_from_trainlog,
    initial_session_optimizer_hparams,
)

# Default paths (datapath: see SITS_MOCO_DATAPATH in .env)
from env_config import resolve_datapath

YIELD_CSV = Path("files/pam_soy_pr_2019_2025.csv")
DEFAULT_REGRESSION_MODEL = "STNetRegression"
DEFAULT_SEED = 42
DEFAULT_SEQUENCELENGTH = 45
DEFAULT_BATCHSIZE = 4
DEFAULT_WORKERS = 4


def target_run_prefix(target: str) -> str:
    if target == "productivity":
        return "Productivity"
    if target == "total_adj":
        return "TotalAdj"
    return "Total"


def format_years_for_run_name(harvest_years_set: set[int]) -> str:
    return "_".join(f"{y % 100:02d}" for y in sorted(harvest_years_set))


def build_run_name(
    *,
    target: str = "total",
    model_class_name: str = DEFAULT_REGRESSION_MODEL,
    harvest_years_set: set[int],
    seed: int,
    suffix: str | None = None,
) -> str:
    parts = [target_run_prefix(target)]
    if model_class_name != DEFAULT_REGRESSION_MODEL:
        parts.append(model_class_name)
    parts.append(format_years_for_run_name(harvest_years_set))
    parts.append(f"Seed{seed}")
    if suffix:
        parts.append(suffix)
    return "_".join(parts)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train yield prediction model using aggregated regression (Polars-optimized)."
    )
    parser.add_argument(
        "--rc",
        action="store_true",
        help="whether to random choice the time series data",
    )
    parser.add_argument(
        "--interp",
        action="store_true",
        help="whether to interpolate the time series data",
    )
    parser.add_argument(
        "-seq",
        "--sequencelength",
        type=int,
        default=DEFAULT_SEQUENCELENGTH,
        help=(
            f"Max time steps per pixel (pad/sample to this; default: {DEFAULT_SEQUENCELENGTH}). "
            "Match your .npy data; larger values use more GPU memory."
        ),
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"number of CPU workers to load the next batch (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=100, help="number of training epochs"
    )
    parser.add_argument(
        "-b",
        "--batchsize",
        type=int,
        default=DEFAULT_BATCHSIZE,
        help=f"batch size (number of municipalities per batch, default: {DEFAULT_BATCHSIZE})",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=1e-5,
        help="optimizer learning rate (default 1e-5)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="optimizer weight_decay (default 1e-4)",
    )
    parser.add_argument("--warmup-epochs", type=int, default=0, help="warmup epochs")
    parser.add_argument(
        "--schedule",
        default=None,
        nargs="*",
        type=int,
        help="learning rate schedule (when to drop lr by a ratio)",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="./results",
        help="logdir to store progress and models (defaults to ./results)",
    )
    parser.add_argument("-s", "--suffix", default=None, help="suffix to output_dir")
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        help='torch.Device. either "cpu" or "cuda". default will check by torch.cuda.is_available()',
    )
    parser.add_argument(
        "--pretrained", default=None, type=str, help="path to pretrained checkpoint"
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile (recommended on WSL to avoid orphan compile workers)",
    )
    parser.add_argument(
        "--pixel-chunk-size",
        type=int,
        default=16000,
        help="Pixels per GPU forward pass (default: 16000)",
    )
    parser.add_argument(
        "--prefetch-chunks",
        type=int,
        default=2,
        metavar="N",
        help=(
            "Prefetch N upcoming pixel chunks on a CPU thread while the GPU runs "
            "(default: 2). Overlaps .npy read + transform + stack with forward/backward. "
            "Uses a small amount of RAM, not VRAM. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--quiet-training",
        action="store_true",
        help=(
            "Skip per-chunk GPU NaN/Inf sync checks during training (faster). "
            "Municipality summary lines and one batch progress line per completed batch are kept."
        ),
    )
    parser.add_argument(
        "--disable-pipeline-h2d",
        action="store_true",
        help=(
            "Disable CUDA-stream overlap for host→device pixel-chunk copies "
            "(default: enabled on CUDA when --quiet-training)."
        ),
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default=None,
        help="path to dataset root directory (default: SITS_MOCO_DATAPATH from .env)",
    )
    parser.add_argument(
        "--yield-csv",
        type=str,
        default=None,
        help=f"path to yield CSV (default: {YIELD_CSV})",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=tuple(TARGET_SPECS.keys()),
        help=(
            "Municipality label: 'total' = production_t (tons, sum pixels); "
            "'total_adj' = production_t_s2_adj (tons scaled to S2 coverage, sum pixels); "
            "'productivity' = yield_t_ha (t/ha, mean pixels)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (used in run name and dataloaders; default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--overwrite-run",
        action="store_true",
        help=(
            "Allow starting a fresh run in an existing results folder. "
            "Without this, training aborts if model_best.pth already exists and "
            "--pretrained is not set."
        ),
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="Fraction of municipalities to sample per epoch (0.0-1.0, default: 1.0 = use all)",
    )
    parser.add_argument(
        "--harvest-years",
        type=str,
        required=True,
        help=(
            "Comma-separated harvest years, e.g. 2020,2021,2022,2023,2024. "
            "Requires a 'year' column in the yield CSV (harvest year). "
            "Folder Y1-Y2 under --datapath maps to harvest year Y2."
        ),
    )
    parser.add_argument(
        "--feature-layout",
        type=str,
        default="spectral",
        choices=feature_layout_choices(),
        metavar="NAME",
        help=(
            "Explicit per-pixel inputs to STNet: "
            "'spectral' = 10 S2 bands only; "
            "'spectral_xavier' = 10 bands + 2 rain channels (requires 13-channel daily .npy). "
            "See datasets/feature_layout.py to add layouts (e.g. ET)."
        ),
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=30,
        help="Stop training after this many epochs without val-loss improvement (default: 30)",
    )
    parser.add_argument(
        "--model-d-model",
        type=int,
        default=128,
        help="STNet transformer width d_model (default: 128)",
    )
    parser.add_argument(
        "--model-n-head",
        type=int,
        default=16,
        help="STNet transformer attention heads (default: 16)",
    )
    parser.add_argument(
        "--model-n-layers",
        type=int,
        default=1,
        help="STNet transformer encoder layers (default: 1)",
    )
    parser.add_argument(
        "--model-d-inner",
        type=int,
        default=128,
        help="STNet transformer FFN inner dim (default: 128)",
    )
    parser.add_argument(
        "--model-dropout",
        type=float,
        default=0.2,
        help="STNet dropout probability (default: 0.2)",
    )
    # --- Training runtime (defaults in training_runtime.py; override when debugging) ---
    parser.add_argument(
        "--legacy-zero-grad",
        action="store_true",
        help="Use optimizer.zero_grad() instead of zero_grad(set_to_none=True)",
    )
    parser.add_argument(
        "--batch-empty-cache",
        action="store_true",
        help="gc.collect + cuda.empty_cache after each batch (legacy; slower)",
    )
    parser.add_argument(
        "--dataloader-prefetch-factor",
        type=int,
        default=None,
        help="DataLoader prefetch_factor per worker (default: 4)",
    )
    parser.add_argument(
        "--npy-cache-size",
        type=int,
        default=None,
        help="LRU cache size for mmap'd municipality .npy files (default: 20)",
    )
    parser.add_argument(
        "--no-dataloader-pin-memory",
        action="store_true",
        help="DataLoader pin_memory=False (metadata batches only)",
    )
    parser.add_argument(
        "--chunks-per-grad",
        type=int,
        default=1,
        help="Backward every N pixel chunks (default: 1; >1 retains autograd graphs longer)",
    )
    parser.add_argument(
        "--auto-chunks-per-grad",
        action="store_true",
        help=(
            "At startup, search the largest chunks-per-grad x pixel-chunk-size that "
            "fits dedicated VRAM (binary search + local refine), then train all "
            "epochs with that value"
        ),
    )
    parser.add_argument(
        "--auto-chunks-per-grad-min",
        type=int,
        default=1,
        help="Minimum chunks-per-grad when --auto-chunks-per-grad is enabled (default: 1)",
    )
    parser.add_argument(
        "--auto-chunks-per-grad-vram-frac",
        type=float,
        default=0.95,
        help=(
            "Hard dedicated VRAM cap during the probe; reject immediately when exceeded "
            "(default: 0.95)"
        ),
    )
    parser.add_argument(
        "--auto-chunks-per-grad-vram-target-frac",
        type=float,
        default=0.90,
        help=(
            "Target dedicated VRAM for the startup probe; search accepts settings at "
            "or below this fraction (default: 0.90)"
        ),
    )
    parser.add_argument(
        "--auto-pixel-chunk-size-min",
        type=int,
        default=None,
        help=(
            "Minimum pixel_chunk_size when auto-tuning the chunk pipeline "
            "(default: max(1000, requested pixel_chunk_size // 2))"
        ),
    )
    parser.add_argument(
        "--auto-pixel-chunk-size-step",
        type=int,
        default=100,
        help=(
            "Step size for pixel_chunk_size during auto chunk pipeline search "
            "(default: 100)"
        ),
    )
    parser.add_argument(
        "--auto-chunks-per-grad-narrow-threshold",
        type=int,
        default=32,
        help=(
            "When the binary search window has this many candidates or fewer, "
            "probe all of them (default: 32)"
        ),
    )
    parser.add_argument(
        "--auto-chunks-per-grad-refine-radius",
        type=int,
        default=24,
        help=(
            "After binary search, probe up to this many candidates on each side "
            "of the best fit (default: 24)"
        ),
    )
    args = parser.parse_args()

    from training_runtime import DEFAULTS

    for key, value in DEFAULTS.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
    if args.batch_empty_cache:
        args.skip_batch_empty_cache = False

    args.dataset = "USCropsAggregatedNPY"
    args.datapath = (
        Path(args.datapath).expanduser()
        if args.datapath
        else resolve_datapath()
    )
    args.yield_csv = Path(args.yield_csv).expanduser() if args.yield_csv else YIELD_CSV
    args.target_column, args.aggregation, args.target_unit = resolve_target(args.target)

    # WSL/Linux convenience: convert Windows absolute paths (e.g. C:\...) to /mnt/c/...
    # This allows passing the same checkpoint path from Windows into WSL.
    if args.pretrained and os.name != "nt":
        p = str(args.pretrained)
        if len(p) >= 3 and p[1:3] == ":\\":
            wp = PureWindowsPath(p)
            drive = (wp.drive or "").rstrip(":").lower()
            args.pretrained = str(Path("/mnt") / drive / Path(*wp.parts[1:]))

    args.harvest_years_set = {
        int(x.strip()) for x in args.harvest_years.split(",") if x.strip()
    }
    if not args.harvest_years_set:
        raise ValueError("--harvest-years must contain at least one year")

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


def find_available_years(datapath):
    """Find all years for which we have imagery data.
    Accepts integer year dirs (2023) and year-range dirs (2022-2023); range maps to end year.
    """
    datapath = Path(datapath)
    available_years = []

    if not datapath.exists():
        return available_years

    for item in datapath.iterdir():
        if not item.is_dir():
            continue
        if not any(item.rglob("*.npy")):
            continue
        if item.name.isdigit():
            available_years.append(int(item.name))
        elif "-" in item.name:
            parts = item.name.split("-")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                available_years.append(
                    int(parts[1])
                )  # end year (e.g. 2022-2023 -> 2023)

    return sorted(set(available_years))


def infer_yield_csv_split_design(yield_csv: Path) -> dict:
    """
    Infer holdout vs train years from the yield CSV split column.

    Holdout years = years with valid/test rows but no train rows (see
    add_train_valid_test_split.py).
    """
    yield_df = pl.read_csv(
        yield_csv,
        null_values=["-", "", "nan", "NaN", "null", "NULL"],
        infer_schema_length=10000,
    )
    if "split" not in yield_df.columns:
        raise ValueError(
            "Yield CSV has no 'split' column. Run add_train_valid_test_split.py first."
        )
    if "year" not in yield_df.columns:
        raise ValueError("Yield CSV has no 'year' column.")

    train_years = sorted(
        int(y)
        for y in yield_df.filter(pl.col("split") == "train")["year"].unique().to_list()
        if y is not None
    )
    eval_years: set[int] = set()
    for split in ("valid", "test"):
        for y in yield_df.filter(pl.col("split") == split)["year"].unique().to_list():
            if y is not None:
                eval_years.add(int(y))
    train_year_set = set(train_years)
    holdout_years = sorted(y for y in eval_years if y not in train_year_set)

    return {
        "holdout_years": holdout_years,
        "train_years_in_csv": train_years,
        "eval_years_in_csv": sorted(eval_years),
    }


def split_design_for_config(
    split_design: dict,
    harvest_years_set: set[int] | None,
) -> dict:
    """Augment split design with how --harvest-years intersects holdout years."""
    holdout = split_design["holdout_years"]
    if harvest_years_set is None:
        return {
            **split_design,
            "harvest_years_filter": None,
            "holdout_years_in_harvest_filter": holdout,
            "holdout_years_excluded_by_harvest_filter": [],
        }
    hset = set(harvest_years_set)
    return {
        **split_design,
        "harvest_years_filter": sorted(hset),
        "holdout_years_in_harvest_filter": sorted(set(holdout) & hset),
        "holdout_years_excluded_by_harvest_filter": sorted(set(holdout) - hset),
    }


def compute_target_statistics(yield_csv, datapath, target_column, harvest_years=None):
    """
    Compute mean and std of training targets for normalization.
    Only uses training data from years where we have imagery data.
    Automatically excludes test years (marked as 'test' in split column).

    Args:
        yield_csv: Path to yield CSV file
        datapath: Path to dataset directory (to find available years)
        harvest_years: Optional set of harvest years; intersects with years that have imagery.
    """

    print(f"Computing target statistics from {yield_csv}...")
    yield_df = pl.read_csv(
        yield_csv,
        null_values=["-", "", "nan", "NaN", "null", "NULL"],
        infer_schema_length=10000,
    )

    yield_col = target_column
    if yield_col not in yield_df.columns:
        hint = ""
        if yield_col == "production_t_s2_adj":
            hint = " Run scripts/compute_imagery_area_coverage.py first."
        raise ValueError(
            f"Yield CSV missing {yield_col!r}.{hint} Available: {list(yield_df.columns)}"
        )

    available_years = find_available_years(datapath)
    if not available_years:
        raise ValueError(
            f"No available years found in {datapath}. Check that year subdirectories exist."
        )

    if harvest_years is not None:
        hset = set(harvest_years)
        available_years = sorted(set(available_years) & hset)
        if not available_years:
            raise ValueError(
                f"No overlap between --harvest-years and imagery under {datapath}."
            )
        print(f"  Restricted to harvest years (with imagery): {available_years}")
    else:
        print(f"  Found imagery data for years: {available_years}")

    train_df = yield_df

    if "year" in yield_df.columns:
        train_df = train_df.filter(pl.col("year").is_in(available_years))
        print(f"  Filtered to available years: {len(train_df)} rows")

    if "split" not in yield_df.columns:
        raise ValueError(
            "Yield CSV has no 'split' column. Run add_train_valid_test_split.py first."
        )
    train_df = train_df.filter(pl.col("split") == "train")
    print(f"  Filtered to train split: {len(train_df)} rows")

    valid_targets = train_df.select(yield_col).drop_nulls()[yield_col].to_list()
    valid_targets = [float(t) for t in valid_targets if t is not None and t != ""]

    if len(valid_targets) == 0:
        raise ValueError("No valid training targets found!")

    target_mean = float(np.mean(valid_targets))
    target_std = float(np.std(valid_targets))

    # Avoid division by zero
    if target_std < 1e-6:
        target_std = 1.0
        print("  ⚠️  Warning: target_std is very small, using 1.0")

    print(f"  Target statistics: mean={target_mean:.2f}, std={target_std:.2f}")
    print(f"  Target range: min={min(valid_targets):.2f}, max={max(valid_targets):.2f}")
    print(f"  Number of training samples used: {len(valid_targets)}")

    return target_mean, target_std


def _json_sanitize(obj):
    """Make values safe for JSON (paths, sets, numpy scalars)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj.resolve())
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, frozenset):
        return sorted(obj)
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)


CONFIG_JSON_VERSION = 1


def apply_cli_optimizer_settings(optimizer, args) -> None:
    """After loading a checkpoint, CLI lr/weight_decay must win over saved param_groups."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = args.learning_rate
        param_group["weight_decay"] = args.weight_decay


def checkpoint_optimizer_hparams(checkpoint: dict) -> dict | None:
    try:
        pg = checkpoint["optimizer_state"]["param_groups"][0]
        return {
            "learning_rate": float(pg["lr"]),
            "weight_decay": float(pg.get("weight_decay", 0.0)),
        }
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def cli_optimizer_hparams(args) -> dict:
    return {
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
    }


def model_kwargs_from_args(args) -> dict:
    """STNetRegression constructor kwargs from CLI."""
    return {
        "d_model": int(args.model_d_model),
        "n_head": int(args.model_n_head),
        "n_layers": int(args.model_n_layers),
        "d_inner": int(args.model_d_inner),
        "dropout": float(args.model_dropout),
    }


def optimizer_hparams_changed(ck: dict | None, args) -> bool:
    if ck is None:
        return False
    cli = cli_optimizer_hparams(args)
    if abs(ck["learning_rate"] - cli["learning_rate"]) > 1e-12:
        return True
    return abs(ck["weight_decay"] - cli["weight_decay"]) > 1e-12


def _args_to_cli_dict(args) -> dict:
    cli = {}
    for k in sorted(vars(args).keys()):
        if k.startswith("_"):
            continue
        try:
            cli[k] = _json_sanitize(getattr(args, k))
        except Exception:
            cli[k] = str(getattr(args, k))
    return cli


def append_training_config_session(
    training_dir: Path,
    args,
    *,
    computed: dict,
    run_paths: dict,
    resume: dict | None = None,
    torch_compiled: bool,
) -> None:
    """
    Append one training session record to training/config.json.

    Each invocation (fresh run or resume) adds a session so continuation runs
    keep a full audit trail of CLI + computed state at start time.
    """
    config_path = Path(training_dir) / "config.json"
    sessions = load_training_sessions(config_path)
    session_index = len(sessions)
    kind = "resume" if resume is not None else "initial"

    session = {
        "session_index": session_index,
        "kind": kind,
        "cli": _args_to_cli_dict(args),
        "computed": _json_sanitize(computed),
        "run_paths": _json_sanitize(run_paths),
        "resume": _json_sanitize(resume) if resume is not None else None,
        "environment": {
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "argv": list(sys.argv),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "polars": pl.__version__,
            "torch_compiled": bool(torch_compiled),
        },
    }
    sessions.append(session)

    payload = {"version": CONFIG_JSON_VERSION, "sessions": sessions}
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(
        f"Wrote training config session {session_index} ({kind}) to {config_path} "
        f"({len(sessions)} session(s) total)"
    )


def train(args):
    split_design = split_design_for_config(
        infer_yield_csv_split_design(args.yield_csv),
        args.harvest_years_set,
    )
    holdout_years = split_design["holdout_years"]
    print(f"Yield CSV holdout year(s) (valid/test only): {holdout_years}")
    excluded = split_design["holdout_years_excluded_by_harvest_filter"]
    if excluded:
        print(
            f" --harvest-years excludes holdout {excluded}; "
            "valid/test dataloaders will be empty unless those years are included."
        )

    target_mean, target_std = compute_target_statistics(
        args.yield_csv,
        args.datapath,
        args.target_column,
        harvest_years=args.harvest_years_set,
    )
    print(
        f"Target: {args.target} ({args.target_column}, {args.aggregation} over pixels)"
    )

    print("=> creating dataloader (Polars-optimized)")
    from training_runtime import dataloader_options_from_args

    dl_opts = dataloader_options_from_args(args)
    traindataloader, train_meta = get_aggregated_dataloader(
        args.datapath,
        args.yield_csv,
        args.batchsize,
        args.workers,
        args.sequencelength,
        args.rc,
        args.interp,
        args.seed,
        mode="train",
        sample_ratio=args.sample_ratio,
        target_mean=target_mean,
        target_std=target_std,
        harvest_years=args.harvest_years_set,
        feature_layout=args.feature_layout,
        target_column=args.target_column,
        **dl_opts,
    )
    valdataloader, val_meta = get_aggregated_dataloader(
        args.datapath,
        args.yield_csv,
        args.batchsize,
        args.workers,
        args.sequencelength,
        args.rc,
        args.interp,
        args.seed,
        mode="valid",
        target_mean=target_mean,
        target_std=target_std,
        harvest_years=args.harvest_years_set,
        feature_layout=args.feature_layout,
        target_column=args.target_column,
        **dl_opts,
    )
    testdataloader, test_meta = get_aggregated_dataloader(
        args.datapath,
        args.yield_csv,
        args.batchsize,
        args.workers,
        args.sequencelength,
        args.rc,
        args.interp,
        args.seed,
        mode="test",
        target_mean=target_mean,
        target_std=target_std,
        harvest_years=args.harvest_years_set,
        feature_layout=args.feature_layout,
        target_column=args.target_column,
        **dl_opts,
    )

    print("=> creating model")
    device = torch.device(args.device)
    from models import STNetRegression

    input_dim = int(traindataloader.dataset.input_feature_dim)
    print(
        f"Model input_dim={input_dim} (--feature-layout {args.feature_layout}; "
        "see datasets/feature_layout.py)"
    )
    model_kw = model_kwargs_from_args(args)
    model = STNetRegression(
        input_dim=input_dim,
        num_outputs=1,
        max_seq_len=args.sequencelength,
        **model_kw,
    ).to(device)
    print("Model architecture: " + ", ".join(f"{k}={v}" for k, v in model_kw.items()))

    print(
        f"Initialized {model.modelname}: Total trainable parameters: {get_ntrainparams(model)}"
    )
    # Use regression-specific initialization with smaller output layer weights
    model.apply(weight_init_regression)

    # Load pretrained model weights (will load full state later if resuming)
    pretrained_checkpoint = None
    if args.pretrained:
        pretrained_path = Path(args.pretrained)
        print(f"Loading pretrained model from {pretrained_path}")
        pretrained_checkpoint = torch.load(
            pretrained_path, map_location=device, weights_only=False
        )

        # Load all weights including decoder (for regression checkpoints)
        pretrain_state = pretrained_checkpoint["model_state"]
        model_dict = model.state_dict()

        state_dict = {k: v for k, v in pretrain_state.items() if k in model_dict.keys()}
        missing_keys = set(model_dict.keys()) - set(state_dict.keys())
        unexpected_keys = set(state_dict.keys()) - set(model_dict.keys())

        if missing_keys:
            print(
                f"  ⚠️  Missing keys (will use initialized values): {list(missing_keys)[:5]}..."
            )
        if unexpected_keys:
            print(
                f"  ⚠️  Unexpected keys (will be ignored): {list(unexpected_keys)[:5]}..."
            )

        load_result = model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys:
            print(f"  ⚠️  Missing keys after load: {load_result.missing_keys[:5]}...")
        if load_result.unexpected_keys:
            print(
                f"  ⚠️  Unexpected keys after load: {load_result.unexpected_keys[:5]}..."
            )
        print(f"  ✓ Loaded {len(state_dict)}/{len(model_dict)} model parameters")

        decoder_keys = [k for k in state_dict.keys() if "decoder" in k]
        if decoder_keys:
            print(f"  ✓ Loaded decoder weights: {len(decoder_keys)} parameters")
        else:
            print("  ⚠️  Warning: No decoder weights found in checkpoint!")

    model.modelname = build_run_name(
        target=args.target,
        model_class_name=model.__class__.__name__,
        harvest_years_set=args.harvest_years_set,
        seed=args.seed,
        suffix=args.suffix,
    )

    # Compile model for faster execution (PyTorch 2.0+)
    # Wrap in try-except in case compilation fails (e.g., missing Python headers in WSL)
    torch_compiled = False
    if hasattr(torch, "compile") and not args.no_compile:
        try:
            from training_runtime import compile_model, compile_mode

            print(f"Compiling model with torch.compile(mode={compile_mode(args)!r})...")
            model = compile_model(model, args)
            torch_compiled = True
            print("Model compilation successful!")
        except Exception as e:
            print(
                f"⚠️  Warning: torch.compile() failed ({type(e).__name__}), continuing without compilation"
            )
            print(f"   Error: {str(e)[:200]}...")
            # Continue with uncompiled model

    run_dir = run_dir_for_experiment(args.logdir, model.modelname, args.feature_layout)
    experiment_dir_path = experiment_dir(args.logdir, model.modelname)
    training_dir_path, figures_dir_path, predictions_dir_path = ensure_run_layout(
        run_dir
    )
    best_model_path = training_dir_path / "model_best.pth"
    if best_model_path.is_file() and args.pretrained is None and not args.overwrite_run:
        raise SystemExit(
            f"Refusing to overwrite existing run: {run_dir}\n"
            f"  Found {best_model_path}\n"
            "Use a different --seed (or other run name), pass --pretrained to resume, "
            "or pass --overwrite-run to start fresh."
        )
    print(f"Experiment directory: {experiment_dir_path}")
    print(f"Run directory ({args.feature_layout}): {run_dir}")
    print(f"  training:    {training_dir_path}")
    print(f"  predictions: {predictions_dir_path}")
    print(f"  figures:     {figures_dir_path}")

    # Pass normalization stats for denormalization in evaluation
    criterion = AggregatedMSELoss(
        target_mean=target_mean,
        target_std=target_std,
        aggregation=args.aggregation,
        target_column=args.target_column,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # Initialize training state
    start_epoch = 0
    log = list()
    val_loss_min = np.inf
    not_improved_count = 0

    resume_info = None
    pretrained_path_resolved = None
    ck_optimizer_hparams = None
    not_improved_count_reset = False
    if args.pretrained:
        pretrained_path_resolved = Path(args.pretrained).resolve()

    # Resume from checkpoint if pretrained model is provided
    if pretrained_checkpoint is not None:
        checkpoint = pretrained_checkpoint
        print(f"Resuming training from checkpoint...")

        ck_optimizer_hparams = checkpoint_optimizer_hparams(checkpoint)
        loaded_optimizer = False
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            apply_cli_optimizer_settings(optimizer, args)
            loaded_optimizer = True
            print(
                "  ✓ Loaded optimizer state (Adam momentum); CLI lr/weight_decay applied"
            )
            cli_opt = cli_optimizer_hparams(args)
            if optimizer_hparams_changed(ck_optimizer_hparams, args):
                print(
                    "  ✓ Optimizer hparams from CLI: "
                    f"lr={cli_opt['learning_rate']:.2e}, "
                    f"weight_decay={cli_opt['weight_decay']:.2e} "
                    f"(checkpoint lr={ck_optimizer_hparams['learning_rate']:.2e}, "
                    f"weight_decay={ck_optimizer_hparams['weight_decay']:.2e})"
                )
            else:
                print(
                    f"  ✓ Optimizer hparams: lr={cli_opt['learning_rate']:.2e}, "
                    f"weight_decay={cli_opt['weight_decay']:.2e}"
                )

        if "epoch" in checkpoint:
            start_epoch = int(checkpoint["epoch"])
            print(f"  ✓ Resuming from epoch {start_epoch}")

        if "val_loss_min" in checkpoint:
            val_loss_min = float(checkpoint["val_loss_min"])
            print(f"  ✓ Loaded best validation loss: {val_loss_min:.4f}")
        elif "val_loss" in checkpoint:
            val_loss_min = float(checkpoint["val_loss"])
            print(f"  ✓ Loaded validation loss: {val_loss_min:.4f}")

        loaded_not_improved_from_ck = None
        if "not_improved_count" in checkpoint:
            loaded_not_improved_from_ck = int(checkpoint["not_improved_count"])
            reference_opt = (
                initial_session_optimizer_hparams(training_dir_path)
                or ck_optimizer_hparams
            )
            if optimizer_hparams_changed(reference_opt, args):
                not_improved_count = 0
                not_improved_count_reset = True
                print(
                    f"  ✓ Reset not_improved_count (checkpoint had {loaded_not_improved_from_ck}) "
                    "because optimizer hparams changed vs initial training session"
                )
            else:
                not_improved_count = loaded_not_improved_from_ck
                print(f"  ✓ Loaded not_improved_count: {not_improved_count}")

        ck_target_mean = checkpoint.get("target_mean")
        ck_target_std = checkpoint.get("target_std")
        normalization_stats_match = None
        ck_target_column = checkpoint.get("target_column")
        if ck_target_column is not None and ck_target_column != args.target_column:
            raise ValueError(
                f"Checkpoint target_column={ck_target_column!r} does not match "
                f"--target {args.target!r} ({args.target_column!r})"
            )
        ck_aggregation = checkpoint.get("aggregation")
        if ck_aggregation is not None and ck_aggregation != args.aggregation:
            raise ValueError(
                f"Checkpoint aggregation={ck_aggregation!r} does not match "
                f"--target {args.target!r} ({args.aggregation!r})"
            )

        if ck_target_mean is not None and ck_target_std is not None:
            ck_target_mean = float(ck_target_mean)
            ck_target_std = float(ck_target_std)
            normalization_stats_match = (
                abs(ck_target_mean - target_mean) <= 1e-3
                and abs(ck_target_std - target_std) <= 1e-3
            )
            if not normalization_stats_match:
                print(f"  ⚠️  Warning: Normalization stats differ!")
                print(
                    f"      Checkpoint: mean={ck_target_mean:.2f}, std={ck_target_std:.2f}"
                )
                print(f"      This run:   mean={target_mean:.2f}, std={target_std:.2f}")
            else:
                print(f"  ✓ Normalization stats match")

        trainlog_file = trainlog_path(run_dir)
        trainlog_entries_loaded = 0
        if trainlog_file.exists():
            try:
                existing_log_df = pd.read_csv(trainlog_file, index_col="epoch")
                log = existing_log_df.reset_index().to_dict("records")
                for entry in log:
                    ep = entry.get("epoch")
                    if ep is None or ep == "test":
                        continue
                    try:
                        entry["epoch"] = int(ep)
                    except (ValueError, TypeError):
                        pass
                trainlog_entries_loaded = len(log)
                print(
                    f"  ✓ Loaded existing training log with {trainlog_entries_loaded} entries"
                )
            except Exception as e:
                print(f"  ⚠️  Could not load existing log: {e}, starting fresh")
        else:
            print("  ℹ️  No existing training log found, starting fresh")

        es_state = early_stop_state_from_trainlog(trainlog_file)
        if es_state is not None:
            val_loss_min = es_state["val_loss_min"]
            not_improved_count = es_state["not_improved_count"]
            print(
                f"  ✓ Early-stop state from trainlog: best epoch {es_state['best_epoch']}, "
                f"val_loss_min={val_loss_min:.6g}, "
                f"not_improved={not_improved_count} "
                f"(last completed epoch {es_state['last_epoch']})"
            )

        resume_info = {
            "from_checkpoint": str(pretrained_path_resolved),
            "resumed_from_epoch": int(start_epoch),
            "loaded_optimizer_state": loaded_optimizer,
            "loaded_val_loss_min": (
                float(val_loss_min) if np.isfinite(val_loss_min) else None
            ),
            "loaded_not_improved_count": loaded_not_improved_from_ck,
            "not_improved_count_after_resume": int(not_improved_count),
            "checkpoint_target_mean": ck_target_mean,
            "checkpoint_target_std": ck_target_std,
            "checkpoint_feature_layout": checkpoint.get("feature_layout"),
            "checkpoint_target": checkpoint.get("target"),
            "checkpoint_target_column": ck_target_column,
            "checkpoint_aggregation": ck_aggregation,
            "normalization_stats_match": normalization_stats_match,
            "trainlog_path": str(trainlog_file.resolve()),
            "trainlog_entries_loaded": trainlog_entries_loaded,
            "checkpoint_optimizer_hparams": ck_optimizer_hparams,
            "cli_optimizer_hparams": cli_optimizer_hparams(args),
            "optimizer_hparams_overridden_from_cli": optimizer_hparams_changed(
                ck_optimizer_hparams, args
            ),
            "not_improved_count_reset": not_improved_count_reset,
        }

    def _make_optimizer():
        return torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    if args.auto_chunks_per_grad and args.pretrained is not None:
        print(
            "Skipping auto chunks_per_grad probe (pretrained/resume run); "
            f"using chunks_per_grad={args.chunks_per_grad}"
        )
    elif args.auto_chunks_per_grad:
        from training.vram_adjuster import resolve_chunks_per_grad_before_training

        optimizer = resolve_chunks_per_grad_before_training(
            args=args,
            device=device,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            traindataloader=traindataloader,
            target_mean=target_mean,
            target_std=target_std,
            make_optimizer=_make_optimizer,
        )

    chunk_pipeline_config = None
    resolver = getattr(args, "_chunks_per_grad_resolver", None)
    if resolver is not None and resolver.enabled:
        chunk_pipeline_config = resolver.to_config_dict()
    else:
        chunk_pipeline_config = {
            "auto_chunks_per_grad": bool(args.auto_chunks_per_grad),
            "chunks_per_grad_requested": int(args.chunks_per_grad),
            "chunks_per_grad_effective": int(args.chunks_per_grad),
            "pixel_chunk_size_requested": int(args.pixel_chunk_size),
            "pixel_chunk_size_effective": int(args.pixel_chunk_size),
            "max_pixels_per_backward_requested": max(
                1, int(args.chunks_per_grad)
            )
            * max(1, int(args.pixel_chunk_size)),
            "max_pixels_per_backward_effective": max(
                1, int(args.chunks_per_grad)
            )
            * max(1, int(args.pixel_chunk_size)),
            "search_trials": [],
        }

    imagery_years_available = find_available_years(args.datapath)
    append_training_config_session(
        training_dir_path,
        args,
        computed={
            "model_run_name": model.modelname,
            "experiment_dir": str(experiment_dir_path.resolve()),
            "feature_layout": args.feature_layout,
            "input_dim": int(input_dim),
            "target": args.target,
            "yield_target_column": args.target_column,
            "municipality_aggregation": args.aggregation,
            "target_unit": args.target_unit,
            "target_mean": float(target_mean),
            "target_std": float(target_std),
            "num_train_municipalities": len(traindataloader.dataset),
            "num_valid_municipalities": len(valdataloader.dataset),
            "num_test_municipalities": len(testdataloader.dataset),
            "dataloader_meta": {
                "train": train_meta,
                "valid": val_meta,
                "test": test_meta,
            },
            "imagery_years_available": imagery_years_available,
            "imagery_years_used": sorted(args.harvest_years_set),
            "split_design": split_design,
            "model_kwargs": model_kwargs_from_args(args),
            "early_stop_patience": int(args.early_stop_patience),
            "chunk_pipeline": chunk_pipeline_config,
            "epoch_plan": {
                "start_epoch": int(start_epoch),
                "end_epoch_exclusive": int(args.epochs),
                "num_epochs_this_session": max(0, int(args.epochs) - int(start_epoch)),
            },
        },
        run_paths={
            "experiment_dir": experiment_dir_path,
            "run_dir": run_dir,
            "training_dir": training_dir_path,
            "predictions_dir": predictions_dir_path,
            "figures_dir": figures_dir_path,
            "best_model_path": best_model_path,
        },
        resume=resume_info,
        torch_compiled=torch_compiled,
    )

    print(f"Training {model.modelname}...")
    from training.pipeline import describe_chunk_pipeline

    print(describe_chunk_pipeline(args, device))

    if (
        pretrained_checkpoint is not None
        and not_improved_count >= args.early_stop_patience
    ):
        print(
            f"\nEarly-stop patience ({args.early_stop_patience}) already reached "
            f"(not_improved={not_improved_count}). Skipping training; testing model_best."
        )
        epoch = start_epoch - 1
    else:
        for epoch in range(start_epoch, args.epochs):
            # Update sampler epoch for different shuffles each epoch
            if hasattr(traindataloader.sampler, "set_epoch"):
                traindataloader.sampler.set_epoch(epoch)

            if args.warmup_epochs > 0:
                if epoch == 0:
                    lr = args.learning_rate * 0.1
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr
                elif epoch == args.warmup_epochs:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = args.learning_rate

            if args.schedule is not None:
                adjust_learning_rate(optimizer, epoch, args)

            train_loss = train_epoch_aggregated(
                model,
                optimizer,
                criterion,
                traindataloader,
                device,
                args,
                target_mean,
                target_std,
            )
            val_loss, scores = test_epoch_aggregated(
                model, criterion, valdataloader, device, args, target_mean, target_std
            )

            scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in scores.items()])
            print(
                f"epoch {epoch + 1}: trainloss={train_loss:.4f}, valloss={val_loss:.4f} "
                + scores_msg
            )

            if val_loss < val_loss_min:
                not_improved_count = 0
                save(
                    model,
                    path=best_model_path,
                    criterion=criterion,
                    optimizer_state=optimizer.state_dict(),
                    epoch=epoch + 1,
                    val_loss=val_loss,
                    train_loss=train_loss,
                    val_loss_min=val_loss,
                    not_improved_count=not_improved_count,
                    target_mean=target_mean,
                    target_std=target_std,
                    feature_layout=args.feature_layout,
                    target=args.target,
                    target_column=args.target_column,
                    aggregation=args.aggregation,
                )
                val_loss_min = val_loss
                print(f"lowest val loss in epoch {epoch + 1}\n")
            else:
                not_improved_count += 1

            scores["epoch"] = epoch + 1
            scores["trainloss"] = train_loss
            scores["valloss"] = val_loss
            log.append(scores)

            log_df = pd.DataFrame(log).set_index("epoch")
            log_df.to_csv(training_dir_path / "trainlog.csv", float_format="%.6g")

            if (epoch + 1) % 1 == 0:
                checkpoint_path = training_dir_path / f"checkpoint_epoch_{epoch + 1}.pth"
                save(
                    model,
                    path=checkpoint_path,
                    criterion=criterion,
                    optimizer_state=optimizer.state_dict(),
                    epoch=epoch + 1,
                    val_loss=val_loss,
                    train_loss=train_loss,
                    val_loss_min=val_loss_min,
                    not_improved_count=not_improved_count,
                    target_mean=target_mean,
                    target_std=target_std,
                    feature_layout=args.feature_layout,
                    target=args.target,
                    target_column=args.target_column,
                    aggregation=args.aggregation,
                )
                print(f"Saved checkpoint at epoch {epoch + 1} to {checkpoint_path.name}")

            if not_improved_count >= args.early_stop_patience:
                print(
                    f"\nValidation performance didn't improve for "
                    f"{args.early_stop_patience} epochs. Training stops."
                )
                break

    if epoch == args.epochs - 1:
        print(f"\n{args.epochs} epochs training finished.")

    # Test
    print("Restoring best model weights for testing...")
    checkpoint = torch.load(best_model_path, weights_only=False)
    state_dict = {k: v for k, v in checkpoint["model_state"].items()}
    criterion = checkpoint["criterion"]
    torch.save(
        {
            "model_state": state_dict,
            "criterion": criterion,
            "feature_layout": checkpoint.get("feature_layout"),
            "target": checkpoint.get("target"),
            "target_column": checkpoint.get("target_column"),
            "aggregation": checkpoint.get("aggregation"),
            "target_mean": checkpoint.get("target_mean"),
            "target_std": checkpoint.get("target_std"),
        },
        best_model_path,
    )

    # Handle compiled models - load into underlying model if compiled
    if hasattr(model, "_orig_mod"):
        model._orig_mod.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    test_loss, scores = test_epoch_aggregated(
        model, criterion, testdataloader, device, args, target_mean, target_std
    )
    scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in scores.items()])
    print(f"Test results: \n\n {scores_msg}")

    scores["epoch"] = "test"
    scores["testloss"] = test_loss

    log_df = pd.DataFrame([scores]).set_index("epoch")
    log_df.to_csv(training_dir_path / "testlog.csv")

    return run_dir


def get_aggregated_dataloader(
    datapath,
    yield_csv,
    batchsize,
    workers,
    sequencelength,
    rc,
    interp,
    seed,
    mode,
    sample_ratio=1.0,
    target_mean=None,
    target_std=None,
    harvest_years=None,
    feature_layout: str = "spectral",
    target_column: str = "production_t",
    *,
    npy_cache_size: int = 20,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: int | None = 4,
):
    """Create dataloader for aggregated regression (Polars-optimized)."""
    dataset = USCropsAggregatedNPY(
        mode=mode,
        root=datapath,
        yield_csv=yield_csv,
        year=None,
        sequencelength=sequencelength,
        dataaug=None,
        randomchoice=rc,
        interp=interp,
        seed=seed,
        preload_ram=False,
        target_mean=target_mean,
        target_std=target_std,
        harvest_years=harvest_years,
        feature_layout=feature_layout,
        target_column=target_column,
        npy_cache_size=npy_cache_size,
    )

    # Use QueueSampler for training if sample_ratio < 1.0
    sampler = None
    shuffle = False
    if mode == "train" and sample_ratio < 1.0:
        from datasets.queue_sampler import QueueSampler

        sampler = QueueSampler(
            dataset,
            sample_ratio=sample_ratio,
            seed=seed,
        )
        print(
            f"Using QueueSampler: {sampler.samples_per_epoch}/{len(dataset)} municipalities per epoch ({sample_ratio*100:.1f}%)"
        )
    elif mode == "train":
        shuffle = True  # Use standard shuffle if using all data

    loader_kwargs: dict = {
        "dataset": dataset,
        "batch_size": batchsize,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "collate_fn": aggregated_collate_fn,
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor or 2))
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
    dataloader = DataLoader(**loader_kwargs)

    meta = dict(
        ndims=dataset.input_feature_dim,
        num_outputs=1,
        num_municipalities=len(dataset),
        use_xavier=dataset.use_xavier,
        feature_layout=dataset.feature_layout,
    )

    return dataloader, meta


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    from training_runtime import enable_cuda_training_kernels

    enable_cuda_training_kernels()


def main():
    args = parse_args()
    print(
        " ===================== Training ONE model on ALL train data ======================= "
    )
    print(f"Harvest years: {sorted(args.harvest_years_set)}")
    print(f"Using seed {args.seed} (from --seed)")
    set_global_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
