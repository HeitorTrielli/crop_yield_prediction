"""
Main training script for municipality-level yield prediction using aggregated regression.
Polars-optimized version for faster index loading.

Pixel-level predictions are summed per municipality and compared to municipality-level targets.
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
from utils import (
    AverageMeter,
    adjust_learning_rate,
    get_ntrainparams,
    recursive_todevice,
    save,
)
from run_paths import ensure_run_layout, trainlog_path
from utils_aggregated import (
    AggregatedMSELoss,
    aggregated_collate_fn,
    regression_metrics,
    stnet_regression_input_dim_from_state_dict,
    test_epoch_aggregated,
    train_epoch_aggregated,
)

# Default paths
DATAPATH = Path(r"files/npy")
YIELD_CSV = Path("files/pam_soy_pr_2019_2025.csv")
YIELD_TARGET_COLUMN = "production_t"  # production_t → Total; yield_t_ha → Productivity
DEFAULT_REGRESSION_MODEL = "STNetRegression"
YEARS = [None]
SEEDS = [6007]


def target_run_prefix(target_column: str = YIELD_TARGET_COLUMN) -> str:
    if target_column == "production_t":
        return "Total"
    if target_column == "yield_t_ha":
        return "Productivity"
    raise ValueError(
        f"Unknown yield target {target_column!r}; expected 'production_t' or 'yield_t_ha'"
    )


def format_years_for_run_name(
    harvest_years_set: set[int] | None,
    single_year: int | None = None,
) -> str:
    if harvest_years_set:
        return "_".join(f"{y % 100:02d}" for y in sorted(harvest_years_set))
    if single_year is not None:
        return f"{int(single_year) % 100:02d}"
    return "AllYears"


def build_run_name(
    *,
    target_column: str = YIELD_TARGET_COLUMN,
    model_class_name: str = DEFAULT_REGRESSION_MODEL,
    harvest_years_set: set[int] | None,
    year: int | None,
    seed: int,
    suffix: str | None = None,
) -> str:
    parts = [target_run_prefix(target_column)]
    if model_class_name != DEFAULT_REGRESSION_MODEL:
        parts.append(model_class_name)
    parts.append(format_years_for_run_name(harvest_years_set, year))
    parts.append(f"Seed{seed}")
    if suffix:
        parts.append(suffix)
    return "_".join(parts)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train yield prediction model using aggregated regression (Polars-optimized)."
    )
    parser.add_argument(
        "--use-doy", action="store_true", help="whether to use doy pe with transformer"
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
        "--year", type=int, default=2023, help="year of dataset (default: 2023)"
    )
    parser.add_argument(
        "-seq",
        "--sequencelength",
        type=int,
        default=45,
        help="Max time steps per pixel (pad/sample to this). Match your data: ~11 for daily with few dates; larger values use more GPU memory (default: 15).",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=4,
        help="number of CPU workers to load the next batch (default: 8)",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=100, help="number of training epochs"
    )
    parser.add_argument(
        "-b",
        "--batchsize",
        type=int,
        default=4,
        help="batch size (number of municipalities per batch, default: 16)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=1e-6,
        help="optimizer learning rate (default 1e-3)",
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
        "--datapath", type=str, default=None, help="path to dataset root directory"
    )
    parser.add_argument(
        "--yield-csv",
        type=str,
        default=None,
        help=f"path to yield CSV (default: {YIELD_CSV})",
    )
    parser.add_argument("--seed", type=int, default=5000, help="random seed")
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="Fraction of municipalities to sample per epoch (0.0-1.0, default: 1.0 = use all)",
    )
    parser.add_argument(
        "--harvest-years",
        type=str,
        default=None,
        help=(
            "Comma-separated harvest years for multi-season training, e.g. 2019,2020,2022,2023,2024 "
            "to use those seasons only (omit 2021 to hold out that harvest). Requires a 'year' column "
            "in the yield CSV (harvest year). Folder Y1-Y2 under --datapath maps to harvest year Y2. "
            "Implies multi-year mode; --year is ignored when this is set."
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
    args = parser.parse_args()

    args.dataset = "USCropsAggregatedNPY"
    args.datapath = Path(args.datapath) if args.datapath else DATAPATH
    args.yield_csv = Path(args.yield_csv) if args.yield_csv else YIELD_CSV

    # WSL/Linux convenience: convert Windows absolute paths (e.g. C:\...) to /mnt/c/...
    # This allows passing the same checkpoint path from Windows into WSL.
    if args.pretrained and os.name != "nt":
        p = str(args.pretrained)
        if len(p) >= 3 and p[1:3] == ":\\":
            wp = PureWindowsPath(p)
            drive = (wp.drive or "").rstrip(":").lower()
            args.pretrained = str(Path("/mnt") / drive / Path(*wp.parts[1:]))

    if args.harvest_years:
        args.harvest_years_set = {
            int(x.strip()) for x in args.harvest_years.split(",") if x.strip()
        }
    else:
        args.harvest_years_set = None

    if args.harvest_years_set:
        if args.year is not None:
            print(
                "Note: --harvest-years is set; using multi-year mode (--year ignored)."
            )
        args.year = None

    if args.use_doy:
        if args.suffix:
            args.suffix = "doy_" + args.suffix
        else:
            args.suffix = "doy"

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


def compute_target_statistics(yield_csv, datapath, harvest_years=None):
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

    yield_col = "production_t"
    if yield_col not in yield_df.columns:
        raise ValueError(
            f"Yield CSV missing {yield_col!r}. Available: {list(yield_df.columns)}"
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


def _load_config_sessions(config_path: Path) -> list[dict]:
    if not config_path.exists():
        return []
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "sessions" in data:
        return list(data["sessions"])
    if isinstance(data, dict) and "cli" in data:
        return [
            {
                "session_index": 0,
                "kind": "initial",
                "cli": data.get("cli"),
                "computed": data.get("computed"),
                "environment": data.get("environment"),
                "resume": None,
            }
        ]
    return []


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
    sessions = _load_config_sessions(config_path)
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
    target_mean, target_std = compute_target_statistics(
        args.yield_csv,
        args.datapath,
        harvest_years=args.harvest_years_set,
    )

    print("=> creating dataloader (Polars-optimized)")
    traindataloader, train_meta = get_aggregated_dataloader(
        args.datapath,
        args.yield_csv,
        args.year,
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
    )
    valdataloader, val_meta = get_aggregated_dataloader(
        args.datapath,
        args.yield_csv,
        args.year,
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
    )
    testdataloader, test_meta = get_aggregated_dataloader(
        args.datapath,
        args.yield_csv,
        args.year,
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
    )

    print("=> creating model")
    device = torch.device(args.device)
    from models import STNetRegression

    input_dim = int(traindataloader.dataset.input_feature_dim)
    print(
        f"Model input_dim={input_dim} (--feature-layout {args.feature_layout}; "
        "see datasets/feature_layout.py)"
    )
    model = STNetRegression(
        input_dim=input_dim,
        num_outputs=1,
        max_seq_len=args.sequencelength,
    ).to(device)

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

        state_dict = {
            k: v for k, v in pretrain_state.items() if k in model_dict.keys()
        }
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
            print(
                f"  ⚠️  Missing keys after load: {load_result.missing_keys[:5]}..."
            )
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
        target_column=YIELD_TARGET_COLUMN,
        model_class_name=model.__class__.__name__,
        harvest_years_set=args.harvest_years_set,
        year=args.year,
        seed=args.seed,
        suffix=args.suffix,
    )

    # Compile model for faster execution (PyTorch 2.0+)
    # Wrap in try-except in case compilation fails (e.g., missing Python headers in WSL)
    torch_compiled = False
    if hasattr(torch, "compile"):
        try:
            print("Compiling model with torch.compile() for faster execution...")
            model = torch.compile(model)
            torch_compiled = True
            print("Model compilation successful!")
        except Exception as e:
            print(
                f"⚠️  Warning: torch.compile() failed ({type(e).__name__}), continuing without compilation"
            )
            print(f"   Error: {str(e)[:200]}...")
            # Continue with uncompiled model

    run_dir = Path(args.logdir) / model.modelname
    training_dir_path, figures_dir_path, predictions_dir_path = ensure_run_layout(run_dir)
    best_model_path = training_dir_path / "model_best.pth"
    print(f"Run directory: {run_dir}")
    print(f"  training:    {training_dir_path}")
    print(f"  predictions: {predictions_dir_path}")
    print(f"  figures:     {figures_dir_path}")

    # Pass normalization stats for denormalization in evaluation
    criterion = AggregatedMSELoss(
        target_mean=target_mean,
        target_std=target_std,
    )
    print(f"DEBUG: args.learning_rate = {args.learning_rate:.2e}")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    print(
        f"DEBUG: optimizer.param_groups[0]['lr'] = {optimizer.param_groups[0]['lr']:.2e}"
    )

    # Initialize training state
    start_epoch = 0
    log = list()
    val_loss_min = np.inf
    not_improved_count = 0

    resume_info = None
    pretrained_path_resolved = None
    if args.pretrained:
        pretrained_path_resolved = Path(args.pretrained).resolve()

    # Resume from checkpoint if pretrained model is provided
    if pretrained_checkpoint is not None:
        checkpoint = pretrained_checkpoint
        print(f"Resuming training from checkpoint...")

        loaded_optimizer = False
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            loaded_optimizer = True
            print("  ✓ Loaded optimizer state")

        if "epoch" in checkpoint:
            start_epoch = int(checkpoint["epoch"])
            print(f"  ✓ Resuming from epoch {start_epoch}")

        if "val_loss_min" in checkpoint:
            val_loss_min = float(checkpoint["val_loss_min"])
            print(f"  ✓ Loaded best validation loss: {val_loss_min:.4f}")
        elif "val_loss" in checkpoint:
            val_loss_min = float(checkpoint["val_loss"])
            print(f"  ✓ Loaded validation loss: {val_loss_min:.4f}")

        if "not_improved_count" in checkpoint:
            not_improved_count = int(checkpoint["not_improved_count"])
            print(f"  ✓ Loaded not_improved_count: {not_improved_count}")

        ck_target_mean = checkpoint.get("target_mean")
        ck_target_std = checkpoint.get("target_std")
        normalization_stats_match = None
        if ck_target_mean is not None and ck_target_std is not None:
            ck_target_mean = float(ck_target_mean)
            ck_target_std = float(ck_target_std)
            normalization_stats_match = (
                abs(ck_target_mean - target_mean) <= 1e-3
                and abs(ck_target_std - target_std) <= 1e-3
            )
            if not normalization_stats_match:
                print(f"  ⚠️  Warning: Normalization stats differ!")
                print(f"      Checkpoint: mean={ck_target_mean:.2f}, std={ck_target_std:.2f}")
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
                print(f"  ✓ Loaded existing training log with {trainlog_entries_loaded} entries")
            except Exception as e:
                print(f"  ⚠️  Could not load existing log: {e}, starting fresh")
        else:
            print("  ℹ️  No existing training log found, starting fresh")

        resume_info = {
            "from_checkpoint": str(pretrained_path_resolved),
            "resumed_from_epoch": int(start_epoch),
            "loaded_optimizer_state": loaded_optimizer,
            "loaded_val_loss_min": float(val_loss_min) if np.isfinite(val_loss_min) else None,
            "loaded_not_improved_count": int(not_improved_count),
            "checkpoint_target_mean": ck_target_mean,
            "checkpoint_target_std": ck_target_std,
            "checkpoint_feature_layout": checkpoint.get("feature_layout"),
            "normalization_stats_match": normalization_stats_match,
            "trainlog_path": str(trainlog_file.resolve()),
            "trainlog_entries_loaded": trainlog_entries_loaded,
        }

    imagery_years_available = find_available_years(args.datapath)
    append_training_config_session(
        training_dir_path,
        args,
        computed={
            "model_run_name": model.modelname,
            "input_dim": int(input_dim),
            "feature_layout": args.feature_layout,
            "yield_target_column": YIELD_TARGET_COLUMN,
            "run_name_target_prefix": target_run_prefix(YIELD_TARGET_COLUMN),
            "municipality_aggregation": "sum",
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
            "imagery_years_used": sorted(args.harvest_years_set)
            if args.harvest_years_set
            else imagery_years_available,
            "epoch_plan": {
                "start_epoch": int(start_epoch),
                "end_epoch_exclusive": int(args.epochs),
                "num_epochs_this_session": max(0, int(args.epochs) - int(start_epoch)),
            },
        },
        run_paths={
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
            )
            print(f"Saved checkpoint at epoch {epoch + 1} to {checkpoint_path.name}")

        if not_improved_count >= 30:
            print(
                "\nValidation performance didn't improve for 30 epochs. Training stops."
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
    year,
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
):
    """Create dataloader for aggregated regression (Polars-optimized)."""
    dataset = USCropsAggregatedNPY(
        mode=mode,
        root=datapath,
        yield_csv=yield_csv,
        year=year,
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

    dataloader = DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        collate_fn=aggregated_collate_fn,
    )

    meta = dict(
        ndims=dataset.input_feature_dim,
        num_outputs=1,
        num_municipalities=len(dataset),
        use_xavier=dataset.use_xavier,
        feature_layout=dataset.feature_layout,
    )

    return dataloader, meta


def main():
    args = parse_args()
    years = YEARS
    use_all_years = None in years or "all" in years

    if use_all_years:
        print(
            " ===================== Training ONE model on ALL train data ======================= "
        )
        args.year = None
        seeds = SEEDS
        print("seed in", seeds)
        for seed in seeds:
            args.seed = seed
            print(f"Seed = {args.seed} --------------- ")

            SEED = args.seed
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = (
                True  # Enable cuDNN benchmarking for faster convolutions
            )

            run_dir = train(args)
    else:
        for year in years:
            print(
                f" ===================== {year} (model name only, uses all train data) ======================= "
            )
            args.year = year
            seeds = SEEDS
            print("seed in", seeds)
            for seed in seeds:
                args.seed = seed
                print(f"Seed = {args.seed} --------------- ")

                SEED = args.seed
                random.seed(SEED)
                np.random.seed(SEED)
                torch.manual_seed(SEED)
                torch.cuda.manual_seed_all(SEED)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = (
                    True  # Enable cuDNN benchmarking for faster convolutions
                )

                run_dir = train(args)


if __name__ == "__main__":
    main()
