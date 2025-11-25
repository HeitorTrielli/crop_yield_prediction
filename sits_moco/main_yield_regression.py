"""
Main training script for municipality-level yield prediction using aggregated regression.
Pixel-level predictions are summed per municipality and compared to municipality-level targets.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim
from torch.utils.data import DataLoader

from datasets import USCropsAggregatedNPY
from utils import (
    AverageMeter,
    adjust_learning_rate,
    get_ntrainparams,
    recursive_todevice,
    save,
    weight_init,
)
from utils_aggregated import (
    AggregatedMSELoss,
    aggregated_collate_fn,
    regression_metrics,
    test_epoch_aggregated,
    train_epoch_aggregated,
)

# Default paths
DATAPATH = Path(r"files/yield_dataset")
YIELD_CSV = Path(r"files/municipality_production_with_codes.csv")
YEARS = [2023]
SEEDS = [111]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train yield prediction model using aggregated regression."
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
        default=6,
        help="Maximum length of time series data (default: 6 for 6 months)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=0,
        help="number of CPU workers to load the next batch",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=100, help="number of training epochs"
    )
    parser.add_argument(
        "-b",
        "--batchsize",
        type=int,
        default=16,
        help="batch size (number of municipalities per batch, default: 16)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=1e-3,
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
        help="path to yield CSV file with municipality codes and production",
    )
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="Fraction of municipalities to sample per epoch (0.0-1.0, default: 1.0 = use all)",
    )
    args = parser.parse_args()

    args.dataset = "USCropsAggregatedNPY"
    args.datapath = Path(args.datapath) if args.datapath else DATAPATH
    args.yield_csv = Path(args.yield_csv) if args.yield_csv else YIELD_CSV

    if args.interp and args.rc:
        args.rc_str = "IntRC"
    elif args.interp:
        args.rc_str = "Int"
    elif args.rc:
        args.rc_str = "RC"
    else:
        args.rc_str = "Pad"

    if args.use_doy:
        if args.suffix:
            args.suffix = "doy_" + args.suffix
        else:
            args.suffix = "doy"

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


def train(args):
    print("=> creating dataloader")
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
        mode="eval",
    )

    print("=> creating model")
    device = torch.device(args.device)
    from models import STNetRegression

    model = STNetRegression(
        input_dim=10,
        num_outputs=1,
        max_seq_len=args.sequencelength,
    ).to(device)

    print(
        f"Initialized {model.modelname}: Total trainable parameters: {get_ntrainparams(model)}"
    )
    model.apply(weight_init)

    if args.pretrained:
        print(f"Loading pretrained model from {args.pretrained}")
        checkpoint = torch.load(
            args.pretrained, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state"], strict=False)

    if args.suffix:
        model.modelname = f"Yield_{model.modelname}_{args.rc_str}_{args.year}_Seed{args.seed}_{args.suffix}"
    else:
        model.modelname = (
            f"Yield_{model.modelname}_{args.rc_str}_{args.year}_Seed{args.seed}"
        )

    logdir = Path(args.logdir) / model.modelname
    logdir.mkdir(parents=True, exist_ok=True)
    best_model_path = logdir / "model_best.pth"
    print(f"Logging results to {logdir}")

    criterion = AggregatedMSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    log = list()
    val_loss_min = np.Inf
    not_improved_count = 0

    print(f"Training {model.modelname}...")
    for epoch in range(args.epochs):
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
            model, optimizer, criterion, traindataloader, device, args
        )
        val_loss, scores = test_epoch_aggregated(
            model, criterion, valdataloader, device, args
        )

        scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in scores.items()])
        print(
            f"epoch {epoch + 1}: trainloss={train_loss:.4f}, valloss={val_loss:.4f} "
            + scores_msg
        )

        if val_loss < val_loss_min:
            not_improved_count = 0
            save(model, path=best_model_path, criterion=criterion)
            val_loss_min = val_loss
            print(f"lowest val loss in epoch {epoch + 1}\n")
        else:
            not_improved_count += 1

        scores["epoch"] = epoch + 1
        scores["trainloss"] = train_loss
        scores["valloss"] = val_loss
        log.append(scores)

        log_df = pd.DataFrame(log).set_index("epoch")
        log_df.to_csv(Path(logdir) / "trainlog.csv")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = logdir / f"checkpoint_epoch_{epoch + 1}.pth"
            save(
                model,
                path=checkpoint_path,
                criterion=criterion,
                optimizer_state=optimizer.state_dict(),
                epoch=epoch + 1,
                val_loss=val_loss,
                train_loss=train_loss,
            )
            print(f"Saved checkpoint at epoch {epoch + 1} to {checkpoint_path.name}")

        if not_improved_count >= 10:
            print(
                "\nValidation performance didn't improve for 10 epochs. Training stops."
            )
            break

    if epoch == args.epochs - 1:
        print(f"\n{args.epochs} epochs training finished.")

    # Test
    print("Restoring best model weights for testing...")
    checkpoint = torch.load(best_model_path, weights_only=False)
    state_dict = {k: v for k, v in checkpoint["model_state"].items()}
    criterion = checkpoint["criterion"]
    torch.save({"model_state": state_dict, "criterion": criterion}, best_model_path)
    model.load_state_dict(state_dict)

    test_loss, scores = test_epoch_aggregated(
        model, criterion, testdataloader, device, args
    )
    scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in scores.items()])
    print(f"Test results: \n\n {scores_msg}")

    scores["epoch"] = "test"
    scores["testloss"] = test_loss

    log_df = pd.DataFrame([scores]).set_index("epoch")
    log_df.to_csv(logdir / f"testlog.csv")

    return logdir


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
):
    """Create dataloader for aggregated regression."""
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
        ndims=10,
        num_outputs=1,
        num_municipalities=len(dataset),
    )

    return dataloader, meta


def main():
    args = parse_args()
    years = YEARS
    for year in years:
        print(f" ===================== {year} ======================= ")
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

            logdir = train(args)


if __name__ == "__main__":
    main()
