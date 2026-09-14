"""
This script is for SITS-MoCo pre-training task.

Paraná municipal .npy (files/npy/YYYY-YYYY/…) is used when --datapath points at that
layout; otherwise the original US-toy MoCoDataset path is kept.
"""
import argparse
import sys
from pathlib import Path

import torch.nn as nn
from tqdm import tqdm

from env_config import resolve_datapath
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-Train a time series feature extractor.")
    parser.add_argument(
        "model",
        type=str,
        default="transformer",
        help="select pretrain method model architecture (default Transformer).",
    )
    parser.add_argument(
        "--use-doy",
        action="store_true",
        help="whether to use doy pe with trsf",
    )
    parser.add_argument(
        "--rc",
        action="store_true",
        help="whether to random choice the time series data",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--useall",
        action="store_true",
        help=(
            "US-toy: use full unsupervised cache. "
            "Paraná: use all municipalities up to --max-samples (still a subsample)."
        ),
    )
    parser.add_argument(
        "-n",
        "--num",
        default=3000,
        type=int,
        help="US-toy labeled subset size, or Paraná max samples when --useall is off (default 3000)",
    )
    parser.add_argument(
        "--max-samples",
        default=500_000,
        type=int,
        help="Paraná: stratified pixel subsample size when --useall (default 500000)",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Paraná: rebuild .moco_cache subsample even if cache file exists",
    )
    parser.add_argument(
        "-c",
        "--nclasses",
        type=int,
        default=20,
        help="num of classes (default: 20)",
    )
    parser.add_argument(
        "--sequencelength",
        type=int,
        default=70,
        help="Maximum length of time series data (default 70)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2019,
        help="US-toy year, or single harvest year if --harvest-years not set (default 2019)",
    )
    parser.add_argument(
        "--harvest-years",
        type=str,
        default=None,
        help=(
            "Comma-separated harvest years for Paraná .npy "
            "(folder (Y-1)-Y). Default when datapath is Paraná: 2020,2021,2022,2023,2024"
        ),
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default=None,
        help=(
            "Dataset root. Paraná: season folders with municipal .npy "
            "(default: files/npy if present, else SITS_MOCO_DATAPATH, else data/US-toy)."
        ),
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=0,
        help="number of CPU workers to load the next batch",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=100,
        help="number of training epochs",
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="warmup epochs",
    )
    parser.add_argument(
        "-b",
        "--batchsize",
        type=int,
        default=512,
        help="batch size (number of time series processed simultaneously)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=1e-3,
        help="optimizer learning rate (default 1e-3)",
    )
    parser.add_argument(
        "--schedule",
        default=None,
        nargs="*",
        type=int,
        help="learning rate schedule (when to drop lr by a ratio)",
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        metavar="M",
        help="momentum",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="optimizer weight_decay (default 0)",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--seed",
        default=111,
        type=int,
        help="seed for initializing training. ",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        help='torch.Device. either "cpu" or "cuda". default will check by torch.cuda.is_available() ',
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="./results",
        help="logdir to store progress and models (defaults to ./results)",
    )
    parser.add_argument(
        "-s",
        "--suffix",
        default=None,
        help="suffix to output_dir",
    )

    # moco specific configs:
    parser.add_argument(
        "--moco-dim",
        default=128,
        type=int,
        help="feature dimension (default: 128)",
    )
    parser.add_argument(
        "--moco-k",
        default=65536,
        type=int,
        help="queue size; number of negative keys (default: 65536)",
    )
    parser.add_argument(
        "--moco-m",
        default=0.999,
        type=float,
        help="moco momentum of updating key encoder (default: 0.999)",
    )
    parser.add_argument(
        "--moco-t",
        default=0.07,
        type=float,
        help="softmax temperature (default: 0.07)",
    )

    # options for moco v2
    parser.add_argument("--mlp", action="store_true", help="use mlp head")
    parser.add_argument(
        "--aug-plus",
        action="store_true",
        help="use moco v2 data augmentation",
    )
    parser.add_argument(
        "--cos",
        action="store_true",
        help="use cosine lr schedule",
    )
    parser.add_argument(
        "--feature-layout",
        type=str,
        default="spectral",
        help=(
            "Pixel feature layout for Paraná MoCo (default: spectral). "
            "Must match yield finetune layout for full weight transfer."
        ),
    )
    parser.add_argument(
        "--model-d-model",
        type=int,
        default=128,
        help="Transformer d_model (default: 128)",
    )
    parser.add_argument(
        "--model-n-head",
        type=int,
        default=16,
        help="Transformer n_head (default: 16)",
    )
    parser.add_argument(
        "--model-n-layers",
        type=int,
        default=1,
        help="Transformer n_layers (default: 1)",
    )
    parser.add_argument(
        "--model-d-inner",
        type=int,
        default=128,
        help="Transformer d_inner / FFN width (default: 128)",
    )
    parser.add_argument(
        "--model-dropout",
        type=float,
        default=0.2,
        help="Encoder dropout (default: 0.2)",
    )

    args = parser.parse_args()

    from datasets.feature_layout import normalize_feature_layout

    args.feature_layout = normalize_feature_layout(args.feature_layout)

    # Resolve datapath: CLI > local files/npy > .env > US-toy
    if args.datapath:
        args.datapath = Path(args.datapath).expanduser().resolve()
    else:
        local_npy = Path("files/npy")
        if local_npy.is_dir() and any(local_npy.iterdir()):
            args.datapath = local_npy.resolve()
        else:
            try:
                args.datapath = resolve_datapath()
            except EnvironmentError:
                args.datapath = Path("data/US-toy").resolve()

    from datasets.moco_parana import harvest_years_to_year_ranges, is_parana_npy_layout

    args.is_parana = is_parana_npy_layout(args.datapath)
    if args.harvest_years:
        args.harvest_years_list = [
            int(x.strip()) for x in args.harvest_years.split(",") if x.strip()
        ]
    elif args.is_parana:
        args.harvest_years_list = [2020, 2021, 2022, 2023, 2024]
    else:
        args.harvest_years_list = [int(args.year)]

    if args.is_parana:
        args.year_ranges = harvest_years_to_year_ranges(args.harvest_years_list)
        # Tag run names with harvest span (e.g. 2020-2024)
        args.year = (
            f"{min(args.harvest_years_list)}-{max(args.harvest_years_list)}"
            if len(args.harvest_years_list) > 1
            else int(args.harvest_years_list[0])
        )
    else:
        args.year_ranges = None

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.rc:
        args.rc_str = "RC"
    else:
        args.rc_str = "Pad"

    if args.use_doy:
        if args.suffix:
            args.suffix = "doy_" + args.suffix
        else:
            args.suffix = "doy"

    # Encode trunk fingerprint so matching yield trials resolve a unique MoCo run dir.
    arch_tag = (
        f"{args.feature_layout}"
        f"_d{int(args.model_d_model)}"
        f"_h{int(args.model_n_head)}"
        f"_i{int(args.model_d_inner)}"
        f"_L{int(args.model_n_layers)}"
    )
    if args.suffix:
        args.suffix = f"{args.suffix}_{arch_tag}"
    else:
        args.suffix = arch_tag

    if args.seed is not None:
        SEED = args.seed
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    return args


def train(args):
    print("=> creating dataloader")
    print(f"  datapath={args.datapath} (parana={args.is_parana})")
    if args.year_ranges:
        print(f"  year_ranges={args.year_ranges}")

    traindataloader, valdataloader, meta = get_moco_dataloader(
        args.datapath,
        args.year if isinstance(args.year, int) else args.harvest_years_list[0],
        args.batchsize,
        args.workers,
        args.sequencelength,
        args.num,
        args.rc,
        args.seed,
        args.useall,
        year_ranges=args.year_ranges,
        max_samples=args.max_samples,
        rebuild_cache=args.rebuild_cache,
        feature_layout=args.feature_layout,
    )
    print(f"  samples={meta.get('n_samples')} batches/train≈{len(traindataloader)}")
    print(
        f"  feature_layout={args.feature_layout} input_dim={meta.get('ndims')} "
        f"d_model={args.model_d_model} n_head={args.model_n_head} "
        f"d_inner={args.model_d_inner} n_layers={args.model_n_layers}"
    )

    print("=> creating model")
    device = torch.device(args.device)
    model = get_moco_model(args.model, device, args)

    if args.useall:
        model.modelname = f"P_{model.modelname}_{args.rc_str}_{args.year}"
    else:
        model.modelname = (
            f"P_{model.modelname}_R{args.num}_{args.rc_str}_{args.year}_Seed{args.seed}"
        )

    if args.suffix:
        model.modelname += f"_{args.suffix}"

    from run_paths import ensure_run_layout, run_dir_from_path, trainlog_path

    run_dir = Path(args.logdir) / model.modelname
    training_dir_path, _, _ = ensure_run_layout(run_dir)
    best_model_path = training_dir_path / "model_best.pth"
    print(f"Run directory: {run_dir}")
    print(f"  training: {training_dir_path}")

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    log = list()
    val_loss_min = np.inf
    not_improved_count = 0
    if args.resume:
        path = Path(args.resume).absolute().relative_to(Path(__file__).absolute().parent)
        print("=> loading checkpoint '{}'".format(str(path)))

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state"])
        args.start_epoch = torch.load(path)["epoch"] + 1
        val_loss_min = checkpoint["val_loss_min"]
        not_improved_count = checkpoint["not_improved_count"]

        print("=> loaded checkpoint '{}'".format(str(path)))

        log_fn = trainlog_path(run_dir_from_path(path))
        log = pd.read_csv(log_fn).to_dict("records")[: args.start_epoch]

    print(f"Pre-training {args.model}...")
    for epoch in range(args.start_epoch, args.epochs):

        if args.warmup_epochs > 0:
            if epoch == 0:
                lr = args.learning_rate * 0.01
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
            elif epoch == args.warmup_epochs:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = args.learning_rate
        if args.schedule is not None:
            adjust_learning_rate(optimizer, epoch, args)
        train_loss = train_epoch(
            model, optimizer, criterion, traindataloader, device, use_doy=args.use_doy
        )
        val_loss = test_epoch(
            model, criterion, valdataloader, device, use_doy=args.use_doy
        )

        print(f"epoch {epoch}: trainloss {train_loss:.4f}, valloss {val_loss:.4f} ")

        scores = {}
        scores["epoch"] = epoch
        scores["trainloss"] = train_loss
        scores["testloss"] = val_loss
        log.append(scores)

        log_df = pd.DataFrame(log).set_index("epoch")
        log_df.to_csv(training_dir_path / "trainlog.csv")

        if val_loss < val_loss_min:
            not_improved_count = 0
            val_loss_min = val_loss
            save(
                model,
                path=best_model_path,
                epoch=epoch,
                val_loss_min=val_loss,
                not_improved_count=not_improved_count,
            )
        else:
            not_improved_count += 1

        if not_improved_count >= 10:
            print(
                "\nValidation performance didn't improve for 10 epochs. Training stops."
            )
            break


def train_epoch(model, optimizer, criterion, dataloader, device, use_doy=False):
    losses = AverageMeter("Loss", ":.4e")
    model.train()

    # When stdout is a log file (tuning capture), skip per-batch bars — they
    # expand to megabytes of \\r noise. Interactive TTY keeps a single updating bar.
    show_bar = sys.stdout.isatty()
    with tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        leave=False,
        disable=not show_bar,
        mininterval=1.0,
        dynamic_ncols=True,
    ) as iterator:
        for idx, (data_q, data_k) in iterator:
            data_q = recursive_todevice(data_q, device)
            data_k = recursive_todevice(data_k, device)

            output, target = model(data_q=data_q, data_k=data_k, use_doy=use_doy)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if show_bar:
                iterator.set_description(f"train loss={loss:.2f}")
            losses.update(loss.item(), data_q[0].size(0))

    return losses.avg


def test_epoch(model, criterion, dataloader, device, use_doy=False):
    losses = AverageMeter("Loss", ":.4e")
    model.eval()
    show_bar = sys.stdout.isatty()
    with torch.no_grad():
        with tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            leave=False,
            disable=not show_bar,
            mininterval=1.0,
            dynamic_ncols=True,
        ) as iterator:
            for idx, (data_q, data_k) in iterator:
                data_q = recursive_todevice(data_q, device)
                data_k = recursive_todevice(data_k, device)

                output, target = model(data_q=data_q, data_k=data_k, use_doy=use_doy)
                loss = criterion(output, target)

                if show_bar:
                    iterator.set_description(f"test loss={loss:.2f}")
                losses.update(loss.item(), data_q[0].size(0))

    return losses.avg


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
