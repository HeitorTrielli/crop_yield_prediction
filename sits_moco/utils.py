import re
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import moco.builder
from datasets import (
    ParanaMoCoDataset,
    RandomAddNoise,
    RandomSampleTimeSteps,
    RandomTempRemoval,
    RandomTempShift,
    harvest_years_to_year_ranges,
    is_parana_npy_layout,
)
from datasets.feature_layout import feature_layout_input_dim, normalize_feature_layout
from models import STNet, TransformerModel


# -------------------------- #
#          dataset           #
# -------------------------- #
def get_moco_dataloader(
    datapath,
    year,
    batchsize,
    workers,
    sequencelength,
    num,
    rc,
    seed,
    useall,
    year_ranges=None,
    max_samples=500_000,
    rebuild_cache=False,
    feature_layout="spectral",
):
    """Paraná municipal .npy MoCo dataloader (US-toy path archived)."""
    feature_layout = normalize_feature_layout(feature_layout)
    input_dim = feature_layout_input_dim(feature_layout)

    train_dataaug = transforms.Compose(
        [
            RandomTempShift(),
            RandomAddNoise(),
            RandomTempRemoval(),
            RandomSampleTimeSteps(sequencelength, rc=rc),
        ]
    )

    datapath = Path(datapath)
    if not year_ranges:
        if not is_parana_npy_layout(datapath):
            raise ValueError(
                f"MoCo expects a Paraná municipal .npy layout under {datapath}. "
                "The US-toy MoCo path was moved to archive/paper_us_classification/."
            )
        year_ranges = harvest_years_to_year_ranges([int(year)])
    if useall:
        n_samples = int(max_samples) if max_samples and max_samples > 0 else 500_000
    else:
        n_samples = int(num) if num and num > 0 else int(max_samples)
    pretraindataset = ParanaMoCoDataset(
        root=datapath,
        year_ranges=list(year_ranges),
        sequencelength=sequencelength,
        dataaug=train_dataaug,
        max_samples=n_samples,
        seed=seed,
        rebuild_cache=rebuild_cache,
        feature_layout=feature_layout,
    )

    num = len(pretraindataset)
    num_train = int(num * 0.9)
    indices = list(range(num))
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[:num_train], indices[num_train:]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    traindataloader = torch.utils.data.DataLoader(
        pretraindataset,
        batch_size=batchsize,
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    valdataloader = torch.utils.data.DataLoader(
        pretraindataset,
        batch_size=batchsize,
        sampler=valid_sampler,
        num_workers=workers,
        drop_last=True,
    )
    meta = dict(
        ndims=input_dim,
        n_samples=num,
        year_ranges=list(year_ranges) if year_ranges else None,
        feature_layout=feature_layout,
    )

    return traindataloader, valdataloader, meta


# -------------------------- #
#           Model            #
# -------------------------- #
def get_moco_model(modelname, device, args):
    modelname = modelname.lower()
    if modelname == "transformer":
        basemodel = TransformerModel
    elif modelname == "stnet":
        basemodel = STNet
    else:
        raise ValueError(
            "invalid MoCo backbone; choose 'transformer' or 'stnet' "
            "(LSTM/LTAE/TempCNN archived under archive/paper_us_classification/)"
        )

    feature_layout = normalize_feature_layout(
        getattr(args, "feature_layout", "spectral")
    )
    input_dim = feature_layout_input_dim(feature_layout)
    d_model = int(getattr(args, "model_d_model", 128))
    n_head = int(getattr(args, "model_n_head", 16))
    n_layers = int(getattr(args, "model_n_layers", 1))
    d_inner = int(getattr(args, "model_d_inner", 128))
    dropout = float(getattr(args, "model_dropout", 0.2))

    basemodel_factory = partial(
        basemodel,
        input_dim=input_dim,
        d_model=d_model,
        n_head=n_head,
        n_layers=n_layers,
        d_inner=d_inner,
        dropout=dropout,
    )

    model = moco.builder.MoCo(
        basemodel_factory, args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp
    )

    model.modelname = f"{model.modelname}{basemodel_factory().modelname}"

    model = model.to(device)

    return model


# -------------------------- #
#           Utils            #
# -------------------------- #
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, num_classes=21):
    num = target.shape[0]

    confusion_matrix = get_confusion_matrix(output, target, num_classes)
    TP = confusion_matrix.diagonal()
    FP = confusion_matrix.sum(1) - TP
    FN = confusion_matrix.sum(0) - TP

    po = TP.sum() / num
    pe = (confusion_matrix.sum(0) * confusion_matrix.sum(1)).sum() / num**2
    if pe == 1:
        kappa = 1
    else:
        kappa = (po - pe) / (1 - pe)

    p = TP / (TP + FP + 1e-12)
    r = TP / (TP + FN + 1e-12)
    f1 = 2 * p * r / (p + r + 1e-12)

    oa = po
    kappa = kappa
    macro_f1 = f1.mean()
    weight = confusion_matrix.sum(0) / confusion_matrix.sum()
    weighted_f1 = (weight * f1).sum()
    class_f1 = f1

    return dict(
        oa=oa,
        kappa=kappa,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        class_f1=class_f1,
        confusion_matrix=confusion_matrix,
    )


def get_confusion_matrix(y_pred, y_true, num_classes=21):
    idx = y_pred * num_classes + y_true
    return np.bincount(idx, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.learning_rate
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return [recursive_todevice(c, device) for c in x]


def save(model, path="model.pth", **kwargs):
    print(f"saving model to {str(path)}\n")
    # Handle torch.compile() wrapped models - get underlying model's state_dict
    if hasattr(model, "_orig_mod"):
        # Model is compiled, access underlying model
        model_state = model._orig_mod.state_dict()
    else:
        model_state = model.state_dict()
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    torch.save(dict(model_state=model_state, **kwargs), path)


def overall_performance(logdir):
    overall_metrics = defaultdict(list)

    for seed in [111, 222, 333, 444, 555]:
        log_dir = Path(
            logdir.replace(re.findall("Seed\d+", str(logdir))[0], f"Seed{seed}")
        )
        log_fn = log_dir / f"testlog.csv"
        if log_fn.exists():
            test_metrics = pd.read_csv(log_fn).iloc[0].to_dict()
            for metric, value in test_metrics.items():
                overall_metrics[metric].append(value)

    print(f"Overall result across 5 trials:")
    for metric, values in overall_metrics.items():
        values = np.array(values)
        if isinstance(values[0], (str)) or np.any(np.isnan(values)):
            continue
        if "loss" in metric or "f1" in metric or "kappa" in metric:
            print(f"{metric}: {np.mean(values):.4}")
        else:
            values *= 100
            print(f"{metric}: {np.mean(values):.2f}")

    print(
        f'{np.mean(overall_metrics["oa"])*100:.2f}\t'
        f'{np.mean(overall_metrics["kappa"]):.4f}\t'
        f'{np.mean(overall_metrics["weighted_f1"]):.4f}'
    )
    print()
