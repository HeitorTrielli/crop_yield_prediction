#!/usr/bin/env python3
"""t-SNE of trial_005 encoder outputs (spectral / climate / fused).

For each holdout year Y, embed munis of year Y with the LOO encoder trained
without Y. Plot t-SNE colored by actual yield, decoder prediction, and year.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.debug.analyze_trial005_encoder_vs_decoder import (  # noqa: E402
    EXP,
    FEAT_CSV,
    PRED_HI,
    PRED_LO,
    TRIAL,
    build_dataset_for_fold,
)
from training.muni_agg_batch import (  # noqa: E402
    load_stacked_muni_agg_batch,
    muni_agg_on_device,
)

YEARS = tuple(range(2020, 2025))
DEFAULT_OUT = EXP / "encoder_tsne"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--perplexity", type=float, default=40.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trial-dir", type=Path, default=TRIAL, help="LOO trial root with looYYYY/")
    p.add_argument("--feat-csv", type=Path, default=FEAT_CSV)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--title", type=str, default="PR trial_005")
    p.add_argument("--rebuild-cache", action="store_true")
    return p.parse_args()


def encode_parts(model, stacked, device, t_mean: float, t_std: float):
    x = muni_agg_on_device(stacked, device, argparse.Namespace(h2d_pin_host=False))
    use_amp = device.type == "cuda"
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
    ):
        x_s2, mask_s2, doy_s2, w_s2, x_c, mask_c, doy_c, w_c = x
        x_spec, soil = model._split_soil(x_s2, mask_s2)
        e_spec = model.spectral(x_spec, mask_s2, doy_s2, w_s2)
        e_clim = model.climate(x_c, mask_c, doy_c, w_c)
        parts = [e_spec, e_clim]
        e_soil = None
        if soil is not None:
            e_soil = soil
            parts.append(soil)
        fused = torch.cat(parts, dim=-1)
        pred = model.decoder(fused).reshape(-1)
    out = {
        "e_spec": e_spec.float().cpu().numpy(),
        "e_clim": e_clim.float().cpu().numpy(),
        "e_fused": fused.float().cpu().numpy(),
        "pred": (pred.float().cpu().numpy() * float(t_std)) + float(t_mean),
    }
    if e_soil is not None:
        out["e_soil"] = e_soil.float().cpu().numpy()
    return out


def build_holdout_cache(
    labels: pd.DataFrame,
    output_dir: Path,
    *,
    trial_dir: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
    rebuild: bool,
) -> pd.DataFrame:
    """One row per muni-year, embeddings from that year's LOO encoder."""
    cache_path = output_dir / "holdout_encoder_embeddings.parquet"
    if cache_path.is_file() and not rebuild:
        print(f"Loading {cache_path}")
        return pd.read_parquet(cache_path)

    rows: list[pd.DataFrame] = []
    for year in YEARS:
        ckpt = trial_dir / f"loo{year}" / "training" / "model_best.pth"
        holdout = labels[labels["harvest_year"] == year].reset_index(drop=True)
        print(f"LOO {year}: encoding {len(holdout)} holdout munis with {ckpt}")
        model, dataset, t_mean, t_std, _ = build_dataset_for_fold(ckpt, device)
        model.eval()
        for start in range(0, len(holdout), batch_size):
            source = holdout.iloc[start : start + batch_size].reset_index(drop=True)
            stacked, kept = load_stacked_muni_agg_batch(
                dataset,
                source["municipality_code"].tolist(),
                source["harvest_year"].tolist(),
                workers=workers,
            )
            if stacked is None or not kept:
                continue
            enc = encode_parts(model, stacked, device, t_mean=t_mean, t_std=t_std)
            meta = source.iloc[kept][
                ["municipality_code", "municipality_name", "harvest_year", "actual"]
            ].reset_index(drop=True)
            meta["decoder_prediction"] = enc["pred"]
            meta["encoder_fold"] = int(year)
            frames = [meta]
            for key, prefix in (
                ("e_spec", "spec"),
                ("e_clim", "clim"),
                ("e_fused", "fused"),
            ):
                mat = enc[key]
                frames.append(
                    pd.DataFrame(
                        mat,
                        columns=[f"{prefix}_{i:03d}" for i in range(mat.shape[1])],
                    )
                )
            if "e_soil" in enc:
                mat = enc["e_soil"]
                frames.append(
                    pd.DataFrame(
                        mat,
                        columns=[f"soil_{i:03d}" for i in range(mat.shape[1])],
                    )
                )
            rows.append(pd.concat(frames, axis=1))
            print(f"  {min(start + batch_size, len(holdout))}/{len(holdout)}", flush=True)
        del model, dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = pd.concat(rows, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache_path, index=False)
    print(f"Wrote {cache_path} ({len(result)} rows)")
    return result


def matrix_from_prefix(df: pd.DataFrame, prefix: str) -> np.ndarray:
    cols = [c for c in df.columns if c.startswith(f"{prefix}_")]
    if not cols:
        raise ValueError(f"No columns with prefix {prefix}_")
    return df[cols].to_numpy(dtype=np.float64)


def run_tsne(x: np.ndarray, *, perplexity: float, seed: int) -> np.ndarray:
    n = len(x)
    perp = min(perplexity, max(5.0, (n - 1) / 3.0))
    z = StandardScaler().fit_transform(x)
    return TSNE(
        n_components=2,
        perplexity=perp,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        metric="euclidean",
    ).fit_transform(z)


def scatter_panel(ax, xy, color, *, cmap, title, cbar_label=None, discrete=False):
    if discrete:
        years = np.asarray(color)
        uniq = sorted(np.unique(years))
        cmap_d = plt.get_cmap("tab10")
        for i, y in enumerate(uniq):
            m = years == y
            ax.scatter(
                xy[m, 0],
                xy[m, 1],
                s=10,
                alpha=0.65,
                c=[cmap_d(i % 10)],
                label=str(int(y)),
                rasterized=True,
                edgecolors="none",
            )
        ax.legend(frameon=False, fontsize=7, loc="best", title="year")
    else:
        sc = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=color,
            s=10,
            alpha=0.7,
            cmap=cmap,
            rasterized=True,
            edgecolors="none",
        )
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    feat = pd.read_csv(args.feat_csv)
    feat["municipality_code"] = feat["municipality_code"].astype(str)
    labels = feat.rename(
        columns={
            "actual_productivity_t_ha": "actual",
            "predicted_productivity_t_ha": "pred_csv",
        }
    )[
        [
            "municipality_code",
            "municipality_name",
            "harvest_year",
            "actual",
            "pred_csv",
        ]
    ].copy()

    emb = build_holdout_cache(
        labels,
        out,
        trial_dir=Path(args.trial_dir),
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
        rebuild=args.rebuild_cache,
    )
    emb["municipality_code"] = emb["municipality_code"].astype(str)
    emb = emb.merge(
        labels[["municipality_code", "harvest_year", "pred_csv"]],
        on=["municipality_code", "harvest_year"],
        how="left",
    )
    actual = emb["actual"].to_numpy(dtype=np.float64)
    pred = emb["decoder_prediction"].to_numpy(dtype=np.float64)
    year = emb["harvest_year"].to_numpy(dtype=np.int64)
    in_band = (pred >= PRED_LO) & (pred <= PRED_HI)

    modalities = [
        ("spec", "Spectral encoder"),
        ("clim", "Climate encoder"),
        ("fused", "Fused pre-decoder"),
    ]
    coords = {}
    for prefix, _ in modalities:
        print(f"t-SNE on {prefix} ({matrix_from_prefix(emb, prefix).shape[1]}-d)…")
        coords[prefix] = run_tsne(
            matrix_from_prefix(emb, prefix),
            perplexity=args.perplexity,
            seed=args.seed,
        )

    # Full sample: 3 modalities × (actual, pred, year)
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 12.5))
    for r, (prefix, name) in enumerate(modalities):
        xy = coords[prefix]
        scatter_panel(
            axes[r, 0],
            xy,
            actual,
            cmap="viridis",
            title=f"{name} · actual",
            cbar_label="t/ha",
        )
        scatter_panel(
            axes[r, 1],
            xy,
            pred,
            cmap="magma",
            title=f"{name} · prediction",
            cbar_label="t/ha",
        )
        scatter_panel(
            axes[r, 2],
            xy,
            year,
            cmap="tab10",
            title=f"{name} · year",
            discrete=True,
        )
    fig.suptitle(
        f"{args.title} · t-SNE of encoder outputs (holdout LOO embeddings)\n"
        "If structure tracks actual but not prediction → decoder flattening",
        fontsize=12,
    )
    fig.tight_layout()
    fig_path = out / "tsne_encoder_actual_pred_year.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {fig_path}")

    # Highlight pred band [3.50, 3.75] on fused t-SNE
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    xy = coords["fused"]
    axes[0].scatter(
        xy[~in_band, 0],
        xy[~in_band, 1],
        s=8,
        c="0.75",
        alpha=0.35,
        rasterized=True,
        edgecolors="none",
        label="outside band",
    )
    sc = axes[0].scatter(
        xy[in_band, 0],
        xy[in_band, 1],
        s=12,
        c=actual[in_band],
        cmap="viridis",
        alpha=0.85,
        rasterized=True,
        edgecolors="none",
        label="pred∈[3.50,3.75]",
    )
    plt.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.04).set_label("actual t/ha")
    axes[0].set_title("Fused t-SNE · band colored by actual")
    axes[0].legend(frameon=False, fontsize=8, loc="best")

    sc2 = axes[1].scatter(
        xy[in_band, 0],
        xy[in_band, 1],
        s=12,
        c=pred[in_band],
        cmap="magma",
        alpha=0.85,
        rasterized=True,
        edgecolors="none",
    )
    plt.colorbar(sc2, ax=axes[1], fraction=0.046, pad=0.04).set_label("pred t/ha")
    axes[1].set_title("Fused t-SNE · band only, colored by prediction")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(
        f"Same-prediction band: n={int(in_band.sum())} · "
        f"actual std={actual[in_band].std():.2f} · pred std={pred[in_band].std():.2f}",
        fontsize=11,
    )
    fig.tight_layout()
    band_path = out / "tsne_fused_pred_band.png"
    fig.savefig(band_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {band_path}")

    # Save 2D coords for reuse
    coord_df = emb[
        ["municipality_code", "municipality_name", "harvest_year", "actual", "decoder_prediction"]
    ].copy()
    for prefix, _ in modalities:
        coord_df[f"tsne_{prefix}_1"] = coords[prefix][:, 0]
        coord_df[f"tsne_{prefix}_2"] = coords[prefix][:, 1]
    coord_path = out / "tsne_coordinates.csv"
    coord_df.to_csv(coord_path, index=False)

    summary = {
        "n": int(len(emb)),
        "n_band": int(in_band.sum()),
        "pred_band": [PRED_LO, PRED_HI],
        "band_actual_std": float(actual[in_band].std()),
        "band_pred_std": float(pred[in_band].std()),
        "all_actual_std": float(actual.std()),
        "all_pred_std": float(pred.std()),
        "figures": [str(fig_path), str(band_path)],
        "coordinates": str(coord_path),
        "perplexity": args.perplexity,
        "seed": args.seed,
    }
    with open(out / "tsne_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
