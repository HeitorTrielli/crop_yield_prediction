#!/usr/bin/env python3
"""Quantify input similarity and encoder vs decoder collapse for trial_005.

For each LOO fold (holdout year Y), load model_best.pth trained without Y and
encode holdout munis. Compare:
  - raw summary-feature distances
  - spectral / climate / fused pre-decoder embeddings
  - final predictions

If inputs differ but embeddings are tight → encoder collapse.
If embeddings differ but predictions are tight → decoder collapse.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from datasets.daily_climate import climate_extra_spec_from_run_config  # noqa: E402
from datasets.extra_scaler import (  # noqa: E402
    InputScaler,
    apply_extra_scaler_to_dataset,
)
from datasets.feature_layout import normalize_feature_layout  # noqa: E402
from datasets.pixel_chunk import unpack_pixel_chunk  # noqa: E402
from datasets.uscrops_aggregated_npy_polars import USCropsAggregatedNPY  # noqa: E402
from training.inference_core import (  # noqa: E402
    build_stnet_from_checkpoint,
    load_checkpoint,
)
from run_paths import load_latest_run_config  # noqa: E402

EXP = ROOT / "loo_pixel_results" / "pr_only_trial_005"
FEAT_CSV = EXP / "muni_mean_pred_band_inputs" / "muni_year_input_features.csv"
TRIAL = (
    ROOT
    / "results"
    / "tuning"
    / "productivity_muni_mean_dual_stnet_wide_soy_pr_0.5_1.5_top100_year_loo"
    / "trial_005"
)
OUT = EXP / "muni_mean_pred_band_inputs"
PRED_LO, PRED_HI = 3.50, 3.75
FEATURE_COLS = [
    "ndvi_peak",
    "ndvi_mean",
    "ndvi_window",
    "n_valid_days",
    "rain_sum_mm",
    "rain_dec_feb_mm",
    "eto_sum_mm",
    "tmax_mean",
    "tmin_mean",
    "rs_mean",
    "dry_spell_max",
    "clay_pct",
    "silt_pct",
    "sand_pct",
    "soc_t_ha",
]


def zscore(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (x - mu) / sd


def pairwise_stats(
    x: np.ndarray,
    actual: np.ndarray,
    pred: np.ndarray,
    *,
    rng: np.random.Generator,
    n_pairs: int = 12000,
) -> dict:
    """Sample pairwise L2 and cosine on rows of x; relate to |Δactual|, |Δpred|."""
    n = len(x)
    if n < 5:
        return {}
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    d = x[i] - x[j]
    l2 = np.linalg.norm(d, axis=1)
    ni = np.linalg.norm(x[i], axis=1) + 1e-8
    nj = np.linalg.norm(x[j], axis=1) + 1e-8
    cos = 1.0 - np.sum(x[i] * x[j], axis=1) / (ni * nj)
    d_act = np.abs(actual[i] - actual[j])
    d_pred = np.abs(pred[i] - pred[j])
    return {
        "l2_mean": float(np.mean(l2)),
        "l2_median": float(np.median(l2)),
        "l2_p10": float(np.percentile(l2, 10)),
        "l2_p90": float(np.percentile(l2, 90)),
        "cosine_mean": float(np.mean(cos)),
        "cosine_median": float(np.median(cos)),
        "spearman_l2_vs_dact": float(stats.spearmanr(l2, d_act).correlation),
        "spearman_l2_vs_dpred": float(stats.spearmanr(l2, d_pred).correlation),
        "frac_pairs_cosine_lt_0.1": float(np.mean(cos < 0.1)),
        "frac_pairs_cosine_lt_0.05": float(np.mean(cos < 0.05)),
    }


def encode_batch(model, batch, device):
    """Return spectral, climate, soil, fused embeddings + prediction."""
    stacked = unpack_pixel_chunk(batch)
    if stacked is None:
        return None
    stacked = tuple(t.to(device) for t in stacked)
    with torch.no_grad():
        x_s2, mask_s2, doy_s2, w_s2, x_c, mask_c, doy_c, w_c = stacked
        x_spec, soil = model._split_soil(x_s2, mask_s2)
        parts = []
        e_spec = model.spectral(x_spec, mask_s2, doy_s2, w_s2)
        parts.append(e_spec)
        e_clim = model.climate(x_c, mask_c, doy_c, w_c)
        parts.append(e_clim)
        if soil is not None:
            parts.append(soil)
        fused = torch.cat(parts, dim=-1)
        pred = model.decoder(fused)
        # mean over pixels (muni_mean → 1)
        return {
            "e_spec": e_spec.mean(0).cpu().numpy(),
            "e_clim": e_clim.mean(0).cpu().numpy(),
            "e_soil": (
                soil.mean(0).cpu().numpy() if soil is not None else None
            ),
            "e_fused": fused.mean(0).cpu().numpy(),
            "pred_norm": float(pred.mean().cpu()),
        }


def build_dataset_for_fold(ckpt_path: Path, device: torch.device):
    run_config = load_latest_run_config(str(ckpt_path))
    ckpt = load_checkpoint(ckpt_path, device)
    cli = (run_config.get("cli") or {}) if run_config else {}
    sequencelength = int(cli.get("sequencelength") or 45)
    model, meta = build_stnet_from_checkpoint(
        ckpt,
        device=device,
        sequencelength=sequencelength,
        run_config=run_config,
        feature_layout=cli.get("feature_layout"),
    )
    feature_layout = normalize_feature_layout(
        cli.get("feature_layout") or meta["feature_layout"]
    )
    datapath = Path(cli["datapath"]).expanduser()
    yield_csv = Path(cli.get("yield_csv") or "files/pam_soy_pr_mapbiomas_coverage_0.5_1.5.csv")
    if not yield_csv.is_file():
        yield_csv = ROOT / yield_csv
    dataset = USCropsAggregatedNPY(
        mode="all",
        root=datapath.resolve(),
        yield_csv=yield_csv,
        year=None,
        sequencelength=sequencelength,
        randomchoice=False,
        interp=False,
        seed=101010,
        feature_layout=feature_layout,
        target_column="yield_t_ha",
        min_coverage_ratio=None,
        max_coverage_ratio=None,
        climate_sequencelength=int(meta["climate_sequencelength"]),
        climate_extra_features=climate_extra_spec_from_run_config(run_config),
    )
    computed = (run_config or {}).get("computed") or {}
    payload = computed.get("extra_scaler")
    if isinstance(payload, dict) and payload.get("spectral") is not None:
        scaler = InputScaler.from_dict(payload)
    else:
        # fall back to files/
        name = Path(str(cli.get("extra_scaler") or "")).name
        scaler = InputScaler.load(ROOT / "files" / name)
    apply_extra_scaler_to_dataset(dataset, scaler)
    t_mean = float(meta["target_mean"])
    t_std = float(meta["target_std"])
    return model, dataset, t_mean, t_std, meta


def collect_embeddings(device: torch.device) -> pd.DataFrame:
    feat = pd.read_csv(FEAT_CSV)
    feat["municipality_code"] = feat["municipality_code"].astype(str)
    rows = []
    for year in range(2020, 2025):
        ckpt = TRIAL / f"loo{year}" / "training" / "model_best.pth"
        print(f"\n=== LOO {year}: {ckpt} ===")
        model, dataset, t_mean, t_std, meta = build_dataset_for_fold(ckpt, device)
        sub = feat[feat["harvest_year"] == year]
        for k, r in enumerate(sub.itertuples(index=False)):
            code = str(r.municipality_code)
            try:
                chunks = list(
                    dataset.load_pixels_from_municipality(
                        code, year=year, chunk_size=8
                    )
                )
            except Exception as e:  # noqa: BLE001
                print(f"  skip {code}: {e}")
                continue
            if not chunks:
                continue
            enc = encode_batch(model, chunks[0], device)
            if enc is None:
                continue
            pred = enc["pred_norm"] * t_std + t_mean
            row = {
                "municipality_code": code,
                "harvest_year": year,
                "actual": float(r.actual_productivity_t_ha),
                "pred_csv": float(r.predicted_productivity_t_ha),
                "pred_model": pred,
                "e_spec": enc["e_spec"],
                "e_clim": enc["e_clim"],
                "e_fused": enc["e_fused"],
            }
            if enc["e_soil"] is not None:
                row["e_soil"] = enc["e_soil"]
            rows.append(row)
            if (k + 1) % 50 == 0:
                print(f"  {k+1}/{len(sub)}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def stack_field(rows, key):
    return np.stack([r[key] for r in rows], axis=0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    feat = pd.read_csv(FEAT_CSV)
    band_mask = feat["predicted_productivity_t_ha"].between(PRED_LO, PRED_HI)
    band = feat[band_mask].copy()
    other = feat[~band_mask].copy()

    # ---- 1) Input-space similarity ----
    cols = [c for c in FEATURE_COLS if feat[c].notna().all()]
    x_all = zscore(feat[cols].to_numpy(dtype=np.float64))
    x_band = zscore(band[cols].to_numpy(dtype=np.float64))
    x_other = zscore(other[cols].to_numpy(dtype=np.float64))

    input_stats = {
        "all": pairwise_stats(
            x_all,
            feat["actual_productivity_t_ha"].to_numpy(),
            feat["predicted_productivity_t_ha"].to_numpy(),
            rng=rng,
        ),
        "pred_band": pairwise_stats(
            x_band,
            band["actual_productivity_t_ha"].to_numpy(),
            band["predicted_productivity_t_ha"].to_numpy(),
            rng=rng,
        ),
        "outside_band": pairwise_stats(
            x_other,
            other["actual_productivity_t_ha"].to_numpy(),
            other["predicted_productivity_t_ha"].to_numpy(),
            rng=rng,
        ),
    }
    # Feature-wise CV in band vs all
    cv_rows = []
    for c in cols:
        a = feat[c].to_numpy()
        b = band[c].to_numpy()
        cv_rows.append(
            {
                "feature": c,
                "std_all": float(np.std(a)),
                "std_band": float(np.std(b)),
                "std_band_over_all": float(np.std(b) / (np.std(a) + 1e-12)),
                "cv_band": float(np.std(b) / (abs(np.mean(b)) + 1e-12)),
            }
        )

    print("\n=== Input pairwise distances (z-scored features) ===")
    for k, st in input_stats.items():
        print(
            f"  {k:14s} L2 median={st['l2_median']:.3f}  "
            f"cos median={st['cosine_median']:.3f}  "
            f"frac cos<0.1={st['frac_pairs_cosine_lt_0.1']:.2f}  "
            f"ρ(L2,Δact)={st['spearman_l2_vs_dact']:+.3f}"
        )

    # ---- 2) Encoder embeddings ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    rows = collect_embeddings(device)
    print(f"Encoded {len(rows)} muni-years")

    actual = np.array([r["actual"] for r in rows])
    pred = np.array([r["pred_model"] for r in rows])
    pred_csv = np.array([r["pred_csv"] for r in rows])
    print(
        f"pred_model vs pred_csv: MAE={np.mean(np.abs(pred - pred_csv)):.4f} "
        f"corr={np.corrcoef(pred, pred_csv)[0,1]:.4f}"
    )

    e_spec = zscore(stack_field(rows, "e_spec"))
    e_clim = zscore(stack_field(rows, "e_clim"))
    e_fused = zscore(stack_field(rows, "e_fused"))
    in_band = (pred >= PRED_LO) & (pred <= PRED_HI)

    emb_stats = {}
    for name, mat in (
        ("input_feats", x_all),  # same order? NO - different order. skip cross.
        ("e_spec", e_spec),
        ("e_clim", e_clim),
        ("e_fused", e_fused),
    ):
        if name == "input_feats":
            continue
        emb_stats[f"{name}_all"] = pairwise_stats(mat, actual, pred, rng=rng)
        emb_stats[f"{name}_band"] = pairwise_stats(
            mat[in_band], actual[in_band], pred[in_band], rng=rng
        )

    # Collapse ratios: variance of preds vs variance of embeddings among band
    # Effective dimensionality / participation ratio of embeddings
    def participation_ratio(z: np.ndarray) -> float:
        # z already zscored columns; use covariance eigenvalues
        c = np.cov(z, rowvar=False)
        w = np.linalg.eigvalsh(c)
        w = np.clip(w, 0, None)
        return float((w.sum() ** 2) / ((w**2).sum() + 1e-12))

    # Within band: normalize each space so median pairwise L2 = 1, then compare
    # relative spread of predictions vs embeddings
    def relative_pred_collapse(mat, mask):
        st = pairwise_stats(mat[mask], actual[mask], pred[mask], rng=rng, n_pairs=8000)
        pred_std = float(np.std(pred[mask]))
        act_std = float(np.std(actual[mask]))
        # per-dim std of embeddings
        dim_std = float(np.mean(np.std(mat[mask], axis=0)))
        return {
            **st,
            "pred_std": pred_std,
            "actual_std": act_std,
            "mean_dim_std": dim_std,
            "pred_std_over_actual_std": pred_std / (act_std + 1e-12),
            "participation_ratio": participation_ratio(mat[mask]),
        }

    collapse = {
        "e_spec_band": relative_pred_collapse(e_spec, in_band),
        "e_clim_band": relative_pred_collapse(e_clim, in_band),
        "e_fused_band": relative_pred_collapse(e_fused, in_band),
        "e_fused_all": relative_pred_collapse(e_fused, np.ones(len(rows), dtype=bool)),
    }

    # Linear probe: can fused embedding linearly predict actual / pred?
    def ridge_r2(x, y):
        # simple centered least squares
        x = x - x.mean(0)
        y = y - y.mean()
        # ridge
        lam = 1.0
        xtx = x.T @ x + lam * np.eye(x.shape[1])
        w = np.linalg.solve(xtx, x.T @ y)
        yhat = x @ w
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum(y**2) + 1e-12
        return float(1 - ss_res / ss_tot)

    probes = {
        "fused_band_r2_actual": ridge_r2(e_fused[in_band], actual[in_band]),
        "fused_band_r2_pred": ridge_r2(e_fused[in_band], pred[in_band]),
        "fused_all_r2_actual": ridge_r2(e_fused, actual),
        "fused_all_r2_pred": ridge_r2(e_fused, pred),
        "spec_band_r2_actual": ridge_r2(e_spec[in_band], actual[in_band]),
        "clim_band_r2_actual": ridge_r2(e_clim[in_band], actual[in_band]),
        "spec_band_r2_pred": ridge_r2(e_spec[in_band], pred[in_band]),
        "clim_band_r2_pred": ridge_r2(e_clim[in_band], pred[in_band]),
    }

    # Decoder sensitivity: for band samples, perturb fused emb toward another
    # band sample and see if pred changes — approximate via Jacobian of last layers
    # Simpler: among band, correlation of pairwise fused L2 with pairwise |Δpred|
    print("\n=== Embedding pairwise (band) ===")
    for k in ("e_spec_band", "e_clim_band", "e_fused_band"):
        st = emb_stats[k]
        print(
            f"  {k:14s} L2 med={st['l2_median']:.3f}  cos med={st['cosine_median']:.3f}  "
            f"frac cos<0.1={st['frac_pairs_cosine_lt_0.1']:.2f}  "
            f"ρ(L2,Δact)={st['spearman_l2_vs_dact']:+.3f}  "
            f"ρ(L2,Δpred)={st['spearman_l2_vs_dpred']:+.3f}"
        )

    print("\n=== Linear probes (ridge) ===")
    for k, v in probes.items():
        print(f"  {k}: {v:.3f}")

    # Compare input feat distances for the SAME rows as embeddings
    feat_align = feat.copy()
    feat_align["municipality_code"] = feat_align["municipality_code"].astype(str)
    feat_idx = feat_align.set_index(["municipality_code", "harvest_year"])
    aligned = []
    for r in rows:
        key = (str(r["municipality_code"]), int(r["harvest_year"]))
        aligned.append(feat_idx.loc[key, cols].to_numpy(dtype=np.float64))
    x_aligned = zscore(np.stack(aligned))
    input_aligned_stats = {
        "all": pairwise_stats(x_aligned, actual, pred, rng=rng),
        "band": pairwise_stats(x_aligned[in_band], actual[in_band], pred[in_band], rng=rng),
    }
    print("\n=== Aligned input pairwise ===")
    for k, st in input_aligned_stats.items():
        print(
            f"  input_{k:4s} L2 med={st['l2_median']:.3f}  cos med={st['cosine_median']:.3f}  "
            f"frac cos<0.1={st['frac_pairs_cosine_lt_0.1']:.2f}  "
            f"ρ(L2,Δact)={st['spearman_l2_vs_dact']:+.3f}  "
            f"ρ(L2,Δpred)={st['spearman_l2_vs_dpred']:+.3f}"
        )

    # Ratio: how much does encoder shrink distances relative to inputs?
    # Compare median cosine of inputs vs fused embeddings in band
    shrink = {
        "input_band_cos_median": input_aligned_stats["band"]["cosine_median"],
        "fused_band_cos_median": emb_stats["e_fused_band"]["cosine_median"],
        "spec_band_cos_median": emb_stats["e_spec_band"]["cosine_median"],
        "clim_band_cos_median": emb_stats["e_clim_band"]["cosine_median"],
        "input_band_frac_cos_lt_0.1": input_aligned_stats["band"][
            "frac_pairs_cosine_lt_0.1"
        ],
        "fused_band_frac_cos_lt_0.1": emb_stats["e_fused_band"][
            "frac_pairs_cosine_lt_0.1"
        ],
        "input_all_cos_median": input_aligned_stats["all"]["cosine_median"],
        "fused_all_cos_median": emb_stats["e_fused_all"]["cosine_median"],
    }

    # Decoder: among band, is fused distance predictive of pred distance?
    # If embeddings vary but pred doesn't track → decoder saturation
    decoder_signal = {
        "band_pred_std": float(np.std(pred[in_band])),
        "band_actual_std": float(np.std(actual[in_band])),
        "band_fused_mean_dim_std": float(np.mean(np.std(e_fused[in_band], axis=0))),
        "all_fused_mean_dim_std": float(np.mean(np.std(e_fused, axis=0))),
        "band_over_all_fused_dim_std": float(
            np.mean(np.std(e_fused[in_band], axis=0))
            / (np.mean(np.std(e_fused, axis=0)) + 1e-12)
        ),
        "rho_fusedL2_dpred_band": emb_stats["e_fused_band"]["spearman_l2_vs_dpred"],
        "rho_fusedL2_dact_band": emb_stats["e_fused_band"]["spearman_l2_vs_dact"],
        "rho_inputL2_dpred_band": input_aligned_stats["band"]["spearman_l2_vs_dpred"],
        "rho_inputL2_dact_band": input_aligned_stats["band"]["spearman_l2_vs_dact"],
    }

    summary = {
        "n_encoded": len(rows),
        "n_band": int(in_band.sum()),
        "input_feature_std_ratio": cv_rows,
        "input_pairwise": input_stats,
        "input_aligned_pairwise": input_aligned_stats,
        "embedding_pairwise": emb_stats,
        "collapse": collapse,
        "probes": probes,
        "shrink": shrink,
        "decoder_signal": decoder_signal,
        "pred_model_vs_csv_mae": float(np.mean(np.abs(pred - pred_csv))),
    }
    with open(OUT / "encoder_vs_decoder_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plot: pairwise distance histograms
    def sample_l2(mat, n=8000):
        n0 = len(mat)
        i = rng.integers(0, n0, size=n)
        j = rng.integers(0, n0, size=n)
        keep = i != j
        return np.linalg.norm(mat[i[keep]] - mat[j[keep]], axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = [
        (axes[0, 0], x_aligned[in_band], "Input features (band)", "#4c78a8"),
        (axes[0, 1], e_spec[in_band], "Spectral encoder (band)", "#f58518"),
        (axes[1, 0], e_clim[in_band], "Climate encoder (band)", "#54a24b"),
        (axes[1, 1], e_fused[in_band], "Fused pre-decoder (band)", "#e45756"),
    ]
    for ax, mat, title, color in panels:
        d = sample_l2(mat)
        ax.hist(d, bins=60, color=color, alpha=0.85, edgecolor="white", lw=0.3)
        ax.axvline(np.median(d), color="0.15", ls="--", lw=1, label=f"med={np.median(d):.2f}")
        ax.set_title(title)
        ax.set_xlabel("Pairwise L2 (column-zscored)")
        ax.set_ylabel("Pair count")
        ax.legend(frameon=False, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(
        "PR trial_005 · distance among munis with pred ∈ [3.50, 3.75]\n"
        "Wider histogram ⇒ less collapse at that stage",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT / "pairwise_l2_input_vs_embeddings_band.png", dpi=160)
    plt.close(fig)

    # Scatter: fused pairwise L2 vs |Δpred| and |Δactual| for band
    def pair_clouds(mat, mask, n=5000):
        m = mat[mask]
        a = actual[mask]
        p = pred[mask]
        i = rng.integers(0, len(m), size=n)
        j = rng.integers(0, len(m), size=n)
        keep = i != j
        i, j = i[keep], j[keep]
        l2 = np.linalg.norm(m[i] - m[j], axis=1)
        return l2, np.abs(a[i] - a[j]), np.abs(p[i] - p[j])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    l2, da, dp = pair_clouds(e_fused, in_band)
    axes[0].scatter(l2, da, s=6, alpha=0.25, c="#4c78a8", rasterized=True)
    axes[0].set_xlabel("Fused embedding L2")
    axes[0].set_ylabel("|Δ actual| (t/ha)")
    axes[0].set_title(
        f"Embeddings vs actual gap  ρ={stats.spearmanr(l2, da).correlation:+.2f}"
    )
    axes[1].scatter(l2, dp, s=6, alpha=0.25, c="#e45756", rasterized=True)
    axes[1].set_xlabel("Fused embedding L2")
    axes[1].set_ylabel("|Δ prediction| (t/ha)")
    axes[1].set_title(
        f"Embeddings vs pred gap  ρ={stats.spearmanr(l2, dp).correlation:+.2f}"
    )
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Within pred band: do embedding distances move the head?", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fused_distance_vs_deltas_band.png", dpi=160)
    plt.close(fig)

    print(f"\nWrote {OUT / 'encoder_vs_decoder_summary.json'}")
    print("shrink:", json.dumps(shrink, indent=2))
    print("decoder_signal:", json.dumps(decoder_signal, indent=2))
    print("probes:", json.dumps(probes, indent=2))


if __name__ == "__main__":
    main()
