"""Audit what the model actually ingests, per municipality-year.

Reads a run's training/config.json, rebuilds the exact PixelTransform used during
training, and dumps raw-.npy stats, post-transform tensor stats, cross-year
identity checks, a target-vs-NDVI signal check and (optionally) model predictions.

    python debug_model_inputs.py --run-dir results/tuning/<study>/trial_005

Writes <out>/muni_year_audit.csv plus a printed report.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import polars as pl


def _install_torch_stub() -> bool:
    """Let the audit run without torch installed.

    PixelTransform is pure numpy except for the final ``torch.from_numpy`` wrap,
    and datautils only touches torch inside functions we never call. Stubbing
    ``from_numpy`` as identity keeps the real transform code on the audit path;
    the returned arrays are numpy, which the stats helpers already expect.
    Returns True when a stub was installed (i.e. --predict is unavailable).
    """
    try:
        import torch  # noqa: F401

        return False
    except ImportError:
        pass

    class _Arr(np.ndarray):
        def numpy(self):
            return np.asarray(self)

    stub = types.ModuleType("torch")
    stub.from_numpy = lambda a: np.asarray(a).view(_Arr)
    stub.Tensor = _Arr
    stub.__getattr__ = lambda name: (_ for _ in ()).throw(
        RuntimeError(f"torch.{name} needs real torch; install it for --predict")
    )
    sys.modules["torch"] = stub
    return True


TORCH_STUBBED = _install_torch_stub()

# Bypass datasets/__init__.py (pulls in the whole training stack) and import the
# two modules the audit needs straight from the package directory.
_pkg = types.ModuleType("datasets")
_pkg.__path__ = [str(Path(__file__).resolve().parent / "datasets")]
sys.modules.setdefault("datasets", _pkg)

from datasets.extra_scaler import InputScaler  # noqa: E402
from datasets.pixel_transform import (  # noqa: E402
    DOY_CHANNEL,
    NO_DATA_VALUE,
    NUM_SPECTRAL_CHANNELS,
    PixelTransform,
)

RED, NIR = 2, 6  # B4, B8 within the 10-band stack


# --------------------------------------------------------------------------- config


def load_run_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "training" / "config.json"
    if not cfg_path.is_file():
        raise SystemExit(f"No config.json at {cfg_path}")
    sessions = json.loads(cfg_path.read_text())["sessions"]
    last = sessions[-1]
    return {"cli": last["cli"], "computed": last["computed"], "path": cfg_path}


def localize(p: str | None) -> Path | None:
    """Map a path recorded under WSL/Linux onto whatever host runs this script."""
    if not p:
        return None
    cand = Path(p)
    if cand.exists():
        return cand
    s = str(p)
    if s.startswith("/mnt/") and len(s) > 6:  # /mnt/c/... -> C:/...
        alt = Path(f"{s[5].upper()}:/{s[7:]}")
        if alt.exists():
            return alt
    return cand


# ----------------------------------------------------------------- npy path lookup


def index_seasons(root: Path) -> dict[int, list[Path]]:
    """harvest year -> season dirs, mirroring USCropsAggregatedNPY's rule (Y1-Y2 -> Y2)."""
    by_year: dict[int, list[Path]] = {}
    for item in sorted(root.iterdir()):
        if not item.is_dir():
            continue
        name = item.name
        if name.isdigit() and len(name) == 4:
            by_year.setdefault(int(name), []).append(item)
        elif "-" in name:
            a, _, b = name.partition("-")
            if a.isdigit() and b.isdigit() and len(a) == 4 and len(b) == 4:
                by_year.setdefault(int(b), []).append(item)
    return by_year


def resolve_npy(by_year: dict[int, list[Path]], root: Path, code: str, year: int):
    for d in by_year.get(year, []) or ([root] if not by_year else []):
        for cand in (d / code / f"{code}.npy", d / f"{code}.npy"):
            if cand.is_file():
                return cand
    return None


# ------------------------------------------------------------------------- metrics


def raw_stats(arr: np.ndarray) -> dict:
    """Stats on the untouched .npy contents (N, T, C)."""
    spec = arr[:, :, :NUM_SPECTRAL_CHANNELS].astype(np.float32)
    doy = arr[:, :, DOY_CHANNEL].astype(np.int64)
    bad = (spec == NO_DATA_VALUE) | ~np.isfinite(spec)
    day_all_bad = bad.all(axis=2)  # (N, T) fully-missing observations
    day_any_bad = bad.any(axis=2)
    valid = ~bad

    refl = np.where(valid, spec, np.nan) * 1e-4
    with np.errstate(invalid="ignore"):
        nir, red = refl[:, :, NIR], refl[:, :, RED]
        ndvi = (nir - red) / (nir + red + 1e-8)
    ok = np.isfinite(ndvi) & ~day_all_bad

    return {
        "n_pixels": int(arr.shape[0]),
        "n_timesteps": int(arr.shape[1]),
        "n_channels": int(arr.shape[2]),
        "frac_days_all_nodata": float(day_all_bad.mean()),
        "frac_days_any_nodata": float(day_any_bad.mean()),
        "n_valid_days_mean": float((~day_all_bad).sum(axis=1).mean()),
        "doy_min": int(doy.min()),
        "doy_max": int(doy.max()),
        "n_unique_doy": int(np.unique(doy).size),
        "refl_mean": float(np.nanmean(refl)) if valid.any() else float("nan"),
        "refl_max": float(np.nanmax(refl)) if valid.any() else float("nan"),
        "ndvi_mean": float(np.nanmean(ndvi[ok])) if ok.any() else float("nan"),
        "ndvi_p95": float(np.nanpercentile(ndvi[ok], 95)) if ok.any() else float("nan"),
        "ndvi_peak": float(np.nanmax(ndvi[ok])) if ok.any() else float("nan"),
        "ndvi_integral": float(np.nansum(np.clip(ndvi, 0, None)) / max(arr.shape[0], 1))
        if ok.any()
        else float("nan"),
    }


def tensor_stats(tf: PixelTransform, arr: np.ndarray, seed: int) -> tuple[dict, np.ndarray]:
    """Stats on the exact tensor handed to STNetRegression, plus that tensor."""
    np.random.seed(seed)  # transform_chunk subsamples via np.random when T > seq_len
    x, mask, doy, weight = tf.transform_chunk(np.ascontiguousarray(arr, np.float32))
    xn = x.numpy()
    mn = mask.numpy()
    wn = weight.numpy()

    # A day that was nodata is exactly 0 reflectance pre-z-score, i.e. -mean/std after.
    zero_day_z = (-tf.mean.reshape(-1) / tf.std.reshape(-1)).astype(np.float32)
    spec = xn[:, :, :NUM_SPECTRAL_CHANNELS]
    is_zero_day = np.isclose(spec, zero_day_z, atol=1e-3).all(axis=2)
    real = ~is_zero_day & ~mn

    stats = {
        "seq_len_out": int(xn.shape[1]),
        "feat_dim_out": int(xn.shape[2]),
        "frac_padded": float(mn.mean()),
        "frac_zerofill_days": float(is_zero_day.mean()),
        "frac_real_days": float(real.mean()),
        "x_mean": float(xn.mean()),
        "x_std": float(xn.std()),
        "x_absmax": float(np.abs(xn).max()),
        "frac_abs_z_gt5": float((np.abs(xn) > 5).mean()),
        "x_nan": int(np.isnan(xn).sum()),
        "weight_on_zerofill": float(wn[is_zero_day].sum() / max(wn.sum(), 1e-9)),
        "weight_on_padding": float(wn[mn].sum() / max(wn.sum(), 1e-9)),
        "doy_out_min": int(doy.numpy().min()),
        "doy_out_max": int(doy.numpy().max()),
    }
    return stats, xn


# ---------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=250, help="municipality-years to audit")
    ap.add_argument(
        "--max-pixels",
        type=int,
        default=256,
        help="pixels sampled per municipality (pixel-level runs have thousands)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--datapath", type=Path, default=None, help="override .npy root")
    ap.add_argument("--yield-csv", type=Path, default=None)
    ap.add_argument("--predict", action="store_true", help="also run model_best.pth")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = load_run_config(args.run_dir)
    cli, comp = cfg["cli"], cfg["computed"]
    out_dir = args.out or (args.run_dir / "debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    root = args.datapath or localize(cli["datapath"])
    yield_csv = args.yield_csv or localize(cli["yield_csv"])
    scaler_path = localize(comp.get("extra_scaler_path"))
    layout = comp.get("feature_layout", cli.get("feature_layout", "spectral"))
    seq_len = int(cli["sequencelength"])
    tgt_col = comp.get("yield_target_column", "yield_t_ha")
    tgt_mean, tgt_std = comp.get("target_mean"), comp.get("target_std")

    print("=" * 78)
    print(f"RUN            {args.run_dir}")
    print(f"datapath       {root}   exists={root is not None and root.exists()}")
    print(f"yield csv      {yield_csv}")
    print(f"scaler         {scaler_path}")
    print(f"layout         {layout}   seq_len={seq_len}   head={comp.get('head_output')}")
    print(f"target         {tgt_col}  mean={tgt_mean:.4f}  std={tgt_std:.4f}")
    print(f"rc={cli.get('rc')}  interp={cli.get('interp')}")
    print("=" * 78)

    if root is None or not root.exists():
        raise SystemExit(f"datapath not reachable from this host: {root}")

    # ---------------------------------------------------------------- 1. targets
    df = pl.read_csv(yield_csv, null_values=["-", "", "nan", "NaN", "null", "NULL"])
    df = df.with_columns(pl.col("municipality_code").cast(pl.Utf8))
    years = comp.get("imagery_years_used") or sorted(df["year"].unique().to_list())
    df = df.filter(pl.col("year").is_in(years) & pl.col(tgt_col).is_not_null())

    print("\n[1] TARGET DISTRIBUTION (does the training set contain high yields?)")
    split_col = "split" if "split" in df.columns else None
    group = ["year", split_col] if split_col else ["year"]
    summary = (
        df.group_by(group)
        .agg(
            pl.len().alias("n"),
            pl.col(tgt_col).mean().round(3).alias("mean"),
            pl.col(tgt_col).std().round(3).alias("std"),
            pl.col(tgt_col).min().round(3).alias("min"),
            pl.col(tgt_col).quantile(0.95).round(3).alias("p95"),
            pl.col(tgt_col).max().round(3).alias("max"),
        )
        .sort(group)
    )
    print(summary)
    if split_col:
        tr = df.filter(pl.col(split_col) == "train")[tgt_col].to_numpy()
        ev = df.filter(pl.col(split_col) != "train")[tgt_col].to_numpy()
        if tr.size and ev.size:
            print(
                f"    train  max={tr.max():.2f}  p99={np.percentile(tr, 99):.2f}  "
                f"n>{tgt_mean + 2 * tgt_std:.2f} = {(tr > tgt_mean + 2 * tgt_std).sum()}"
            )
            print(
                f"    eval   max={ev.max():.2f}  p99={np.percentile(ev, 99):.2f}  "
                f"n>{tgt_mean + 2 * tgt_std:.2f} = {(ev > tgt_mean + 2 * tgt_std).sum()}"
            )

    # ------------------------------------------------------- 2/3. per-muni inputs
    scaler = InputScaler.require_load(scaler_path) if scaler_path else None
    tf = PixelTransform(
        sequencelength=seq_len,
        feature_layout=layout,
        randomchoice=bool(cli.get("rc")),
        interp=bool(cli.get("interp")),
        extra_scaler=scaler,
        extra_scaler_path=scaler_path,
    )
    by_year = index_seasons(root)
    print(f"\n[2] SEASON DIRS: {{y: [d.name for d in v] for ...}} -> "
          f"{ {y: [d.name for d in v] for y, v in sorted(by_year.items())} }")

    pairs = df.select(["municipality_code", "year", tgt_col] + ([split_col] if split_col else []))
    pairs = pairs.sample(min(args.limit, pairs.height), seed=args.seed, shuffle=True)

    rows, tensors = [], {}
    missing = 0
    rng = np.random.default_rng(args.seed)
    for i, rec in enumerate(pairs.iter_rows(named=True)):
        code, year = rec["municipality_code"], int(rec["year"])
        p = resolve_npy(by_year, root, code, year)
        if p is None:
            missing += 1
            continue
        mm = np.load(p, mmap_mode="r")
        n_total = int(mm.shape[0])
        if n_total > args.max_pixels:
            sel = np.sort(rng.choice(n_total, args.max_pixels, replace=False))
            arr = np.asarray(mm[sel], dtype=np.float32)
        else:
            arr = np.asarray(mm, dtype=np.float32)
        row = {
            "municipality_code": code,
            "year": year,
            "split": rec.get(split_col, ""),
            tgt_col: rec[tgt_col],
            "npy": str(p),
            "season_dir": p.parent.parent.name if p.parent.name == code else p.parent.name,
            "n_pixels_total": n_total,
        }
        row.update(raw_stats(arr))
        ts, xn = tensor_stats(tf, arr, args.seed)
        row.update(ts)
        rows.append(row)
        tensors[(code, year)] = xn.mean(axis=0)
        if (i + 1) % 50 == 0:
            print(f"    ...{i + 1}/{pairs.height}", flush=True)

    audit = pl.DataFrame(rows)
    audit.write_csv(out_dir / "muni_year_audit.csv")
    print(f"\n[3] INPUT AUDIT  n={audit.height}  (missing .npy: {missing})")
    print(f"    written to {out_dir / 'muni_year_audit.csv'}")

    def col_report(name: str, fmt: str = "{:.4f}") -> None:
        if name not in audit.columns:
            return
        v = audit[name].drop_nulls().to_numpy()
        if v.size == 0:
            return
        print(
            f"    {name:<24} min={fmt.format(v.min())}  med={fmt.format(np.median(v))}  "
            f"max={fmt.format(v.max())}"
        )

    for c in [
        "n_pixels", "n_timesteps", "n_valid_days_mean",
        "frac_days_all_nodata", "frac_zerofill_days", "frac_real_days",
        "weight_on_zerofill", "frac_padded", "x_std", "x_absmax",
        "frac_abs_z_gt5", "ndvi_peak", "ndvi_mean",
    ]:
        col_report(c)

    # loud flags
    print("\n[4] FLAGS")
    t_over = (audit["n_timesteps"] > seq_len).mean()
    if t_over > 0:
        print(
            f"    !! {t_over:.0%} of files have T > sequencelength ({seq_len}); "
            f"transform_chunk RANDOMLY SUBSAMPLES {seq_len} days per call "
            f"(non-deterministic at eval too)."
        )
    zf = audit["frac_zerofill_days"].mean()
    if zf > 0.05:
        print(
            f"    !! mean {zf:.1%} of ingested timesteps are nodata days zero-filled "
            f"then z-scored. They are NOT masked and carry "
            f"{audit['weight_on_zerofill'].mean():.1%} of the pooling weight."
        )
    if audit["x_nan"].sum() > 0:
        print(f"    !! NaNs in model input: {audit['x_nan'].sum()}")
    if audit["frac_abs_z_gt5"].mean() > 0.01:
        print(f"    !! {audit['frac_abs_z_gt5'].mean():.1%} of input values have |z| > 5.")

    # --------------------------------------------- 5. same muni, different years?
    print("\n[5] CROSS-YEAR VARIATION (do different years give different inputs?)")
    byc: dict[str, list[int]] = {}
    for code, year in tensors:
        byc.setdefault(code, []).append(year)
    diffs = []
    for code, yrs in byc.items():
        for i in range(len(yrs)):
            for j in range(i + 1, len(yrs)):
                a, b = tensors[(code, yrs[i])], tensors[(code, yrs[j])]
                if a.shape == b.shape:
                    diffs.append(float(np.abs(a - b).mean()))
    if diffs:
        d = np.array(diffs)
        print(f"    mean |Δx| across year pairs: min={d.min():.5f} med={np.median(d):.5f} max={d.max():.5f}")
        if d.min() < 1e-6:
            print("    !! Some municipality-years share IDENTICAL inputs -> "
                  "the model cannot separate those years.")
    else:
        print("    (not enough municipalities sampled twice; raise --limit)")

    # --------------------------------------------------- 6. per-year input shape
    print("\n[6] PER-YEAR INPUT AVAILABILITY (is a 'year effect' encoded as data volume?)")
    print(
        audit.group_by("year")
        .agg(
            pl.len().alias("n"),
            pl.col("n_timesteps").median().alias("T_med"),
            pl.col("n_timesteps").min().alias("T_min"),
            pl.col("n_timesteps").max().alias("T_max"),
            pl.col("frac_padded").mean().round(3).alias("padded"),
            pl.col("ndvi_peak").mean().round(3).alias("ndvi_peak"),
            pl.col(tgt_col).mean().round(3).alias("yield"),
        )
        .sort("year")
    )

    # ------------------------------------------------------- 7. is there signal?
    def corrs(sub: pl.DataFrame, label: str) -> None:
        yv = sub[tgt_col].to_numpy().astype(float)
        parts = []
        for feat in ["ndvi_peak", "ndvi_p95", "ndvi_mean", "n_timesteps", "refl_mean"]:
            v = sub[feat].to_numpy().astype(float)
            m = np.isfinite(v) & np.isfinite(yv)
            if m.sum() < 10 or np.std(v[m]) == 0:
                parts.append(f"{feat}=  n/a")
                continue
            parts.append(f"{feat}={np.corrcoef(v[m], yv[m])[0, 1]:+.3f}")
        print(f"    {label:<14} n={sub.height:4d}  " + "  ".join(parts))

    print("\n[7] SIGNAL CHECK: pearson(input summary, target)")
    corrs(audit, "POOLED")
    for yv in sorted(audit["year"].unique().to_list()):
        corrs(audit.filter(pl.col("year") == yv), f"within {yv}")
    print(
        "    Pooled correlation mostly reflects BETWEEN-year differences.\n"
        "    The within-year rows are what the model needs to rank municipalities;\n"
        "    if those are near zero, predictions collapse toward a per-year constant."
    )

    # ------------------------------------------------------------ 7. predictions
    if args.predict:
        if TORCH_STUBBED:
            print("\n[7] skipped: torch is not installed in this environment.")
        else:
            predict_report(args, cfg, tf, audit, tgt_col, out_dir)


def predict_report(args, cfg, tf, audit, tgt_col, out_dir) -> None:
    import torch

    from models.STNetRegression import STNetRegression
    from utils_aggregated import denormalize_head_output

    comp = cfg["computed"]
    ckpt_path = args.run_dir / "training" / "model_best.pth"
    if not ckpt_path.is_file():
        print(f"\n[7] no checkpoint at {ckpt_path}")
        return

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck.get("model_state", ck)
    kw = ck.get("model_kwargs") or comp.get("model_kwargs", {})
    model = STNetRegression(
        input_dim=comp.get("input_dim", 10),
        num_outputs=comp.get("num_outputs", 1),
        max_seq_len=int(cfg["cli"]["sequencelength"]),
        **kw,
    )
    model.load_state_dict(state, strict=False)
    model.eval().to(args.device)

    head = ck.get("head_output", comp.get("head_output", "zscore"))
    mu = ck.get("target_mean", comp.get("target_mean"))
    sd = ck.get("target_std", comp.get("target_std"))

    preds, truth, yrs = [], [], []
    rng = np.random.default_rng(args.seed)
    with torch.no_grad():
        for i, rec in enumerate(audit.iter_rows(named=True)):
            p = Path(rec["npy"])
            if not p.is_file():
                continue
            mm = np.load(p, mmap_mode="r")
            n_total = int(mm.shape[0])
            if n_total > args.max_pixels:
                sel = np.sort(rng.choice(n_total, args.max_pixels, replace=False))
                arr = np.asarray(mm[sel], dtype=np.float32)
            else:
                arr = np.asarray(mm, dtype=np.float32)
            np.random.seed(args.seed)
            batch = [t.to(args.device) for t in tf.transform_chunk(arr)]
            out = model(tuple(batch)).mean(dim=0)
            val = denormalize_head_output(out, mu, sd, head).cpu().numpy().ravel()[0]
            preds.append(float(val))
            truth.append(float(rec[tgt_col]))
            yrs.append(int(rec["year"]))
            if (i + 1) % 50 == 0:
                print(f"    ...{i + 1}/{audit.height}", flush=True)

    pr, tr, yr = np.array(preds), np.array(truth), np.array(yrs)
    pl.DataFrame({"year": yr, "pred": pr, "true": tr}).write_csv(
        out_dir / "predictions_debug.csv"
    )

    print(f"\n[8] PREDICTIONS  n={pr.size}  -> {out_dir / 'predictions_debug.csv'}")
    print(f"    pred   mean={pr.mean():.3f} std={pr.std():.3f} min={pr.min():.3f} max={pr.max():.3f}")
    print(f"    true   mean={tr.mean():.3f} std={tr.std():.3f} min={tr.min():.3f} max={tr.max():.3f}")
    print(f"    train target mean was {mu:.3f} (std {sd:.3f})")
    print(f"    spread ratio std(pred)/std(true) = {pr.std() / max(tr.std(), 1e-9):.3f}")
    if pr.size > 2 and pr.std() > 1e-9:
        print(f"    corr(pred, true) = {np.corrcoef(pr, tr)[0, 1]:+.3f}")
        print(f"    slope of true~pred = {np.polyfit(pr, tr, 1)[0]:.3f}   (1.0 = calibrated)")
    print(f"    15 largest predictions: {np.round(np.sort(pr)[-15:], 3).tolist()}")
    print(f"    distinct predictions @0.01: {np.unique(np.round(pr, 2)).size} / {pr.size}")

    print("\n    per-year predicted vs true mean:")
    for y in sorted(set(yrs)):
        m = yr == y
        print(
            f"      {y}:  pred {pr[m].mean():.3f} (sd {pr[m].std():.3f}, "
            f"range {pr[m].min():.2f}-{pr[m].max():.2f})   "
            f"true {tr[m].mean():.3f} (sd {tr[m].std():.3f})"
        )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        bins = np.linspace(
            min(pr.min(), tr.min()) - 0.1, max(pr.max(), tr.max()) + 0.1, 60
        )
        ax[0].hist(tr, bins=bins, alpha=0.6, label="true", color="tab:green")
        ax[0].hist(pr, bins=bins, alpha=0.6, label="predicted", color="tab:red")
        ax[0].axvline(mu, ls="--", c="k", lw=1, label=f"train mean {mu:.2f}")
        ax[0].set_xlabel(tgt_col)
        ax[0].set_ylabel("count")
        ax[0].set_title("Prediction vs target distribution")
        ax[0].legend()

        ax[1].scatter(tr, pr, s=12, alpha=0.5)
        lo, hi = bins[0], bins[-1]
        ax[1].plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
        ax[1].axhline(mu, ls=":", c="r", lw=1, label="train mean")
        ax[1].set_xlabel("true")
        ax[1].set_ylabel("predicted")
        ax[1].set_title("Predicted vs true")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(out_dir / "prediction_distribution.png", dpi=130)
        print(f"\n    plot -> {out_dir / 'prediction_distribution.png'}")
    except Exception as e:
        print(f"    (plot skipped: {e})")


if __name__ == "__main__":
    main()
