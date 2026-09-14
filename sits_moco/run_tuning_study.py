#!/usr/bin/env python3
"""
Structured hyperparameter tuning for yield/productivity regression.

Example:
  python run_tuning_study.py run tuning/studies/productivity_baseline.yaml
  python run_tuning_study.py run tuning/studies/productivity_baseline.yaml --dry-run
  python run_tuning_study.py run tuning/studies/productivity_baseline.yaml --skip-completed
  python run_tuning_study.py summarize tuning/studies/productivity_baseline.yaml
  python run_tuning_study.py leaderboard tuning/studies/productivity_baseline.yaml --top 10

Resuming after an interrupted trial (e.g. trial_007 stopped mid-training):
  python run_tuning_study.py run tuning/studies/total_adj_architecture.yaml --skip-completed
  # Skips trials 001-006 (already ok), resumes 007 from checkpoint_epoch_N.pth via --pretrained.
  # Use --no-resume-incomplete to restart interrupted trials from epoch 1 instead.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tuning.collect import (
    aggregate_year_loo_folds,
    best_epoch_metrics,
    format_run_epochs,
    trial_row,
)
from tuning.config import load_study_config, merge_trial_params
from tuning.moco_ensure import ensure_moco_checkpoint, predict_moco_checkpoint_path
from tuning.resume import plan_trial_resume
from tuning.runner import apply_resume_params, predict_run_dir, run_trial_subprocess
from tuning.search import generate_trials
from run_paths import ensure_dir


def _safe_run_tag(name: str) -> str:
    """Filesystem-safe short tag for embedding the study name in run suffixes."""
    tag = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return tag or "study"


def study_output_dir(cfg: dict) -> Path:
    return Path(cfg["output_dir"]) / cfg["name"]


def _trial_id(index: int) -> str:
    return f"trial_{index:03d}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


def _duration_seconds(started: str | None, finished: str | None) -> float | None:
    if not started or not finished:
        return None
    try:
        start_dt = datetime.fromisoformat(started)
        finish_dt = datetime.fromisoformat(finished)
        return max(0.0, (finish_dt - start_dt).total_seconds())
    except (TypeError, ValueError):
        return None


def _print_trial_finished(trial_id: str, finished: str, started: str | None) -> None:
    duration = _duration_seconds(started, finished)
    duration_msg = f" ({_format_duration(duration)})" if duration is not None else ""
    print(f"[{trial_id}] finished at {finished}{duration_msg}")


def _save_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent, quiet=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _append_jsonl(path: Path, record: dict) -> None:
    ensure_dir(path.parent, quiet=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_trials_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _latest_trial_records(path: Path) -> dict[str, dict]:
    """Last jsonl line per trial_id wins (handles reruns and dry-runs)."""
    latest: dict[str, dict] = {}
    for rec in _load_trials_jsonl(path):
        trial_id = rec.get("trial_id")
        if trial_id:
            latest[trial_id] = rec
    return latest


def _trial_appears_trained(run_dir: Path) -> bool:
    """True when a run folder has a finished training artifact but tuning never recorded success."""
    training = run_dir / "training"
    return (training / "model_best.pth").is_file() and (training / "trainlog.csv").is_file()


def _finalize_trial_metrics(
    run_dir: Path,
    *,
    objective: str,
    objective_spec: dict,
) -> dict:
    metrics = best_epoch_metrics(
        run_dir,
        objective=objective,
        objective_spec=objective_spec,
    )
    out = {
        "run_dir": str(run_dir.resolve()),
        "returncode": 0,
        "finalized_without_training": True,
        **metrics,
    }
    if objective_spec.get("log") == "testlog":
        val_side = best_epoch_metrics(
            run_dir,
            objective="val_r2",
            objective_spec={"mode": "max", "column": "r2"},
        )
        if val_side.get("status") == "ok":
            out["metrics"] = val_side.get("metrics") or {}
    return out


def _run_one_yield_job(
    *,
    label: str,
    merged: dict,
    run_dir: Path,
    log_path: Path,
    cfg: dict,
    args: argparse.Namespace,
    training_script: Path,
    repo_root: Path,
    resume_incomplete: bool,
) -> dict:
    """Train one yield run (a trial, or one year-LOO fold). Returns an outcome dict."""
    resume_plan = None
    append_log = False
    train_params = merged

    if resume_incomplete:
        resume_plan = plan_trial_resume(
            run_dir,
            target_epochs=int(merged.get("epochs", 100)),
            early_stop_patience=int(merged.get("early_stop_patience", 0)) or None,
        )
        if resume_plan is not None:
            train_params = apply_resume_params(merged, resume_plan["checkpoint"])
            append_log = log_path.is_file()
            if not args.dry_run:
                epoch = resume_plan.get("epoch")
                es = resume_plan.get("early_stop") or {}
                not_improved = es.get("not_improved_count")
                best_ep = es.get("best_epoch")
                msg = f"[{label}] resume from epoch {epoch} via {resume_plan['checkpoint']}"
                if not_improved is not None:
                    msg += (
                        f" (trainlog: best epoch {best_ep}, not_improved={not_improved})"
                    )
                print(msg)
        elif _trial_appears_trained(run_dir) and not args.dry_run:
            print(f"[{label}] training artifacts found; collecting metrics only")
            return _finalize_trial_metrics(
                run_dir,
                objective=cfg["objective"],
                objective_spec=cfg["objective_spec"],
            )
        elif (run_dir / "training" / "trainlog.csv").is_file() and not args.dry_run:
            print(
                f"[{label}] WARNING: trainlog exists but no checkpoint found; "
                "starting fresh (earlier epoch weights cannot be restored)"
            )

    moco_cfg = cfg.get("moco")
    if moco_cfg and resume_plan is None:
        expected_moco = predict_moco_checkpoint_path(
            merged, moco_cfg, repo_root=repo_root
        )
        if args.dry_run:
            print(f"[{label}] would ensure MoCo: {expected_moco}")
            train_params = dict(train_params)
            train_params["pretrained"] = str(expected_moco)
        else:
            try:
                moco_ckpt = ensure_moco_checkpoint(
                    merged,
                    moco_cfg,
                    repo_root=repo_root,
                    dry_run=False,
                    capture_log=log_path.parent / "moco_stdout.log",
                )
            except RuntimeError as exc:
                print(f"[{label}] FAILED MoCo ensure: {exc}")
                return {
                    "status": "moco_failed",
                    "error": str(exc),
                    "returncode": 1,
                    "run_dir": str(run_dir.resolve()),
                }
            print(f"[{label}] MoCo pretrained: {moco_ckpt}")
            train_params = dict(train_params)
            train_params["pretrained"] = str(moco_ckpt)

    if args.dry_run:
        if resume_plan is not None:
            epoch = resume_plan.get("epoch")
            print(f"[{label}] would resume from epoch {epoch} via {resume_plan['checkpoint']}")
        outcome = run_trial_subprocess(
            train_params,
            training_script=training_script,
            repo_root=repo_root,
            dry_run=True,
        )
        print(f"[{label}] argv: {' '.join(outcome['argv'])}")
        return outcome

    proc_outcome = run_trial_subprocess(
        train_params,
        training_script=training_script,
        repo_root=repo_root,
        capture_log=log_path,
        append_log=append_log,
    )
    if proc_outcome["returncode"] != 0:
        print(f"[{label}] FAILED (returncode={proc_outcome['returncode']})")
        return {
            **proc_outcome,
            "status": "train_failed",
            "error": f"training exited with code {proc_outcome['returncode']}",
        }

    metrics = best_epoch_metrics(
        proc_outcome["run_dir"],
        objective=cfg["objective"],
        objective_spec=cfg["objective_spec"],
    )
    outcome = {**proc_outcome, **metrics}
    if cfg["objective_spec"].get("log") == "testlog":
        val_side = best_epoch_metrics(
            proc_outcome["run_dir"],
            objective="val_r2",
            objective_spec={"mode": "max", "column": "r2"},
        )
        if val_side.get("status") == "ok":
            outcome["metrics"] = val_side.get("metrics") or {}
    if metrics.get("status") == "ok":
        print(
            f"[{label}] OK  {cfg['objective']}={metrics['objective_value']:.6g}"
            f"{format_run_epochs(outcome)}"
        )
    else:
        print(f"[{label}] finished but metrics status={metrics.get('status')}")
    return outcome


def _year_loo_fold_params(merged: dict, fold: dict, *, trial_id: str, fold_dir: Path) -> dict:
    params = {
        key: value
        for key, value in merged.items()
        if key not in ("year_loo_folds", "skip", "skip_reason")
    }
    year = int(fold["holdout_year"])
    params["holdout_year"] = year
    if fold.get("extra_scaler"):
        params["extra_scaler"] = fold["extra_scaler"]
    suffix = str(params.get("suffix") or f"tune_{trial_id}")
    params["suffix"] = f"{suffix}_loo{year}"
    params["run_dir"] = str(fold_dir.resolve())
    return params


def _record_and_summarize(
    *,
    study_dir: Path,
    cfg: dict,
    trial_id: str,
    merged: dict,
    sampled: dict,
    outcome: dict,
    started: str,
    finished: str,
    resumed_from_checkpoint: str | None = None,
    status: str = "completed",
) -> None:
    record = {
        "trial_id": trial_id,
        "status": status,
        "params": merged,
        "sampled": sampled,
        "outcome": outcome,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "resumed_from_checkpoint": resumed_from_checkpoint,
    }
    _append_jsonl(study_dir / "trials.jsonl", record)
    df = _update_summary(study_dir)
    best = _pick_best(df, cfg["objective"], cfg["objective_spec"])
    if best:
        _save_json(study_dir / "best_trial.json", best)


def _trial_run_succeeded(rec: dict) -> bool:
    if rec.get("status") == "dry_run":
        return False
    outcome = rec.get("outcome") or {}
    return outcome.get("returncode", 1) == 0 and outcome.get("status") == "ok"


def _update_summary(study_dir: Path) -> pd.DataFrame:
    trials_path = study_dir / "trials.jsonl"
    rows = []
    for rec in _latest_trial_records(trials_path).values():
        rows.append(
            trial_row(
                rec.get("trial_id", ""),
                rec.get("params") or {},
                rec.get("outcome") or {},
                started_at_utc=rec.get("started_at_utc"),
                finished_at_utc=rec.get("finished_at_utc"),
            )
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(study_dir / "summary.csv", index=False)
    return df


def _pick_best(df: pd.DataFrame, objective: str, spec: dict) -> dict | None:
    if df.empty or "objective_value" not in df.columns:
        return None
    valid = df[df["status"] == "ok"].dropna(subset=["objective_value"])
    if valid.empty:
        return None
    mode = spec.get("mode", "max")
    idx = valid["objective_value"].idxmin() if mode == "min" else valid["objective_value"].idxmax()
    row = valid.loc[idx]
    return row.to_dict()


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_study_config(args.config)
    study_dir = study_output_dir(cfg)
    ensure_dir(study_dir)

    # Snapshot study config (copyfile: WSL /mnt/c rejects copy2 utime)
    shutil.copyfile(args.config, study_dir / "study_config.yaml")

    trials = generate_trials({**cfg["search"], "base": cfg["base"]})
    if args.max_trials is not None:
        trials = trials[: int(args.max_trials)]

    print(f"Study: {cfg['name']}")
    print(f"Objective: {cfg['objective']} ({cfg['objective_spec']['mode']} {cfg['objective_spec']['column']})")
    print(f"Strategy: {cfg['search']['strategy']} -> {len(trials)} trial(s)")
    print(f"Output: {study_dir.resolve()}")
    if cfg.get("moco"):
        if cfg["moco"].get("reuse_only"):
            print("MoCo: reuse existing checkpoints only (will not train MoCo)")
        else:
            print("MoCo: auto-ensure enabled (train matching trunk before each yield trial)")
    else:
        print("MoCo: auto-ensure disabled (use base.pretrained if set)")

    succeeded_ids = set()
    if args.skip_completed and not args.rerun_all:
        for trial_id, rec in _latest_trial_records(study_dir / "trials.jsonl").items():
            if _trial_run_succeeded(rec):
                succeeded_ids.add(trial_id)

    resume_incomplete = args.resume_incomplete and not args.rerun_all
    if resume_incomplete:
        print("Resume: enabled (interrupted trials continue from latest checkpoint)")
    elif not args.rerun_all:
        print("Resume: disabled (interrupted trials restart from scratch)")

    training_script = Path(cfg["training_script"])
    repo_root = REPO_ROOT

    for i, sampled in enumerate(trials, start=1):
        trial_id = _trial_id(i)
        if trial_id in succeeded_ids:
            print(f"\n[{trial_id}] skip (already succeeded)")
            continue
        if sampled.get("skip"):
            reason = sampled.get("skip_reason") or "search.trials skip: true"
            print(f"\n[{trial_id}] skip ({reason})")
            continue

        merged = merge_trial_params(cfg["base"], sampled)
        merged.pop("skip", None)
        merged.pop("skip_reason", None)
        # Prefer explicit suffix. Else "{suffix_prefix}_{trial_id}".
        # Suffix is kept in config.json / model.modelname; the on-disk run
        # lives under the study trial folder (not results/Productivity_*).
        if not merged.get("suffix"):
            prefix = str(
                merged.pop("suffix_prefix", None)
                or f"tune_{_safe_run_tag(cfg['name'])}"
            )
            merged["suffix"] = f"{prefix}_{trial_id}"
        else:
            merged.pop("suffix_prefix", None)
        if "seed" in merged:
            merged["seed"] = int(merged["seed"]) + int(cfg.get("seed_offset", 0))

        trial_dir = study_dir / trial_id
        ensure_dir(trial_dir)
        # Yield artifacts (training/, figures/, predictions/) go in this folder.
        merged["run_dir"] = str(trial_dir.resolve())
        _save_json(trial_dir / "params.json", merged)

        year_loo_folds = list(merged.get("year_loo_folds") or [])
        if year_loo_folds:
            sampled_show = {
                k: v for k, v in sampled.items() if k != "year_loo_folds"
            }
            print(f"\n[{trial_id}] params: {json.dumps(sampled_show, sort_keys=True)}")
            n_run = sum(1 for fold in year_loo_folds if not fold.get("skip"))
            print(
                f"[{trial_id}] year LOO: {n_run} run(s) / {len(year_loo_folds)} fold(s) "
                f"→ mean {cfg['objective']}"
            )
            started = _utc_now_iso()
            print(f"[{trial_id}] started at {started}")
            fold_records: list[dict] = []
            for fold in year_loo_folds:
                year = int(fold["holdout_year"])
                label = f"{trial_id}/loo{year}"
                if fold.get("skip"):
                    reason = fold.get("skip_reason") or "skipped holdout year"
                    print(f"[{label}] skip ({reason})")
                    fold_records.append(
                        {
                            "holdout_year": year,
                            "status": "skipped",
                            "skip_reason": reason,
                        }
                    )
                    continue
                fold_dir = trial_dir / f"loo{year}"
                ensure_dir(fold_dir)
                fold_params = _year_loo_fold_params(
                    merged, fold, trial_id=trial_id, fold_dir=fold_dir
                )
                _save_json(fold_dir / "params.json", fold_params)
                print(f"[{label}] holdout_year={year} run_dir={fold_dir}")
                fold_outcome = _run_one_yield_job(
                    label=label,
                    merged=fold_params,
                    run_dir=fold_dir,
                    log_path=fold_dir / "train_stdout.log",
                    cfg=cfg,
                    args=args,
                    training_script=training_script,
                    repo_root=repo_root,
                    resume_incomplete=resume_incomplete,
                )
                fold_records.append({"holdout_year": year, **fold_outcome})
            if args.dry_run:
                outcome = {
                    "status": "dry_run",
                    "returncode": 0,
                    "folds": fold_records,
                    "run_dir": str(trial_dir.resolve()),
                }
            else:
                outcome = aggregate_year_loo_folds(
                    fold_records,
                    objective=cfg["objective"],
                    trial_dir=trial_dir,
                )
                _save_json(
                    trial_dir / "folds.json",
                    {
                        "folds": fold_records,
                        "aggregate": {
                            key: value
                            for key, value in outcome.items()
                            if key != "folds"
                        },
                    },
                )
            finished = _utc_now_iso()
            _print_trial_finished(trial_id, finished, started)
            if args.dry_run:
                status = "dry_run"
            else:
                status = "completed"
            if outcome.get("status") == "ok":
                mean_val = (outcome.get("metrics") or {}).get("r2")
                mean_test = (outcome.get("test_metrics") or {}).get("r2")
                extra = []
                if mean_val is not None:
                    extra.append(f"val_r2={mean_val:.6g}")
                if mean_test is not None:
                    extra.append(f"test_r2={mean_test:.6g}")
                skip_n = outcome.get("n_folds_skipped") or 0
                skip_msg = f", skipped={skip_n}" if skip_n else ""
                print(
                    f"[{trial_id}] OK  mean {cfg['objective']}="
                    f"{outcome['objective_value']:.6g}"
                    + (f" ({', '.join(extra)})" if extra else "")
                    + f" n_folds={outcome['n_folds']}{skip_msg}"
                    + format_run_epochs(outcome, mean=True)
                )
            elif not args.dry_run:
                print(
                    f"[{trial_id}] {outcome.get('status')} {outcome.get('error', '')}".rstrip()
                )
            _record_and_summarize(
                study_dir=study_dir,
                cfg=cfg,
                trial_id=trial_id,
                merged=merged,
                sampled=sampled,
                outcome=outcome,
                started=started,
                finished=finished,
                status=status,
            )
            continue

        print(f"\n[{trial_id}] params: {json.dumps(sampled, sort_keys=True)}")
        run_dir = predict_run_dir(merged, logdir=merged.get("logdir", "./results"))
        print(f"[{trial_id}] expected run_dir: {run_dir}")

        started = _utc_now_iso()
        print(f"[{trial_id}] started at {started}")

        resume_plan = None
        append_log = False
        train_params = merged
        if resume_incomplete:
            resume_plan = plan_trial_resume(
                run_dir,
                target_epochs=int(merged.get("epochs", 100)),
                early_stop_patience=int(merged.get("early_stop_patience", 0)) or None,
            )
            if resume_plan is not None:
                train_params = apply_resume_params(
                    merged, resume_plan["checkpoint"]
                )
                append_log = (trial_dir / "train_stdout.log").is_file()
                if not args.dry_run:
                    epoch = resume_plan.get("epoch")
                    es = resume_plan.get("early_stop") or {}
                    not_improved = es.get("not_improved_count")
                    best_ep = es.get("best_epoch")
                    msg = (
                        f"[{trial_id}] resume from epoch {epoch} "
                        f"via {resume_plan['checkpoint']}"
                    )
                    if not_improved is not None:
                        msg += (
                            f" (trainlog: best epoch {best_ep}, "
                            f"not_improved={not_improved})"
                        )
                    print(msg)
            elif _trial_appears_trained(run_dir) and not args.dry_run:
                print(f"[{trial_id}] training artifacts found; collecting metrics only")
                outcome = _finalize_trial_metrics(
                    run_dir,
                    objective=cfg["objective"],
                    objective_spec=cfg["objective_spec"],
                )
                finished = _utc_now_iso()
                _print_trial_finished(trial_id, finished, started)
                if outcome.get("status") == "ok":
                    print(
                        f"[{trial_id}] OK  {cfg['objective']}={outcome['objective_value']:.6g}"
                        f"{format_run_epochs(outcome)}"
                    )
                record = {
                    "trial_id": trial_id,
                    "status": "completed",
                    "params": merged,
                    "sampled": sampled,
                    "outcome": outcome,
                    "started_at_utc": started,
                    "finished_at_utc": finished,
                    "resumed_from_checkpoint": None,
                }
                _append_jsonl(study_dir / "trials.jsonl", record)
                df = _update_summary(study_dir)
                best = _pick_best(df, cfg["objective"], cfg["objective_spec"])
                if best:
                    _save_json(study_dir / "best_trial.json", best)
                continue
            elif (run_dir / "training" / "trainlog.csv").is_file() and not args.dry_run:
                print(
                    f"[{trial_id}] WARNING: trainlog exists but no checkpoint found; "
                    "starting fresh (earlier epoch weights cannot be restored)"
                )

        # Fresh yield start (not resuming a yield ckpt): ensure matching MoCo exists.
        moco_cfg = cfg.get("moco")
        if moco_cfg and resume_plan is None:
            expected_moco = predict_moco_checkpoint_path(
                merged, moco_cfg, repo_root=repo_root
            )
            if args.dry_run:
                print(f"[{trial_id}] would ensure MoCo: {expected_moco}")
                train_params = dict(train_params)
                train_params["pretrained"] = str(expected_moco)
                merged["pretrained"] = str(expected_moco)
            else:
                try:
                    moco_ckpt = ensure_moco_checkpoint(
                        merged,
                        moco_cfg,
                        repo_root=repo_root,
                        dry_run=False,
                        capture_log=trial_dir / "moco_stdout.log",
                    )
                except RuntimeError as exc:
                    finished = _utc_now_iso()
                    print(f"[{trial_id}] FAILED MoCo ensure: {exc}")
                    _print_trial_finished(trial_id, finished, started)
                    record = {
                        "trial_id": trial_id,
                        "status": "completed",
                        "params": merged,
                        "sampled": sampled,
                        "outcome": {
                            "status": "moco_failed",
                            "error": str(exc),
                            "returncode": 1,
                            "run_dir": str(run_dir.resolve()),
                        },
                        "started_at_utc": started,
                        "finished_at_utc": finished,
                        "resumed_from_checkpoint": None,
                    }
                    _append_jsonl(study_dir / "trials.jsonl", record)
                    df = _update_summary(study_dir)
                    best = _pick_best(df, cfg["objective"], cfg["objective_spec"])
                    if best:
                        _save_json(study_dir / "best_trial.json", best)
                    continue
                print(f"[{trial_id}] MoCo pretrained: {moco_ckpt}")
                train_params = dict(train_params)
                train_params["pretrained"] = str(moco_ckpt)
                merged["pretrained"] = str(moco_ckpt)
                _save_json(trial_dir / "params.json", merged)

        if args.dry_run:
            if resume_plan is not None:
                epoch = resume_plan.get("epoch")
                print(
                    f"[{trial_id}] would resume from epoch {epoch} "
                    f"via {resume_plan['checkpoint']}"
                )
            outcome = run_trial_subprocess(
                train_params,
                training_script=training_script,
                repo_root=repo_root,
                dry_run=True,
            )
            print(f"[{trial_id}] argv: {' '.join(outcome['argv'])}")
            finished = _utc_now_iso()
            _print_trial_finished(trial_id, finished, started)
            record = {
                "trial_id": trial_id,
                "status": "dry_run",
                "params": merged,
                "sampled": sampled,
                "outcome": outcome,
                "started_at_utc": started,
                "finished_at_utc": finished,
            }
            _append_jsonl(study_dir / "trials.jsonl", record)
            continue

        proc_outcome = run_trial_subprocess(
            train_params,
            training_script=training_script,
            repo_root=repo_root,
            capture_log=trial_dir / "train_stdout.log",
            append_log=append_log,
        )

        if proc_outcome["returncode"] != 0:
            outcome = {
                **proc_outcome,
                "status": "train_failed",
                "error": f"training exited with code {proc_outcome['returncode']}",
            }
            print(f"[{trial_id}] FAILED (returncode={proc_outcome['returncode']})")
        else:
            metrics = best_epoch_metrics(
                proc_outcome["run_dir"],
                objective=cfg["objective"],
                objective_spec=cfg["objective_spec"],
            )
            outcome = {**proc_outcome, **metrics}
            if metrics.get("status") == "ok":
                print(
                    f"[{trial_id}] OK  {cfg['objective']}={metrics['objective_value']:.6g}"
                    f"{format_run_epochs(outcome)}"
                )
            else:
                print(f"[{trial_id}] finished but metrics status={metrics.get('status')}")

        finished = _utc_now_iso()
        _print_trial_finished(trial_id, finished, started)
        record = {
            "trial_id": trial_id,
            "status": "completed",
            "params": merged,
            "sampled": sampled,
            "outcome": outcome,
            "started_at_utc": started,
            "finished_at_utc": finished,
            "resumed_from_checkpoint": (
                str(resume_plan["checkpoint"]) if resume_plan else None
            ),
        }
        _append_jsonl(study_dir / "trials.jsonl", record)

        df = _update_summary(study_dir)
        best = _pick_best(df, cfg["objective"], cfg["objective_spec"])
        if best:
            _save_json(study_dir / "best_trial.json", best)

    df = _update_summary(study_dir)
    best = _pick_best(df, cfg["objective"], cfg["objective_spec"])
    if best:
        _save_json(study_dir / "best_trial.json", best)
        print(f"\nBest so far: {best.get('trial_id')} -> {cfg['objective']}={best.get('objective_value')}")
    else:
        print("\nNo successful trials yet.")
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    cfg = load_study_config(args.config)
    study_dir = study_output_dir(cfg)
    df = _update_summary(study_dir)
    if df.empty:
        print(f"No trials found in {study_dir / 'trials.jsonl'}")
        return 1
    print(f"Wrote {study_dir / 'summary.csv'} ({len(df)} rows)")
    best = _pick_best(df, cfg["objective"], cfg["objective_spec"])
    if best:
        _save_json(study_dir / "best_trial.json", best)
        print(f"Best: {best.get('trial_id')} -> {cfg['objective']}={best.get('objective_value')}")
    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    cfg = load_study_config(args.config)
    study_dir = study_output_dir(cfg)
    summary_path = study_dir / "summary.csv"
    if not summary_path.is_file():
        df = _update_summary(study_dir)
    else:
        df = pd.read_csv(summary_path)

    if df.empty:
        print("No trials to show.")
        return 1

    valid = df[df["status"] == "ok"].copy()
    if valid.empty:
        print("No successful trials.")
        return 1

    col = "objective_value"
    mode = cfg["objective_spec"]["mode"]
    valid = valid.sort_values(col, ascending=(mode == "min"))
    top = valid.head(int(args.top))

    show_cols = [
        c
        for c in (
            "trial_id",
            "started_at_utc",
            "finished_at_utc",
            "duration_seconds",
            "objective_value",
            "n_epochs",
            "best_epoch",
            "val_r2",
            "val_rmse",
            "val_valloss",
            "test_r2",
            "n_folds",
            "param_feature_layout",
            "param_year_loo",
            "param_learning_rate",
            "param_weight_decay",
            "param_batchsize",
            "param_model_n_layers",
            "param_model_dropout",
            "run_dir",
        )
        if c in top.columns
    ]
    print(top[show_cols].to_string(index=False))
    return 0


def cmd_show_config(args: argparse.Namespace) -> int:
    cfg = load_study_config(args.config)
    trials = generate_trials({**cfg["search"], "base": cfg["base"]})
    print(yaml.safe_dump(
        {
            "name": cfg["name"],
            "objective": cfg["objective"],
            "n_trials": len(trials),
            "strategy": cfg["search"]["strategy"],
            "year_loo": cfg["search"].get("year_loo"),
            "output_dir": str(study_output_dir(cfg)),
        },
        sort_keys=False,
    ))
    if args.list_trials:
        for i, t in enumerate(trials, start=1):
            folds = t.get("year_loo_folds")
            if folds:
                years = []
                skipped = []
                for fold in folds:
                    year = int(fold["holdout_year"])
                    if fold.get("skip"):
                        skipped.append(str(year))
                    else:
                        years.append(str(year))
                extra = f" folds={','.join(years)}"
                if skipped:
                    extra += f" skip={','.join(skipped)}"
                layout = t.get("feature_layout", "")
                print(f"{_trial_id(i)}: {layout}{extra}")
            else:
                print(f"{_trial_id(i)}: {json.dumps(t, sort_keys=True)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Structured hyperparameter tuning")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Execute tuning study")
    p_run.add_argument("config", type=Path, help="Path to study YAML")
    p_run.add_argument("--dry-run", action="store_true", help="Print commands without training")
    p_run.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip trials whose latest run succeeded (outcome status ok)",
    )
    p_run.add_argument(
        "--resume-incomplete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Continue interrupted trials from the latest checkpoint under "
            "results/tuning/<study>/<trial_id>/training/ (default: enabled). "
            "Use --no-resume-incomplete to restart them from epoch 1."
        ),
    )
    p_run.add_argument(
        "--rerun-all",
        action="store_true",
        help="Run every trial even if a previous run succeeded (e.g. after recipe change)",
    )
    p_run.add_argument("--max-trials", type=int, default=None, help="Limit number of trials to run")
    p_run.set_defaults(func=cmd_run)

    p_sum = sub.add_parser("summarize", help="Rebuild summary.csv from trials.jsonl")
    p_sum.add_argument("config", type=Path)
    p_sum.set_defaults(func=cmd_summarize)

    p_lb = sub.add_parser("leaderboard", help="Print top trials")
    p_lb.add_argument("config", type=Path)
    p_lb.add_argument("--top", type=int, default=5)
    p_lb.set_defaults(func=cmd_leaderboard)

    p_show = sub.add_parser("show", help="Preview study settings and trial grid")
    p_show.add_argument("config", type=Path)
    p_show.add_argument("--list-trials", action="store_true")
    p_show.set_defaults(func=cmd_show_config)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
