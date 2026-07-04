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
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tuning.collect import best_epoch_metrics, trial_row
from tuning.config import load_study_config, merge_trial_params
from tuning.resume import plan_trial_resume
from tuning.runner import apply_resume_params, predict_run_dir, run_trial_subprocess
from tuning.search import generate_trials


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    return {
        "run_dir": str(run_dir.resolve()),
        "returncode": 0,
        "finalized_without_training": True,
        **metrics,
    }


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
    study_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot study config
    shutil.copy2(args.config, study_dir / "study_config.yaml")

    trials = generate_trials({**cfg["search"], "base": cfg["base"]})
    if args.max_trials is not None:
        trials = trials[: int(args.max_trials)]

    print(f"Study: {cfg['name']}")
    print(f"Objective: {cfg['objective']} ({cfg['objective_spec']['mode']} {cfg['objective_spec']['column']})")
    print(f"Strategy: {cfg['search']['strategy']} -> {len(trials)} trial(s)")
    print(f"Output: {study_dir.resolve()}")

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

        merged = merge_trial_params(cfg["base"], sampled)
        merged["suffix"] = merged.get("suffix") or f"tune_{trial_id}"
        if "seed" in merged:
            merged["seed"] = int(merged["seed"]) + int(cfg.get("seed_offset", 0))

        trial_dir = study_dir / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        _save_json(trial_dir / "params.json", merged)

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
                        f"[{trial_id}] OK  {cfg['objective']}={outcome['objective_value']:.6g} "
                        f"(epoch {outcome['best_epoch']})"
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
                    f"[{trial_id}] OK  {cfg['objective']}={metrics['objective_value']:.6g} "
                    f"(epoch {metrics['best_epoch']})"
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
            "best_epoch",
            "val_r2",
            "val_rmse",
            "val_valloss",
            "test_r2",
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
            "output_dir": str(study_output_dir(cfg)),
        },
        sort_keys=False,
    ))
    if args.list_trials:
        for i, t in enumerate(trials, start=1):
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
            "Continue interrupted trials from the latest checkpoint under the "
            "expected run_dir (default: enabled). Use --no-resume-incomplete to "
            "restart them from epoch 1."
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
