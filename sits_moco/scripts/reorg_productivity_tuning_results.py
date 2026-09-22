"""Merge orphan Productivity_*_tune_* trial dirs into results/tuning/<study>/trial_NNN/.

Promotes a single feature-layout child (e.g. spectral_xavier/training) up to the
trial root so layout matches modern --run-dir trials. Leaves MoCo trunks untouched.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
TUNING = RESULTS / "tuning"

BUCKETS = (
    "productivity_first_real_tuning",
    "productivity_dropout_input",
    "alternative_coverage_tuning",
    "alternative_targets_tuning",
    "moco_first_tuning",
    "moco_second_tuning",
)

FLAT_RE = re.compile(
    r"^(?:Productivity|ProductivityDev|DualProdTotalAdj|TotalAdj)_.*_tune_(.+)_trial_(\d+)$"
)
FIRST_REAL_RE = re.compile(r"^Productivity_.*_tune_trial_(\d+)$")

DOT_FIXES = (
    ("coverage_0_8_1_2", "coverage_0.8_1.2"),
    ("coverage_0_3_1_5", "coverage_0.3_1.5"),
    ("midkeep0_52", "midkeep0.52"),
)

FEATURE_LAYOUT_NAMES = {
    "spectral",
    "spectral_xavier",
    "spectral_xavier_climate",
    "spectral_xavier_climate_soil",
    "spectral_xavier_full",
}


def fix_study_raw(raw: str) -> str:
    s = raw
    for a, b in DOT_FIXES:
        s = s.replace(a, b)
    return s


def collect_sources() -> list[tuple[Path, str, str]]:
    """Return (src_dir, study_name, trial_id)."""
    out: list[tuple[Path, str, str]] = []
    roots = [RESULTS] + [RESULTS / b for b in BUCKETS if (RESULTS / b).is_dir()]
    for root in roots:
        for child in root.iterdir():
            if not child.is_dir():
                continue
            m = FLAT_RE.match(child.name)
            if m:
                study = fix_study_raw(m.group(1))
                # Preserve wave buckets that reused the same study suffix
                if root.name in {"moco_first_tuning", "moco_second_tuning"}:
                    study = f"{study}__{root.name}"
                trial = f"trial_{int(m.group(2)):03d}"
                out.append((child, study, trial))
                continue
            if root.name == "productivity_first_real_tuning":
                m2 = FIRST_REAL_RE.match(child.name)
                if m2:
                    trial = f"trial_{int(m2.group(1)):03d}"
                    out.append((child, "productivity_first_real_tuning", trial))
    return out


def move_nested_study_trees() -> list[dict]:
    """Move bucket/<study_name>/ (modern tuner layout) into results/tuning/<study_name>/."""
    report: list[dict] = []
    for bucket in BUCKETS:
        bp = RESULTS / bucket
        if not bp.is_dir():
            continue
        for child in list(bp.iterdir()):
            if not child.is_dir():
                continue
            # Already handled as flat Productivity_* dirs
            if FLAT_RE.match(child.name) or FIRST_REAL_RE.match(child.name):
                continue
            # Nested study tree: has trial_* children and/or summary.csv
            trials = list(child.glob("trial_*"))
            if not trials and not (child / "summary.csv").is_file():
                continue
            study = fix_study_raw(child.name)
            dest = TUNING / study
            entry = {
                "src": str(child.relative_to(RESULTS)),
                "dest": str(dest.relative_to(RESULTS)),
                "kind": "nested_study_tree",
            }
            if dest.exists():
                # Merge trial folders / meta files
                for item in child.iterdir():
                    target = dest / item.name
                    if target.exists():
                        if item.is_dir() and item.name.startswith("trial_"):
                            # merge trial contents
                            for nested in item.rglob("*"):
                                if nested.is_file():
                                    rel = nested.relative_to(item)
                                    t = target / rel
                                    t.parent.mkdir(parents=True, exist_ok=True)
                                    if not t.exists():
                                        shutil.move(str(nested), str(t))
                        elif not target.exists():
                            shutil.move(str(item), str(target))
                    else:
                        shutil.move(str(item), str(target))
                shutil.rmtree(child, ignore_errors=True)
                entry["merged_into_existing"] = True
            else:
                shutil.move(str(child), str(dest))
                entry["moved"] = True
            report.append(entry)
            print(f"STUDY  {entry['src']} -> {entry['dest']}")
    return report


def promote_layout_contents(src: Path, dest: Path) -> list[str]:
    """
    Copy/move training artifacts into dest.

    If src has a single feature-layout subdirectory containing training/,
    promote that subdirectory's children to dest. Otherwise move src children.
    """
    actions: list[str] = []
    children = [p for p in src.iterdir()]
    layout_kids = [
        p
        for p in children
        if p.is_dir() and (p.name in FEATURE_LAYOUT_NAMES or (p / "training").is_dir())
    ]
    # Prefer promoting when the only meaningful dirs are feature layouts
    non_layout = [
        p
        for p in children
        if p.name not in FEATURE_LAYOUT_NAMES
        and p.name not in {"figures", "predictions", "training"}
    ]
    if len(layout_kids) == 1 and not any(
        p.is_dir() and p.name in {"training", "figures", "predictions"} for p in children
    ):
        payload_root = layout_kids[0]
    elif (src / "training").is_dir():
        payload_root = src
    elif len(layout_kids) >= 1:
        # Multiple layouts: keep them nested under dest
        for kid in children:
            target = dest / kid.name
            if target.exists():
                if kid.is_dir() and target.is_dir():
                    for nested in kid.rglob("*"):
                        if nested.is_file():
                            rel = nested.relative_to(kid)
                            t = target / rel
                            t.parent.mkdir(parents=True, exist_ok=True)
                            if not t.exists():
                                shutil.move(str(nested), str(t))
                                actions.append(f"merge {nested.name} -> {t}")
                else:
                    actions.append(f"skip exists {target}")
            else:
                shutil.move(str(kid), str(target))
                actions.append(f"move {kid.name}")
        return actions
    else:
        payload_root = src

    for item in list(payload_root.iterdir()):
        target = dest / item.name
        if target.exists():
            if item.is_dir() and target.is_dir():
                for nested in item.rglob("*"):
                    if nested.is_file():
                        rel = nested.relative_to(item)
                        t = target / rel
                        t.parent.mkdir(parents=True, exist_ok=True)
                        if not t.exists():
                            shutil.move(str(nested), str(t))
                            actions.append(f"merge {rel}")
                        else:
                            actions.append(f"keep existing {rel}")
            else:
                actions.append(f"keep existing {item.name}")
        else:
            shutil.move(str(item), str(target))
            actions.append(f"move {item.name}")
    return actions


def rmdir_empty(path: Path) -> None:
    if not path.exists():
        return
    for p in sorted(path.rglob("*"), reverse=True):
        if p.is_dir():
            try:
                p.rmdir()
            except OSError:
                pass
        # leftover empty files? leave them
    try:
        path.rmdir()
    except OSError:
        # still has files — move leftovers into dest/_orphan_leftover
        pass


def main() -> int:
    dry = "--dry-run" in sys.argv
    sources = collect_sources()
    print(f"Found {len(sources)} orphan productivity trial dirs")
    report = []
    if dry:
        for src, study, trial in sorted(sources, key=lambda x: (x[1], x[2])):
            dest = TUNING / study / trial
            print(f"OK         {src.relative_to(RESULTS)} -> {dest.relative_to(RESULTS)}")
        print("(dry-run skips nested study trees and merges)")
        return 0
    for src, study, trial in sorted(sources, key=lambda x: (x[1], x[2])):
        dest = TUNING / study / trial
        incoming_model = next(src.rglob("model_best.pth"), None)
        existing_model = next(dest.rglob("model_best.pth"), None) if dest.exists() else None
        entry = {
            "src": str(src.relative_to(RESULTS)),
            "dest": str(dest.relative_to(RESULTS)),
            "incoming_model": incoming_model is not None,
            "existing_model": existing_model is not None,
        }
        dest.mkdir(parents=True, exist_ok=True)
        if existing_model and incoming_model:
            alt = dest / "from_flat_logdir"
            alt.mkdir(exist_ok=True)
            actions = promote_layout_contents(src, alt)
            entry["collision"] = True
            entry["actions"] = actions
        else:
            actions = promote_layout_contents(src, dest)
            entry["actions"] = actions
        leftovers = [p for p in src.rglob("*") if p.is_file()]
        if leftovers:
            dump = dest / "_flat_leftovers"
            dump.mkdir(exist_ok=True)
            for f in leftovers:
                rel = f.relative_to(src)
                t = dump / rel
                t.parent.mkdir(parents=True, exist_ok=True)
                if not t.exists():
                    shutil.move(str(f), str(t))
        rmdir_empty(src)
        if src.exists():
            shutil.rmtree(src, ignore_errors=True)
        print(f"MERGED {entry['src']} -> {entry['dest']}")
        report.append(entry)

    report.extend(move_nested_study_trees())

    for b in BUCKETS:
        bp = RESULTS / b
        if bp.is_dir() and not any(bp.iterdir()):
            bp.rmdir()
            print(f"removed empty bucket {b}")

    out = RESULTS / "archive" / "productivity_trial_reorg_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
