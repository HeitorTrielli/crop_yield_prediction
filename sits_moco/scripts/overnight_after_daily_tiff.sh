#!/usr/bin/env bash
# Overnight pipeline:
#   1) Wait for build_parana_daily_tiff to finish (or verify outputs exist)
#   2) Build pixel MapBiomas Solo sidecars for each season
#   3) Run productivity_soil_sidecar tuning study
#
# Usage (from repo root, in WSL):
#   nohup bash scripts/overnight_after_daily_tiff.sh > files/overnight_pipeline.log 2>&1 &
#   # or:
#   bash scripts/overnight_after_daily_tiff.sh --detach
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${ROOT}/.venv/bin/python"
LOG_DIR="${ROOT}/files"
LOG="${LOG_DIR}/overnight_pipeline.log"
PROGRESS_JSON="${LOG_DIR}/build_parana_daily_tiff_progress.json"
STATUS_JSON="${LOG_DIR}/overnight_pipeline_status.json"

NPY_ROOT="${NPY_ROOT:-$HOME/sits_moco_data/npy}"
TIFF_ROOT="${TIFF_ROOT:-$HOME/sits_moco_data/daily_tiff}"
MOSAIC_ROOT="${MOSAIC_ROOT:-$HOME/sits_moco_data/state_tiff}"
SOLO_DIR="${SOLO_DIR:-files/mapbiomas_solo}"
STUDY="${STUDY:-tuning/studies/productivity_soil_sidecar.yaml}"
SEASONS=(2019-2020 2020-2021 2021-2022 2022-2023 2023-2024)

# Minimum municipality folders expected per season (PR has ~399; allow partial).
MIN_MUNI_DIRS="${MIN_MUNI_DIRS:-350}"
# Poll interval while waiting for TIFF build (seconds).
POLL_SEC="${POLL_SEC:-120}"
# Max wait for TIFF build (default 12h).
MAX_WAIT_SEC="${MAX_WAIT_SEC:-43200}"
SOIL_WORKERS="${SOIL_WORKERS:-6}"
SKIP_SOIL="${SKIP_SOIL:-0}"
SKIP_STUDY="${SKIP_STUDY:-0}"
FORCE_SOIL="${FORCE_SOIL:-1}"

mkdir -p "$LOG_DIR"

if [[ "${1:-}" == "--detach" ]]; then
  shift
  nohup bash "$0" "$@" >>"$LOG" 2>&1 &
  echo "detached pid=$! log=$LOG"
  exit 0
fi

# If not already redirected, tee to log.
if [[ -t 1 ]]; then
  exec > >(tee -a "$LOG") 2>&1
fi

ts() { date -Iseconds; }
log() { echo "[$(ts)] $*"; }

write_status() {
  local phase="$1"
  local detail="${2:-}"
  "$PY" - "$STATUS_JSON" "$phase" "$detail" <<'PY'
import json, sys
from datetime import datetime, timezone
path, phase, detail = sys.argv[1], sys.argv[2], sys.argv[3]
payload = {
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "phase": phase,
    "detail": detail,
}
path_p = __import__("pathlib").Path(path)
if path_p.is_file():
    try:
        old = json.loads(path_p.read_text(encoding="utf-8"))
        if isinstance(old, dict):
            payload = {**old, **payload}
    except Exception:
        pass
path_p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

season_ready() {
  local season="$1"
  local dir="${TIFF_ROOT}/${season}"
  local n=0
  if [[ -d "$dir" ]]; then
    n="$(find "$dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ' || true)"
  fi
  [[ "${n:-0}" -ge "$MIN_MUNI_DIRS" ]]
}

all_seasons_ready() {
  local s
  for s in "${SEASONS[@]}"; do
    if ! season_ready "$s"; then
      return 1
    fi
  done
  return 0
}

build_running() {
  pgrep -f "scripts/build_parana_daily_tiff.py" >/dev/null 2>&1
}

summarize_tiff() {
  local s n_muni n_tiff dir
  for s in "${SEASONS[@]}"; do
    dir="${TIFF_ROOT}/${s}"
    n_muni=0
    n_tiff=0
    if [[ -d "$dir" ]]; then
      n_muni="$(find "$dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ' || true)"
      # Count only one level of files under muni dirs can be slow; sample via maxdepth 3.
      n_tiff="$(find "$dir" -type f \( -name '*.tif' -o -name '*.tiff' \) 2>/dev/null | wc -l | tr -d ' ' || true)"
    fi
    log "  daily_tiff/${s}: muni_dirs=${n_muni:-0} tiffs=${n_tiff:-0}"
  done
}

wait_for_daily_tiff() {
  write_status "wait_daily_tiff" "polling"
  local waited=0
  log "Waiting for daily_tiff (min ${MIN_MUNI_DIRS} muni dirs/season)."
  log "TIFF_ROOT=$TIFF_ROOT"
  summarize_tiff

  if all_seasons_ready; then
    log "All seasons already have enough daily_tiff folders."
    return 0
  fi

  while true; do
    if all_seasons_ready; then
      log "daily_tiff ready for all seasons."
      summarize_tiff
      return 0
    fi

    if ! build_running; then
      # Build dead but incomplete — wait a bit in case user restarts, then fail soft
      # if still incomplete after grace.
      log "WARNING: build_parana_daily_tiff is not running and seasons incomplete."
      summarize_tiff
      if [[ -f "$PROGRESS_JSON" ]]; then
        log "progress_json:"
        cat "$PROGRESS_JSON" || true
      fi
      # Continue with whatever seasons are ready if at least one is complete.
      local any=0
      local s
      for s in "${SEASONS[@]}"; do
        if season_ready "$s"; then
          any=1
          break
        fi
      done
      if [[ "$any" -eq 1 ]]; then
        log "Proceeding with seasons that are ready (partial OK for overnight)."
        return 0
      fi
      log "ERROR: no season ready and build not running. Abort."
      write_status "failed" "daily_tiff incomplete and build not running"
      exit 1
    fi

    if [[ "$waited" -ge "$MAX_WAIT_SEC" ]]; then
      log "ERROR: timed out waiting for daily_tiff after ${MAX_WAIT_SEC}s"
      write_status "failed" "timeout waiting for daily_tiff"
      exit 1
    fi

    log "still waiting… build_running=yes waited=${waited}s (poll ${POLL_SEC}s)"
    if [[ -f "$PROGRESS_JSON" ]]; then
      "$PY" -c "import json; p=json.load(open('$PROGRESS_JSON')); print('  phase=',p.get('phase'),'season=',p.get('season'),'clip=',p.get('clip'),'merge=',p.get('merge'))" 2>/dev/null || true
    fi
    sleep "$POLL_SEC"
    waited=$((waited + POLL_SEC))
  done
}

ready_seasons() {
  local s
  for s in "${SEASONS[@]}"; do
    if season_ready "$s"; then
      echo "$s"
    fi
  done
}

build_soil_sidecars() {
  write_status "soil_sidecars" "starting"
  local seasons=()
  mapfile -t seasons < <(ready_seasons)
  if [[ "${#seasons[@]}" -eq 0 ]]; then
    log "ERROR: no ready seasons for soil sidecars"
    exit 1
  fi
  log "Building pixel soil sidecars for: ${seasons[*]}"
  log "NPY_ROOT=$NPY_ROOT TIFF_ROOT=$TIFF_ROOT workers=$SOIL_WORKERS force=$FORCE_SOIL"

  local yr force_flag=()
  if [[ "$FORCE_SOIL" == "1" ]]; then
    force_flag=(--force)
  fi

  for yr in "${seasons[@]}"; do
    write_status "soil_sidecars" "$yr"
    log "==== soil pixel ${yr} ===="
    "$PY" data_download/build_mapbiomas_solo_sidecars.py \
      --mode pixel \
      --npy-root "$NPY_ROOT" \
      --tiff-root "$TIFF_ROOT" \
      --solo-dir "$SOLO_DIR" \
      --year-range "$yr" \
      -j "$SOIL_WORKERS" \
      "${force_flag[@]}"
  done
  log "Soil sidecars done."
  write_status "soil_sidecars" "done"
}

run_study() {
  write_status "study" "starting"
  log "==== run study $STUDY ===="
  # Prefer datapath env used by training.
  export SITS_MOCO_DATAPATH="${SITS_MOCO_DATAPATH:-$NPY_ROOT}"
  log "SITS_MOCO_DATAPATH=$SITS_MOCO_DATAPATH"
  "$PY" run_tuning_study.py run "$STUDY" --skip-completed
  log "Study finished."
  write_status "study" "done"
}

main() {
  log "==== overnight pipeline start ===="
  log "ROOT=$ROOT"
  write_status "starting" ""

  if [[ ! -x "$PY" ]]; then
    log "ERROR: missing venv python at $PY"
    exit 1
  fi

  wait_for_daily_tiff

  if [[ "$SKIP_SOIL" != "1" ]]; then
    build_soil_sidecars
  else
    log "SKIP_SOIL=1 — skipping sidecar rebuild"
  fi

  if [[ "$SKIP_STUDY" != "1" ]]; then
    run_study
  else
    log "SKIP_STUDY=1 — skipping tuning study"
  fi

  write_status "done" "all steps completed"
  log "==== overnight pipeline DONE ===="
}

main "$@"
