#!/usr/bin/env bash
# Build São Paulo spectral cubes and soy-masked soil/climate sidecars, then
# prepare the PR+RS+SP 0.5-1.5 coverage year-LOO study.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-/home/emap/.venvs/sits_moco/bin/python}"
NPY="${NPY_ROOT:-$HOME/sits_moco_data/npy_muni_mean_soy}"
SOLO="${SOLO_DIR:-$HOME/sits_moco_data/mapbiomas_solo_sp}"
SHAPES="files/shapefiles"
CLIMATE="files/climate"
SEASONS=(2019-2020 2020-2021 2021-2022 2022-2023 2023-2024)
LOG="files/build_sp_sidecars_and_study.log"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:$ROOT/data_download:${PYTHONPATH:-}"
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -Is) SP sidecars and PR+RS+SP study ==="

echo "=== 1/5 Build SP spectral cubes from completed GEE zonal CSVs ==="
"$PY" - <<'PY'
from pathlib import Path
import csv

root = Path("files/zonal_mean_gee")
out = Path("files/zonal_mean_sp_keep.csv")
rows = [
    {
        "municipality_code": path.parent.name,
        "year": int(path.parent.parent.name.split("-")[1]),
    }
    for path in root.glob("*/35*/35*_zonal.csv")
]
rows.sort(key=lambda row: (row["year"], row["municipality_code"]))
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["municipality_code", "year"])
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {out}: {len(rows)} municipality-years")
PY
"$PY" preprocessing/zonal_csv_to_muni_npy.py \
  --zonal-root files/zonal_mean_gee \
  --keep-list files/zonal_mean_sp_keep.csv \
  --output-dir "$NPY" \
  --overwrite

for required in \
  "$SOLO/sp_texture_0_30cm.tif" \
  "$CLIMATE/pr_SP_daily_2019_2024.nc" \
  "$CLIMATE/ETo_SP_daily_2019_2024.nc" \
  "$CLIMATE/Rs_SP_daily_2019_2024.nc" \
  "$CLIMATE/Tmax_SP_daily_2019_2024.nc" \
  "$CLIMATE/Tmin_SP_daily_2019_2024.nc"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing prerequisite: $required" >&2
    exit 2
  fi
done

echo "=== 2/5 Build soy-masked SP soil sidecars ==="
for season in "${SEASONS[@]}"; do
  codes="$("$PY" - "$NPY" "$season" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).expanduser() / sys.argv[2]
codes = sorted(
    path.name
    for path in root.glob("35*")
    if (path / f"{path.name}.npy").is_file()
)
print(",".join(codes))
PY
)"
  [[ -z "$codes" ]] && continue
  "$PY" data_download/build_mapbiomas_solo_sidecars.py \
    --mode muni_mean \
    --npy-root "$NPY" \
    --shapefile-dir "$SHAPES" \
    --solo-dir "$SOLO" \
    --region-tag sp \
    --year-range "$season" \
    --codes "$codes" \
    -j "${SOIL_WORKERS:-4}"
done

echo "=== 3/5 Build soy-weighted SP daily climate sidecars ==="
for season in "${SEASONS[@]}"; do
  codes="$("$PY" - "$NPY" "$season" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).expanduser() / sys.argv[2]
codes = sorted(
    path.name
    for path in root.glob("35*")
    if (path / f"{path.name}.npy").is_file()
)
print(",".join(codes))
PY
)"
  [[ -z "$codes" ]] && continue
  "$PY" data_download/build_muni_daily_climate_npy.py \
    --npy-root "$NPY" \
    --year-range "$season" \
    --shapefile-dir "$SHAPES" \
    --codes "$codes" \
    --xavier-pr-nc "$CLIMATE/pr_SP_daily_2019_2024.nc" \
    --xavier-eto-nc "$CLIMATE/ETo_SP_daily_2019_2024.nc" \
    --xavier-rs-nc "$CLIMATE/Rs_SP_daily_2019_2024.nc" \
    --xavier-tmax-nc "$CLIMATE/Tmax_SP_daily_2019_2024.nc" \
    --xavier-tmin-nc "$CLIMATE/Tmin_SP_daily_2019_2024.nc"
done

echo "=== 4/5 Build PR+RS+SP coverage dataset ==="
"$PY" - <<'PY'
from pathlib import Path
import os
import shutil

root = Path.cwd()
out = root / "files/zonal_mean_pr_rs_sp"
sources = (
    (root / "files/zonal_mean", "41"),
    (root / "files/zonal_mean_gee", "43"),
    (root / "files/zonal_mean_gee", "35"),
)
counts = {}
for src_root, prefix in sources:
    count = 0
    for season_dir in sorted(src_root.glob("*/")):
        if "-" not in season_dir.name:
            continue
        for muni in season_dir.glob(f"{prefix}*/"):
            code = muni.name
            for kind in ("area", "zonal"):
                src = muni / f"{code}_{kind}.csv"
                if not src.is_file():
                    continue
                dst = out / season_dir.name / code / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    dst.unlink()
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
                if kind == "area":
                    count += 1
    counts[prefix] = count
print("Linked area CSVs:", counts)
PY

"$PY" data_download/build_mapbiomas_coverage_yield_csv.py \
  --zonal-root files/zonal_mean_pr_rs_sp \
  --pam-csv files/pam_soy_pr_rs_sp_2019-2025.csv \
  --existing-yield-csv files/pam_soy_pr_rs_mapbiomas_coverage_0.5_1.5.csv \
  --npy-root "$NPY" \
  --harvest-years 2020,2021,2022,2023,2024 \
  --holdout-years 2021 \
  --min-ratio 0.5 --max-ratio 1.5 \
  --output files/pam_soy_pr_rs_sp_mapbiomas_coverage_0.5_1.5.csv \
  --report files/pam_soy_pr_rs_sp_mapbiomas_coverage_0.5_1.5_report.csv \
  --keep-list files/pam_soy_pr_rs_sp_mapbiomas_coverage_0.5_1.5_keep.txt

echo "=== 5/5 Create PR+RS+SP top-100 year-LOO study ==="
"$PY" - <<'PY'
from pathlib import Path

src = Path(
    "tuning/studies/secondary/"
    "productivity_muni_mean_dual_stnet_wide_soy_pr_rs_0.5_1.5_top100_year_loo.yaml"
)
dst = Path(
    "tuning/studies/secondary/"
    "productivity_muni_mean_dual_stnet_wide_soy_pr_rs_sp_0.5_1.5_top100_year_loo.yaml"
)
text = src.read_text(encoding="utf-8")
text = text.replace("PR+RS", "PR+RS+SP")
text = text.replace(
    "productivity_muni_mean_dual_stnet_wide_soy_pr_rs_0.5_1.5_top100_year_loo",
    "productivity_muni_mean_dual_stnet_wide_soy_pr_rs_sp_0.5_1.5_top100_year_loo",
)
text = text.replace(
    "pam_soy_pr_rs_mapbiomas_coverage_0.5_1.5.csv",
    "pam_soy_pr_rs_sp_mapbiomas_coverage_0.5_1.5.csv",
)
text = text.replace(
    "train_input_scaler_dual_muni_mean_soy_pr_rs_0.5_1.5_loo",
    "train_input_scaler_dual_muni_mean_soy_pr_rs_sp_0.5_1.5_loo",
)
text = text.replace(
    "dual_stnet_wide_soy_pr_rs_0.5_1.5_top100_yloo",
    "dual_stnet_wide_soy_pr_rs_sp_0.5_1.5_top100_yloo",
)
dst.write_text(text, encoding="utf-8")
print(f"Wrote {dst}")
PY

"$PY" - "$NPY" <<'PY'
from collections import Counter
from pathlib import Path
import csv
import sys

npy = Path(sys.argv[1]).expanduser()
counts = Counter()
missing = Counter()
for season in ("2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024"):
    for folder in (npy / season).glob("35*"):
        code = folder.name
        if not (folder / f"{code}.npy").is_file():
            continue
        counts[season] += 1
        if not (folder / f"{code}_soil.npy").is_file():
            missing[(season, "soil")] += 1
        if not (folder / f"{code}_climate_daily.npy").is_file():
            missing[(season, "climate")] += 1
rows = list(csv.DictReader(open(
    "files/pam_soy_pr_rs_sp_mapbiomas_coverage_0.5_1.5.csv",
    encoding="utf-8",
)))
by_uf = Counter(str(row["municipality_code"]).zfill(7)[:2] for row in rows)
print("SP cubes by season:", dict(counts))
print("Missing SP sidecars:", dict(missing))
print("Study rows by UF:", dict(by_uf), "total=", len(rows))
if missing:
    raise SystemExit("Verification failed: missing SP sidecars")
PY

echo "=== $(date -Is) DONE ==="
echo "Study: tuning/studies/secondary/productivity_muni_mean_dual_stnet_wide_soy_pr_rs_sp_0.5_1.5_top100_year_loo.yaml"
