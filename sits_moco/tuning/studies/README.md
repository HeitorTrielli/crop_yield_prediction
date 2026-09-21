# Tuning studies

| Folder | Role |
|--------|------|
| `*.yaml` (this directory) | Current primary studies |
| `secondary/` | Municipal-aggregate and zonal ablations |
| `archive/` | Historical / one-off grids |

Run:

```bash
python run_tuning_study.py run tuning/studies/productivity_soil_sidecar_moco.yaml --skip-completed
```
