# sits_moco — Paraná soy productivity (pixel-level)

Predict municipal soy **productivity** (t/ha) from Sentinel-2 time series at **pixel** resolution, then aggregate to municipalities for training against IBGE/PAM labels. Optional **MoCo** pretraining warms the encoder.

Forked from [SITS-MoCo](http://dx.doi.org/10.1016/j.isprsjprs.2023.12.005) (US crop mapping). The live product path is Brazilian yield regression, not US classification.

## Setup

```bash
uv sync
cp .env.example .env   # set SITS_MOCO_DATAPATH to your municipal .npy root
```

Requires Python `>=3.10,<3.13`. Dependencies live in [`pyproject.toml`](pyproject.toml) (do not use the old conda `environment.yml` workflow).

## Primary pipeline

1. **Download** imagery / climate / soil / PAM — see [`docs/PIPELINE.md`](docs/PIPELINE.md)
2. **Preprocess** daily TIFF → municipal pixel `.npy` (+ MapBiomas soil sidecars)
3. **Train / tune**
4. **Predict** municipal forecasts and pixel heatmaps

```bash
# Hyperparameter study (recommended)
python run_tuning_study.py run tuning/studies/productivity_soil_sidecar_moco.yaml --skip-completed

# Single training run
python main_yield_regression_polars.py --help

# MoCo encoder pretrain (Paraná pixels)
python main_moco.py transformer --rc --use-doy --useall --mlp

# Inference / maps
python eval/predict_yield.py --help
python viz/create_pixel_heatmap.py --help
python eval/generate_results.py --help
```

Entry points to remember: `main_yield_regression_polars.py`, `main_moco.py`, `run_tuning_study.py`.

## Secondary tracks

Kept for ablations; not the default onboarding path:

- **Municipal aggregate** — one mean/median time series per municipality (`npy_muni_mean` / `npy_muni_median`). Studies under `tuning/studies/secondary/`.
- **Zonal mean** — municipality stats from GEE/BDC zonal CSVs instead of daily pixel cubes.

## Layout (code)

| Path | Role |
|------|------|
| `datasets/` | Pixel loaders, feature layouts, soil sidecars |
| `models/` | `STNetRegression` (+ MoCo transfer) |
| `training/` | Batch/chunk/VRAM training internals |
| `tuning/` | YAML study runner |
| `preprocessing/` | TIFF → npy, aggregation |
| `data_download/` | Ingest (BDC/GEE, Xavier, MapBiomas, PAM) |
| `eval/` | Predict, metrics, result bundles |
| `viz/` | Heatmaps and scatter/choropleth plots |
| `tools/` | Debug and profiling helpers |
| `archive/` | Original US classification paper code |

Full data DAG and study triage: [`docs/PIPELINE.md`](docs/PIPELINE.md).

## Citation

If you use the MoCo / STNet ideas from the original paper:

```bibtex
@article{xu_self-supervised_2024,
  title = {Self-supervised pre-training for large-scale crop mapping using Sentinel-2 time series},
  volume = {207},
  doi = {10.1016/j.isprsjprs.2023.12.005},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  author = {Xu, Yijia and Ma, Yuchi and Zhang, Zhou},
  year = {2024},
}
```

MoCo implementation based on [Facebook MoCo](https://github.com/facebookresearch/moco).
