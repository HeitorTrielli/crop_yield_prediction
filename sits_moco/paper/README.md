# Paper draft (LaTeX)

Journal-style manuscript on MSTNet: multi-frequency STNet trunks for Sentinel-2 + daily Xavier climate + late soil fusion, trained under municipal IBGE PAM labels with pixel scoring for sub-municipal downscaling (Paraná, 2020--2024; 2021 holdout).

## Build

From this directory:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or:

```bash
latexmk -pdf main.tex
```

Requires a TeX distribution with `natbib`, `booktabs`, `siunitx`, `graphicx`, `hyperref`, `multirow`, `makecell`, and `tikz`.

## Regenerate figures

```bash
python generate_figures.py
```

Rebuilds learning-curve and scatter-panel figures from
`results/tuning/productivity_muni_mean_dual_stnet/trial_859` (configured in that script).
