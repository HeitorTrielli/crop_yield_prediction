# Paper draft (LaTeX)

Journal-style manuscript on MSTNet: multi-frequency STNet trunks for Sentinel-2 + daily Xavier climate + late soil fusion, trained under municipal IBGE PAM labels with pixel scoring for sub-municipal downscaling (Paraná, 2020--2024; 2021 holdout).

## Build

From the repo root (PowerShell recommended on Windows):

```powershell
.\paper\build.ps1           # text / bibliography only
.\paper\build.ps1 -Figures  # also regenerate figures first
```

Or with Git Bash:

```bash
bash paper/build.sh
bash paper/build.sh --figures
```

Uses `latexmk` when available, otherwise `pdflatex` + `bibtex` + two more `pdflatex` passes.

Requires a TeX distribution with `natbib`, `booktabs`, `siunitx`, `graphicx`, `hyperref`, `multirow`, `makecell`, and `tikz`.

Figures are pulled from `loo_pixel_results/pr_only_trial_005` via `generate_figures.py`.
