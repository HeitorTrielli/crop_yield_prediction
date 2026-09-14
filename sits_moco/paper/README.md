# Paper draft (LaTeX)

Journal-style draft on downscaling municipal IBGE PAM soybean productivity labels to pixel predictions with a spectral--temporal Transformer (Sentinel-2 + Xavier climate; spectral-only ablation; intramunicipal maps).

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

Requires a TeX distribution with `natbib`, `booktabs`, `siunitx`, `graphicx`, `hyperref`, `multirow`, and `makecell`.

## Regenerate figures

```bash
python generate_figures.py
```

Rebuilds learning-curve, scatter-panel, and incomplete-series figures from the proposed-model results directory configured in that script.
