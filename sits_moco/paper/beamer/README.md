# Beamer presentation

Slides for the MSTNet paper (Paraná year-LOO, municipal → pixel downscaling).

## Build

```powershell
.\build.ps1
```

Or:

```bash
pdflatex main.tex && pdflatex main.tex
```

Figures are loaded from `../figures/`. Requires a TeX distribution with Beamer, `booktabs`, `siunitx`, `graphicx`, and `tikz`.
