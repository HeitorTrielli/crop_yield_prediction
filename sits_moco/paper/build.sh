#!/usr/bin/env bash
# Rebuild the paper PDF (text/bibliography).
# Usage:
#   bash paper/build.sh
#   bash paper/build.sh --figures
set -eu

cd "$(dirname "$0")"

# Git Bash often misses the Windows MiKTeX user PATH.
for d in \
  "$LOCALAPPDATA/Programs/MiKTeX/miktex/bin/x64" \
  "/c/Users/$USER/AppData/Local/Programs/MiKTeX/miktex/bin/x64" \
  "/c/Program Files/MiKTeX/miktex/bin/x64"
do
  if [[ -d "$d" ]]; then
    export PATH="$d:$PATH"
    break
  fi
done

if [[ "${1:-}" == "--figures" ]]; then
  python generate_figures.py
fi

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "pdflatex not found. Prefer: powershell -File paper/build.ps1" >&2
  exit 1
fi

# Prefer pdflatex+bibtex: MiKTeX latexmk needs Perl, which is often missing.
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo "Wrote $(pwd)/main.pdf"
