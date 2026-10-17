# Rebuild the paper PDF (text/bibliography).
# Usage (from repo root or paper/):
#   .\paper\build.ps1
#   .\paper\build.ps1 -Figures
param(
    [switch]$Figures
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($Figures) {
    python generate_figures.py
    if ($LASTEXITCODE -ne 0) { throw "generate_figures.py failed" }
}

function Find-TeX {
    param([string]$Name)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in @(
        "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64\$Name.exe",
        "$env:ProgramFiles\MiKTeX\miktex\bin\x64\$Name.exe"
    )) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

function Invoke-TeX {
    param([string]$Exe, [Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
    & $Exe @Args
    if ($LASTEXITCODE -ne 0) { throw "$Exe failed with exit code $LASTEXITCODE" }
}

$pdflatex = Find-TeX "pdflatex"
$bibtex = Find-TeX "bibtex"
if (-not $pdflatex) {
    throw "pdflatex not found. Install MiKTeX or add it to PATH."
}
if (-not $bibtex) {
    throw "bibtex not found."
}

# Prefer pdflatex+bibtex: MiKTeX latexmk needs Perl, which is often missing.
Invoke-TeX $pdflatex -interaction=nonstopmode main.tex
Invoke-TeX $bibtex main
Invoke-TeX $pdflatex -interaction=nonstopmode main.tex
Invoke-TeX $pdflatex -interaction=nonstopmode main.tex

Write-Host "Wrote $(Join-Path (Get-Location) 'main.pdf')"
