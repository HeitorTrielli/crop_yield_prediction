# Build the Beamer presentation (paper/beamer/main.tex).
# Usage: .\build.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

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

$pdflatex = Find-TeX "pdflatex"
if (-not $pdflatex) {
    throw "pdflatex not found. Install MiKTeX or add it to PATH."
}

& $pdflatex -interaction=nonstopmode main.tex
if ($LASTEXITCODE -ne 0) { throw "pdflatex failed with exit code $LASTEXITCODE" }
& $pdflatex -interaction=nonstopmode main.tex
if ($LASTEXITCODE -ne 0) { throw "pdflatex failed with exit code $LASTEXITCODE" }

Write-Host "Wrote $(Join-Path (Get-Location) 'main.pdf')"
