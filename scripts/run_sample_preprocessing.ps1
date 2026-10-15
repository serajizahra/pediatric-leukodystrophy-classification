param(
    [Parameter(Mandatory = $true)]
    [string]$InputDir,
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
if (-not $OutputDir) {
    $OutputDir = Join-Path $RepositoryRoot "outputs\sample_preprocessing"
}

& $Python -m paper_code_repo preprocess-patient `
    --input-dir $InputDir `
    --output-dir $OutputDir `
    --allow-self-reference `
    --synthstrip-backend docker `
    --synthseg-backend docker
