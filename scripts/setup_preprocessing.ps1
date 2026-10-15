$ErrorActionPreference = "Stop"

$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $RepositoryRoot ".venv\Scripts\python.exe"
$Dcm2niixDir = Join-Path $RepositoryRoot ".tools\dcm2niix"
$Dcm2niixVersion = "v1.0.20260416"
$Dcm2niixUrl = "https://github.com/rordenlab/dcm2niix/releases/download/$Dcm2niixVersion/dcm2niix_win.zip"

if (-not (Test-Path $VenvPython)) {
    python -m venv (Join-Path $RepositoryRoot ".venv")
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -e "${RepositoryRoot}[preprocessing,test]"

New-Item -ItemType Directory -Force -Path $Dcm2niixDir | Out-Null
$Archive = Join-Path $Dcm2niixDir "dcm2niix_win.zip"
Invoke-WebRequest -Uri $Dcm2niixUrl -OutFile $Archive
Expand-Archive -Force $Archive $Dcm2niixDir

Write-Host "Preprocessing environment is ready."
Write-Host "Run: .\.venv\Scripts\python.exe -m paper_code_repo preprocessing-doctor"
