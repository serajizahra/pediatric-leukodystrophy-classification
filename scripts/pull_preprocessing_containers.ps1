$ErrorActionPreference = "Stop"

docker info | Out-Null
docker pull freesurfer/freesurfer:8.2.0
docker pull antsx/ants:v2.6.5

Write-Host "SynthStrip and SynthSeg containers are ready."

