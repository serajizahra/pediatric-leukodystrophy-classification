from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


def run_command(command: list[str], log_path: str | Path) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        json.dumps(
            {
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed; see {log}")


def docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(["docker", "info"], capture_output=True, check=False)
    return result.returncode == 0


def _container_path(root: Path, path: Path) -> str:
    return "/data/" + path.resolve().relative_to(root.resolve()).as_posix()


def _freesurfer_license_arguments() -> list[str]:
    license_path = os.environ.get("FS_LICENSE") or os.environ.get("FREESURFER_LICENSE")
    if not license_path:
        return []
    resolved = Path(license_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"FreeSurfer license file does not exist: {resolved}")
    return ["-v", f"{resolved}:/license.txt:ro", "-e", "FS_LICENSE=/license.txt"]


def run_synthstrip(
    input_path: Path,
    output_path: Path,
    mask_path: Path,
    patient_root: Path,
    backend: str,
    image: str,
    log_path: Path,
) -> None:
    local = shutil.which("mri_synthstrip") or shutil.which("SynthStrip")
    if backend in {"auto", "local"} and local:
        run_command([local, "-i", str(input_path), "-o", str(output_path), "-m", str(mask_path)], log_path)
        return
    if backend == "local":
        raise RuntimeError("mri_synthstrip was not found")
    if not docker_ready():
        raise RuntimeError("Docker is required for SynthStrip and is not running")
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{patient_root}:/data",
    ]
    command.extend(_freesurfer_license_arguments())
    command.append(image)
    if image.startswith("freesurfer/freesurfer"):
        command.append("mri_synthstrip")
    command.extend([
        "-i",
        _container_path(patient_root, input_path),
        "-o",
        _container_path(patient_root, output_path),
        "-m",
        _container_path(patient_root, mask_path),
    ])
    run_command(command, log_path)


def run_synthseg(
    input_path: Path,
    output_path: Path,
    patient_root: Path,
    backend: str,
    image: str,
    log_path: Path,
) -> None:
    local = shutil.which("mri_synthseg") or shutil.which("SynthSeg")
    if backend in {"auto", "local"} and local:
        run_command(
            [local, "--i", str(input_path), "--o", str(output_path), "--parc"],
            log_path,
        )
        return
    if backend == "local":
        raise RuntimeError("mri_synthseg was not found")
    if not docker_ready():
        raise RuntimeError("Docker is required for SynthSeg and is not running")
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{patient_root}:/data",
    ]
    command.extend(_freesurfer_license_arguments())
    command.append(image)
    if image.startswith("freesurfer/freesurfer"):
        command.append("mri_synthseg")
    command.extend(
        [
            "--i",
            _container_path(patient_root, input_path),
            "--o",
            _container_path(patient_root, output_path),
            "--parc",
            "--threads",
            "1",
            "--cpu",
        ]
    )
    run_command(command, log_path)
