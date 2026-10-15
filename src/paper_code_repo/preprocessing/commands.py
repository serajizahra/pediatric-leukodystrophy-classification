from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import which


@dataclass(frozen=True)
class CommandStep:
    name: str
    command: list[str]
    cwd: str | None = None


def tool_available(name: str) -> bool:
    return which(name) is not None


def dcm2niix_command(input_dir: str | Path, output_dir: str | Path, compress: bool = True) -> list[str]:
    command = [
        "dcm2niix",
        "-z",
        "y" if compress else "n",
        "-o",
        str(output_dir),
        str(input_dir),
    ]
    return command


def ants_registration_command(fixed_image: str | Path, moving_image: str | Path, output_prefix: str | Path) -> list[str]:
    return [
        "antsRegistrationSyN.sh",
        "-d",
        "3",
        "-f",
        str(fixed_image),
        "-m",
        str(moving_image),
        "-o",
        str(output_prefix),
    ]


def synthstrip_command(input_image: str | Path, output_image: str | Path, mask_image: str | Path | None = None) -> list[str]:
    command = ["SynthStrip", "-i", str(input_image), "-o", str(output_image)]
    if mask_image is not None:
        command.extend(["-m", str(mask_image)])
    return command


def synthseg_command(input_image: str | Path, output_dir: str | Path) -> list[str]:
    return [
        "SynthSeg",
        "--i",
        str(input_image),
        "--o",
        str(output_dir),
    ]

