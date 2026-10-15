from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from ..utils.files import normalize_label, slugify


IMAGE_EXTENSIONS = {".dcm", ".dicom", ".nii", ".nii.gz", ".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class PatientRecord:
    class_label: str
    patient_id: str
    class_folder: str
    patient_name: str
    patient_path: str
    patient_slug: str
    file_count: int
    dicom_count: int
    image_count: int
    nested_folder_count: int
    sample_file: str


def _iter_patient_dirs(class_dir: Path) -> Iterable[Path]:
    for child in sorted(class_dir.iterdir()):
        if child.is_dir():
            yield child


def _count_files(patient_dir: Path) -> tuple[int, int, int, int, str]:
    file_count = 0
    dicom_count = 0
    image_count = 0
    nested_folder_count = 0
    first_dicom = ""
    first_image = ""

    for item in sorted(patient_dir.rglob("*")):
        if item.is_dir():
            if item != patient_dir:
                nested_folder_count += 1
            continue
        if not item.is_file():
            continue
        suffix = "".join(item.suffixes).lower()
        if suffix.endswith(".dcm") or suffix == ".dcm":
            file_count += 1
            dicom_count += 1
            if not first_dicom:
                first_dicom = str(item)
        elif suffix in IMAGE_EXTENSIONS:
            file_count += 1
            image_count += 1
            if not first_image:
                first_image = str(item)

    sample_file = first_dicom or first_image
    return file_count, dicom_count, image_count, nested_folder_count, sample_file


def discover_dataset(root: str | Path) -> List[PatientRecord]:
    root_path = Path(root)
    records: list[PatientRecord] = []

    for class_dir in sorted(root_path.iterdir()):
        if not class_dir.is_dir():
            continue
        class_label = normalize_label(class_dir.name)

        for patient_dir in _iter_patient_dirs(class_dir):
            file_count, dicom_count, image_count, nested_folder_count, sample_file = _count_files(patient_dir)
            records.append(
                PatientRecord(
                    class_label=class_label,
                    patient_id=f"{class_label}_{slugify(patient_dir.name)}",
                    class_folder=class_dir.name,
                    patient_name=patient_dir.name,
                    patient_path=str(patient_dir),
                    patient_slug=slugify(patient_dir.name),
                    file_count=file_count,
                    dicom_count=dicom_count,
                    image_count=image_count,
                    nested_folder_count=nested_folder_count,
                    sample_file=sample_file,
                )
            )

    return records


def records_to_frame(records: Iterable[PatientRecord]) -> pd.DataFrame:
    return pd.DataFrame([asdict(record) for record in records])
