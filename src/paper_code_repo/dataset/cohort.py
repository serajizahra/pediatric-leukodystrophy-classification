from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from ..preprocessing.inventory import scan_dicom_series, select_modalities


CLASSES = ("ALD", "CIM", "MLC", "MLD", "PMDL")


def _class_label(name: str) -> str | None:
    value = re.sub(r"[^A-Z0-9]+", " ", name.upper()).strip()
    if value in CLASSES:
        return value
    if value in {"COMPLEX I", "MITOCHONDRIAL COMPLEX I", "MIT COMPLEX I"}:
        return "CIM"
    tokens = value.split()
    for label in ("PMDL", "MLC", "MLD", "ALD", "CIM"):
        if tokens and tokens[-1] == label:
            return label
    if value.endswith(("MIT COMPLEX I", "MITOCHONDRIAL COMPLEX I")):
        return "CIM"
    return None


def _patient_directories(root: Path) -> list[tuple[str, Path]]:
    patients: list[tuple[str, Path]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        label = _class_label(child.name)
        if child.name.upper() in CLASSES or child.name.upper() in {
            "COMPLEX I", "MITOCHONDRIAL COMPLEX I",
        }:
            patients.extend(
                (label, path) for path in sorted(child.iterdir()) if path.is_dir()
            )
        elif label is not None:
            patients.append((label, child))
    return patients


def discover_cohort(
    data_root: str | Path,
    min_series_slices: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find complete T1/T2/FLAIR DICOM studies in class or patient folders."""
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    records: list[dict[str, object]] = []
    excluded: list[dict[str, str]] = []
    patient_counts = {label: 0 for label in CLASSES}
    for label, patient_dir in _patient_directories(root):
        patient_counts[label] += 1
        patient_id = f"{label}_P{patient_counts[label]:04d}"
        series = scan_dicom_series(patient_dir)
        if not series:
            excluded.append({"patient_id": patient_id, "input_dir": str(patient_dir), "reason": "No DICOM series"})
            continue
        study_uids = sorted({item.study_uid for item in series})
        for exam_number, study_uid in enumerate(study_uids, start=1):
            candidates = [item for item in series if item.study_uid == study_uid]
            try:
                selected = select_modalities(candidates, min_slices=min_series_slices)
            except RuntimeError as exc:
                excluded.append({
                    "patient_id": patient_id,
                    "input_dir": str(patient_dir),
                    "reason": f"Study {exam_number}: {exc}",
                })
                continue
            records.append({
                "patient_id": patient_id,
                "exam_id": f"{patient_id}_E{exam_number:02d}",
                "label": label,
                "study_uid": study_uid,
                "input_dir": str(patient_dir),
                **{f"{modality.lower()}_series_uid": selected[modality].uid for modality in ("T1", "T2", "FLAIR")},
            })
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No complete T1/T2/FLAIR DICOM studies were found")
    if frame.groupby("patient_id")["label"].nunique().gt(1).any():
        raise ValueError("A patient cannot have multiple class labels")
    return frame, pd.DataFrame(excluded, columns=["patient_id", "input_dir", "reason"])
