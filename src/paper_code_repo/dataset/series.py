from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils.files import normalize_label


@dataclass(frozen=True)
class SeriesSummary:
    series_key: str
    modality_guess: str
    file_count: int
    sample_file: str


def guess_modality(text: str | None) -> str:
    value = normalize_label(text or "")
    if any(token in value for token in ("T1", "T 1", "MPRAGE", "SPGR")):
        return "T1"
    if "FLAIR" in value:
        return "FLAIR"
    if any(token in value for token in ("T2", "T 2")):
        return "T2"
    return "UNKNOWN"


def optional_pydicom():
    try:
        import pydicom  # type: ignore

        return pydicom
    except Exception:
        return None


def read_dicom_header(path: str | Path) -> dict[str, Any] | None:
    pydicom = optional_pydicom()
    if pydicom is None:
        return None
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return None

    return {
        "series_instance_uid": getattr(ds, "SeriesInstanceUID", ""),
        "series_description": getattr(ds, "SeriesDescription", ""),
        "protocol_name": getattr(ds, "ProtocolName", ""),
        "sequence_name": getattr(ds, "SequenceName", ""),
        "modality": getattr(ds, "Modality", ""),
        "instance_number": getattr(ds, "InstanceNumber", None),
        "image_position_patient": getattr(ds, "ImagePositionPatient", None),
        "rows": getattr(ds, "Rows", None),
        "columns": getattr(ds, "Columns", None),
    }


def group_dicom_series(patient_dir: str | Path) -> list[SeriesSummary]:
    patient_path = Path(patient_dir)
    pydicom = optional_pydicom()
    series_map: dict[str, list[Path]] = {}
    modality_map: dict[str, str] = {}

    dicom_files = sorted(patient_path.rglob("*.dcm"))
    for dicom_path in dicom_files:
        header = read_dicom_header(dicom_path)
        if header is None:
            series_key = dicom_path.parent.name
            guess = "UNKNOWN"
        else:
            series_key = header["series_instance_uid"] or dicom_path.parent.name
            guess = guess_modality(
                " ".join(
                    str(value)
                    for value in (
                        header["series_description"],
                        header["protocol_name"],
                        header["sequence_name"],
                        header["modality"],
                    )
                    if value
                )
            )

        series_map.setdefault(series_key, []).append(dicom_path)
        if guess != "UNKNOWN":
            modality_map[series_key] = guess

    summaries: list[SeriesSummary] = []
    for key, files in sorted(series_map.items()):
        summaries.append(
            SeriesSummary(
                series_key=key,
                modality_guess=modality_map.get(key, "UNKNOWN"),
                file_count=len(files),
                sample_file=str(files[0]),
            )
        )
    return summaries

