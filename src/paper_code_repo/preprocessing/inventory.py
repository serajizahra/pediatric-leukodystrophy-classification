from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pydicom


MODALITIES = ("T1", "T2", "FLAIR")


@dataclass
class DicomSeries:
    uid: str
    study_uid: str = ""
    files: list[Path] = field(default_factory=list, repr=False)
    series_description: str = ""
    protocol_name: str = ""
    sequence_name: str = ""
    image_type: tuple[str, ...] = ()
    rows: int | None = None
    columns: int | None = None
    pixel_spacing: tuple[float, ...] = ()
    slice_thickness: float | None = None
    spacing_between_slices: float | None = None
    echo_time: float | None = None
    repetition_time: float | None = None
    acquisition_number: int | None = None
    series_number: int | None = None
    acquisition_plane: str = "unknown"

    @property
    def count(self) -> int:
        return len(self.files)

    @property
    def search_text(self) -> str:
        return " ".join(
            [self.series_description, self.protocol_name, self.sequence_name, *self.image_type]
        ).upper()

    def public_dict(self) -> dict[str, object]:
        data = asdict(self)
        data.pop("files", None)
        data["count"] = self.count
        data["sample_file"] = self.files[0].name if self.files else ""
        data["scores"] = {modality: score_series(self, modality) for modality in MODALITIES}
        return data


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _acquisition_plane(header: object) -> str:
    orientation = getattr(header, "ImageOrientationPatient", None)
    if orientation is None or len(orientation) != 6:
        return "unknown"
    axes = np.asarray([float(value) for value in orientation], dtype=float).reshape(2, 3)
    normal = np.abs(np.cross(axes[0], axes[1]))
    return ("sagittal", "coronal", "axial")[int(np.argmax(normal))]


def scan_dicom_series(input_dir: str | Path, study_uid: str | None = None) -> list[DicomSeries]:
    root = Path(input_dir)
    grouped: dict[str, list[tuple[Path, object]]] = defaultdict(list)

    dicom_files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".dcm", ".dicom"}
    )
    for path in dicom_files:
        try:
            header = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if study_uid is not None and str(getattr(header, "StudyInstanceUID", "")) != study_uid:
            continue
        uid = str(getattr(header, "SeriesInstanceUID", "")) or str(path.parent)
        grouped[uid].append((path, header))

    series_list: list[DicomSeries] = []
    for uid, entries in grouped.items():
        path, header = entries[0]
        spacing = getattr(header, "PixelSpacing", ()) or ()
        image_type = getattr(header, "ImageType", ()) or ()
        series_list.append(
            DicomSeries(
                uid=uid,
                study_uid=str(getattr(header, "StudyInstanceUID", "")),
                files=[entry[0] for entry in entries],
                series_description=str(getattr(header, "SeriesDescription", "")),
                protocol_name=str(getattr(header, "ProtocolName", "")),
                sequence_name=str(getattr(header, "SequenceName", "")),
                image_type=tuple(str(item) for item in image_type),
                rows=_int(getattr(header, "Rows", None)),
                columns=_int(getattr(header, "Columns", None)),
                pixel_spacing=tuple(float(item) for item in spacing),
                slice_thickness=_float(getattr(header, "SliceThickness", None)),
                spacing_between_slices=_float(getattr(header, "SpacingBetweenSlices", None)),
                echo_time=_float(getattr(header, "EchoTime", None)),
                repetition_time=_float(getattr(header, "RepetitionTime", None)),
                acquisition_number=_int(getattr(header, "AcquisitionNumber", None)),
                series_number=_int(getattr(header, "SeriesNumber", None)),
                acquisition_plane=_acquisition_plane(header),
            )
        )
    return sorted(series_list, key=lambda item: (-item.count, item.uid))


def score_series(series: DicomSeries, modality: str) -> float:
    modality = modality.upper()
    text = series.search_text
    score = 0.0

    excluded = (
        "LOCALIZER",
        "SCOUT",
        "DWI",
        "DIFF",
        "ADC",
        "TRACE",
        "PROJECTION IMAGE",
        "SCREEN SAVE",
        "SECONDARY",
        "DERIVED",
        "MPR -",
    )
    if any(token in text for token in excluded):
        score -= 200.0

    if "ORIGINAL" in text:
        score += 10.0
    if "PRIMARY" in text:
        score += 5.0
    if "3D" in text:
        score += 30.0
    if series.acquisition_plane == "axial":
        score += 25.0
    elif series.acquisition_plane in {"sagittal", "coronal"} and "3D" not in text:
        score -= 20.0
    score += min(series.count, 250) / 10.0

    if series.count < 20:
        score -= 60.0
    if series.slice_thickness is not None:
        if series.slice_thickness <= 1.5:
            score += 12.0
        elif series.slice_thickness > 4.0:
            score -= 15.0

    if modality == "FLAIR":
        score += 150.0 if "FLAIR" in text else -120.0
    elif modality == "T1":
        if "FLAIR" in text or "T2" in text:
            score -= 150.0
        if any(token in text for token in ("T1", "MPRAGE", "SPGR", "BRAVO")):
            score += 150.0
        if "TFE" in text and "T1" in text:
            score += 25.0
    elif modality == "T2":
        if "FLAIR" in text or "T1" in text:
            score -= 150.0
        if any(token in text for token in ("T2", "TSE", "SPACE", "BRAIN_VIEW", "CUBE")):
            score += 150.0
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    return score


def select_modalities(series_list: Iterable[DicomSeries], min_slices: int = 20) -> dict[str, DicomSeries]:
    candidates = list(series_list)
    selected: dict[str, DicomSeries] = {}
    used_uids: set[str] = set()

    for modality in MODALITIES:
        eligible = [series for series in candidates
                    if series.count >= min_slices and series.uid not in used_uids
                    and score_series(series, modality) > 0]
        axial = [series for series in eligible if series.acquisition_plane == "axial"]
        ranked = sorted(
            axial or eligible,
            key=lambda series: (score_series(series, modality), series.count, series.uid),
            reverse=True,
        )
        if not ranked or score_series(ranked[0], modality) <= 0:
            raise RuntimeError(f"No suitable {modality} DICOM series was found")
        selected[modality] = ranked[0]
        used_uids.add(ranked[0].uid)
    return selected

