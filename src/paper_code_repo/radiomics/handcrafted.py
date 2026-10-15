from __future__ import annotations

import json
from importlib import metadata as importlib_metadata
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from .feature_utils import finite_float, roi_ids_from_labels
from .gabor_features import extract_gabor_volume_features
from .manual_config import HandcraftedFeatureConfig
from .pyradiomics3d import extract_pyradiomics_3d_features
from .time_frequency import extract_time_frequency_roi_features
from .ulbp3d import extract_ulbp3d_features


def load_volume(path: str | Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    return np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32), image


def load_labels(path: str | Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    labels = np.rint(image.get_fdata()).astype(np.int32)
    return labels, image


def _geometry_signature(image: nib.Nifti1Image) -> dict[str, object]:
    return {
        "shape": list(image.shape[:3]),
        "zooms": [float(value) for value in image.header.get_zooms()[:3]],
        "affine": np.round(image.affine, 6).tolist(),
    }


def validate_geometry(reference: nib.Nifti1Image, *others: nib.Nifti1Image) -> None:
    ref = _geometry_signature(reference)
    for image in others:
        if _geometry_signature(image) != ref:
            raise ValueError("Image, brain mask, and ROI mask must share shape, spacing, and affine.")


def _package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "not-installed"


def extract_handcrafted_features_for_image(
    image_path: str | Path,
    roi_mask_path: str | Path,
    brain_mask_path: str | Path,
    modality: str,
    patient_id: str = "patient",
    config: HandcraftedFeatureConfig | None = None,
    roi_ids: list[int] | None = None,
    include_pyradiomics: bool = True,
    include_ulbp: bool = True,
    include_gabor: bool = True,
    include_time_frequency: bool = True,
    max_slices: int | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    config = config or HandcraftedFeatureConfig()
    volume, image = load_volume(image_path)
    labels, label_image = load_labels(roi_mask_path)
    brain_mask, brain_image = load_labels(brain_mask_path)
    validate_geometry(image, label_image, brain_image)
    roi_ids = roi_ids_from_labels(labels, requested=roi_ids)

    feature_row: dict[str, float | str] = {"patient_id": patient_id, "modality": modality}
    sections: dict[str, int] = {}

    if include_pyradiomics:
        extracted = extract_pyradiomics_3d_features(image_path, roi_mask_path, modality, roi_ids, config)
        feature_row.update(extracted)
        sections["pyradiomics_3d"] = len(extracted)

    if include_ulbp:
        extracted = extract_ulbp3d_features(volume, labels, modality, roi_ids, config)
        feature_row.update(extracted)
        sections["ulbp_3d"] = len(extracted)

    if include_gabor:
        extracted = extract_gabor_volume_features(volume, brain_mask, modality, config, max_slices=max_slices)
        feature_row.update(extracted)
        sections["gabor"] = len(extracted)

    if include_time_frequency:
        extracted = extract_time_frequency_roi_features(
            volume,
            labels,
            modality,
            roi_ids,
            config,
            transforms=("dct", "swt", "dcst"),
            max_slices=max_slices,
        )
        feature_row.update(extracted)
        sections["time_frequency_roi"] = len(extracted)

    for key, value in list(feature_row.items()):
        if key in {"patient_id", "modality"}:
            continue
        feature_row[key] = finite_float(value)

    frame = pd.DataFrame([feature_row])
    metadata = {
        "patient_id": patient_id,
        "modality": modality,
        "image_path": str(Path(image_path)),
        "roi_mask_path": str(Path(roi_mask_path)),
        "brain_mask_path": str(Path(brain_mask_path)),
        "geometry": _geometry_signature(image),
        "roi_ids": roi_ids,
        "feature_sections": sections,
        "feature_count": int(frame.shape[1] - 2),
        "max_slices": max_slices,
        "parameters": {
            "pyradiomics_bin_width": config.pyradiomics_bin_width,
            "ulbp_points": config.ulbp_points,
            "ulbp_radius": config.ulbp_radius,
            "gabor_frequencies": list(config.gabor_frequencies),
            "gabor_azimuth_degrees": list(config.gabor_azimuth_degrees),
            "gabor_phase_degrees": list(config.gabor_phase_degrees),
            "wavelet": config.wavelet,
            "frequency_band_fractions": list(config.frequency_band_fractions),
        },
        "software_versions": {
            "numpy": np.__version__,
            "nibabel": nib.__version__,
            "pandas": pd.__version__,
            "pyradiomics": _package_version("pyradiomics"),
            "PyWavelets": _package_version("PyWavelets"),
            "scikit-image": _package_version("scikit-image"),
            "scipy": _package_version("scipy"),
        },
    }
    return frame, metadata


def extract_multimodal_handcrafted_features(
    images: dict[str, str | Path],
    roi_mask_path: str | Path,
    brain_mask_path: str | Path,
    patient_id: str,
    config: HandcraftedFeatureConfig | None = None,
    max_slices: int | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Concatenate T1, T2 and FLAIR descriptors in one examination row."""
    required = ("T1", "T2", "FLAIR")
    if set(images) != set(required):
        raise ValueError("Exactly T1, T2 and FLAIR images are required")
    row: dict[str, float | str] = {"patient_id": patient_id}
    metadata: dict[str, object] = {"patient_id": patient_id, "modalities": {}}
    for modality in required:
        frame, details = extract_handcrafted_features_for_image(
            images[modality], roi_mask_path, brain_mask_path, modality,
            patient_id=patient_id, config=config, max_slices=max_slices,
        )
        for key, value in frame.iloc[0].items():
            if key not in {"patient_id", "modality"}:
                if key in row:
                    raise ValueError(f"Duplicate multimodal feature name: {key}")
                row[key] = value
        metadata["modalities"][modality] = details
    result = pd.DataFrame([row])
    metadata["feature_count"] = result.shape[1] - 1
    return result, metadata


def save_handcrafted_features(
    output_csv: str | Path,
    frame: pd.DataFrame,
    metadata: dict[str, object],
    metadata_path: str | Path | None = None,
) -> None:
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    meta_path = Path(metadata_path) if metadata_path else output.with_suffix(".json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_feature_frame(frame: pd.DataFrame) -> dict[str, object]:
    numeric = frame.drop(columns=[column for column in ("patient_id", "modality") if column in frame.columns])
    values = numeric.to_numpy(dtype=np.float64)
    finite = bool(np.isfinite(values).all()) if values.size else True
    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "feature_count": int(numeric.shape[1]),
        "all_numeric_features_finite": finite,
        "nan_count": int(np.isnan(values).sum()) if values.size else 0,
        "inf_count": int(np.isinf(values).sum()) if values.size else 0,
    }
