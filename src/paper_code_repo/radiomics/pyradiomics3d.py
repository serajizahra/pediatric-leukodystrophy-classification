from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import nibabel as nib
import numpy as np

from .feature_utils import feature_slug, finite_float
from .manual_config import HandcraftedFeatureConfig


PYRADIOMICS_CLASSES = ("shape", "glcm", "glrlm", "glszm", "ngtdm", "gldm")


def _build_extractor(config: HandcraftedFeatureConfig):
    try:
        from radiomics import featureextractor
        from radiomics import gldm, glcm, glrlm, glszm, ngtdm, shape
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "PyRadiomics is required for 3D shape and texture features. "
            "Install the handcrafted dependencies first."
        ) from exc

    extractor = featureextractor.RadiomicsFeatureExtractor(
        binWidth=config.pyradiomics_bin_width,
        force2D=False,
        normalize=False,
        geometryTolerance=1e-3,
    )
    extractor.disableAllFeatures()

    # Include deprecated shape features for compactness and
    # spherical-disproportion descriptors.
    shape_names = list(shape.RadiomicsShape.getFeatureNames())
    texture_classes = {
        "glcm": glcm.RadiomicsGLCM,
        "glrlm": glrlm.RadiomicsGLRLM,
        "glszm": glszm.RadiomicsGLSZM,
        "ngtdm": ngtdm.RadiomicsNGTDM,
        "gldm": gldm.RadiomicsGLDM,
    }
    texture_names = {
        name: [key for key, deprecated in cls.getFeatureNames().items() if not deprecated]
        for name, cls in texture_classes.items()
    }
    extractor.enableFeaturesByName(shape=shape_names, **texture_names)
    extractor.settings["distances"] = [1]
    return extractor


def _voxel_shape_complements(mask_path: str | Path, roi_id: int) -> dict[str, float]:
    image = nib.load(str(mask_path))
    labels = np.rint(image.get_fdata()).astype(np.int32)
    mask = labels == int(roi_id)
    if not mask.any():
        return {"VoxelSurfaceArea": 0.0, "VoxelSurfaceVolumeRatio": 0.0}

    spacing = np.sqrt((image.affine[:3, :3] ** 2).sum(axis=0))
    voxel_volume = float(np.prod(spacing))
    volume = float(mask.sum() * voxel_volume)
    surface = 0.0
    for axis in range(3):
        padded = np.pad(mask, [(1, 1) if dim == axis else (0, 0) for dim in range(3)], constant_values=False)
        before = np.take(padded, range(0, mask.shape[axis]), axis=axis)
        after = np.take(padded, range(2, mask.shape[axis] + 2), axis=axis)
        area = float(np.prod(np.delete(spacing, axis)))
        surface += float(np.logical_and(mask, ~before).sum() * area)
        surface += float(np.logical_and(mask, ~after).sum() * area)
    return {
        "VoxelSurfaceArea": surface,
        "VoxelSurfaceVolumeRatio": float(surface / volume) if volume > 0 else 0.0,
    }


def extract_pyradiomics_3d_features(
    image_path: str | Path,
    roi_mask_path: str | Path,
    modality: str,
    roi_ids: Iterable[int],
    config: HandcraftedFeatureConfig | None = None,
) -> dict[str, float]:
    """Extract radiomics features from 3D masks using PyRadiomics.

    The output contains PyRadiomics shape descriptors plus 3D texture families
    GLCM, GLRLM, GLSZM, NGTDM, and GLDM with fixed bin width 25.
    """

    config = config or HandcraftedFeatureConfig()
    extractor = _build_extractor(config)
    features: dict[str, float] = {}
    modality_name = feature_slug(modality)

    for roi_id in roi_ids:
        try:
            result = extractor.execute(str(image_path), str(roi_mask_path), label=int(roi_id))
        except Exception:
            result = {}
        prefix = f"{modality_name}__pyrad3d__roi_{int(roi_id):02d}"
        for key, value in result.items():
            if not key.startswith("original_"):
                continue
            family_feature = key.removeprefix("original_")
            features[f"{prefix}__{feature_slug(family_feature)}"] = finite_float(value)
        for key, value in _voxel_shape_complements(roi_mask_path, int(roi_id)).items():
            features[f"{prefix}__shape_{feature_slug(key)}"] = finite_float(value)
    return features
