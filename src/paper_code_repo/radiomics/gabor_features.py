from __future__ import annotations

import numpy as np
from skimage.filters import gabor

from .feature_utils import entropy_from_values, feature_slug, safe_values
from .manual_config import HandcraftedFeatureConfig


def _gabor_response(slice_image: np.ndarray, frequency: float, theta: float, offset: float) -> np.ndarray:
    try:
        real, imag = gabor(slice_image, frequency=frequency, theta=theta, offset=offset)
    except Exception:
        # Configured frequencies may exceed the discrete Nyquist limit.
        # Rescale only for digital filtering while preserving the
        # original frequency in the feature name.
        safe_frequency = min(abs(float(frequency)), 0.49)
        real, imag = gabor(slice_image, frequency=safe_frequency, theta=theta, offset=offset)
    return np.sqrt(real**2 + imag**2)


def extract_gabor_volume_features(
    volume: np.ndarray,
    brain_mask: np.ndarray,
    modality: str,
    config: HandcraftedFeatureConfig | None = None,
    max_slices: int | None = None,
) -> dict[str, float]:
    """Extract 32 Gabor responses and four descriptors per MRI volume.

    The filter bank uses four frequencies, four azimuths, and two phase
    offsets. Descriptors are mean, std, energy, and entropy.
    """

    config = config or HandcraftedFeatureConfig()
    modality_name = feature_slug(modality)
    n_slices = volume.shape[config.slice_axis]
    indices: list[int] | range = range(n_slices)
    if max_slices is not None and 0 < max_slices < n_slices:
        indices = sorted(set(int(index) for index in np.linspace(0, n_slices - 1, num=max_slices, dtype=int)))

    features: dict[str, float] = {}
    for frequency in config.gabor_frequencies:
        for azimuth in config.gabor_azimuth_degrees:
            theta = np.deg2rad(float(azimuth))
            for phase in config.gabor_phase_degrees:
                offset = np.deg2rad(float(phase))
                values: list[np.ndarray] = []
                for slice_index in indices:
                    image_slice = np.take(volume, slice_index, axis=config.slice_axis)
                    mask_slice = np.take(brain_mask, slice_index, axis=config.slice_axis) > 0
                    if not mask_slice.any():
                        continue
                    response = _gabor_response(image_slice, float(frequency), theta, offset)
                    values.append(response[mask_slice])
                arr = safe_values(np.concatenate(values) if values else np.asarray([], dtype=np.float64))
                prefix = (
                    f"{modality_name}__gabor"
                    f"__freq_{feature_slug(frequency)}"
                    f"__azimuth_{feature_slug(azimuth)}"
                    f"__phase_{feature_slug(phase)}"
                )
                if arr.size == 0:
                    features[f"{prefix}__mean"] = 0.0
                    features[f"{prefix}__std"] = 0.0
                    features[f"{prefix}__energy"] = 0.0
                    features[f"{prefix}__entropy"] = 0.0
                else:
                    features[f"{prefix}__mean"] = float(np.mean(arr))
                    features[f"{prefix}__std"] = float(np.std(arr))
                    features[f"{prefix}__energy"] = float(np.sum(arr**2))
                    features[f"{prefix}__entropy"] = entropy_from_values(arr)
    return features
