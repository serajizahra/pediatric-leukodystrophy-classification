from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HandcraftedFeatureConfig:
    """Default radiomics and transform feature settings."""

    pyradiomics_bin_width: float = 25.0
    ulbp_points: int = 16
    ulbp_radius: float = 1.0
    gabor_frequencies: tuple[float, ...] = (0.1, 0.3, 0.5, 1.5)
    gabor_azimuth_degrees: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)
    gabor_phase_degrees: tuple[float, ...] = (0.0, 45.0)
    wavelet: str = "haar"
    frequency_band_fractions: tuple[float, float] = (0.25, 0.50)
    slice_axis: int = 2
    minimum_roi_voxels: int = 8
