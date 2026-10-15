from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.ndimage import map_coordinates

from .feature_utils import feature_slug
from .manual_config import HandcraftedFeatureConfig


def fibonacci_sphere_points(n_points: int) -> np.ndarray:
    points = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for index in range(n_points):
        z = 1.0 - (2.0 * index + 1.0) / n_points
        radius = np.sqrt(max(0.0, 1.0 - z * z))
        theta = golden_angle * index
        points.append((np.cos(theta) * radius, np.sin(theta) * radius, z))
    return np.asarray(points, dtype=np.float64)


def uniform_lbp_bins(bits: np.ndarray, n_points: int) -> np.ndarray:
    transitions = np.count_nonzero(bits != np.roll(bits, -1, axis=1), axis=1)
    one_counts = np.count_nonzero(bits, axis=1)
    bins = np.where(transitions <= 2, one_counts, n_points + 1)
    return bins.astype(np.int16)


def extract_ulbp3d_features(
    volume: np.ndarray,
    roi_labels: np.ndarray,
    modality: str,
    roi_ids: Iterable[int],
    config: HandcraftedFeatureConfig | None = None,
) -> dict[str, float]:
    """Extract 3D uniform LBP histograms with P=16 and R=1.

    Neighbor intensities are sampled on a sphere by trilinear interpolation.
    The final histogram has P+2 bins and is L1-normalized per ROI.
    """

    config = config or HandcraftedFeatureConfig()
    labels = np.rint(roi_labels).astype(np.int32)
    valid = np.isin(labels, list(roi_ids))
    coords = np.column_stack(np.nonzero(valid)).astype(np.float64)
    if coords.size == 0:
        return {}

    center_values = volume[tuple(coords.astype(np.intp).T)]
    directions = fibonacci_sphere_points(config.ulbp_points) * float(config.ulbp_radius)
    bits = np.zeros((coords.shape[0], config.ulbp_points), dtype=bool)
    for point_index, direction in enumerate(directions):
        sample_coords = (coords + direction).T
        sampled = map_coordinates(volume, sample_coords, order=1, mode="nearest")
        bits[:, point_index] = sampled >= center_values
    bins = uniform_lbp_bins(bits, config.ulbp_points)
    voxel_labels = labels[tuple(coords.astype(np.intp).T)]

    features: dict[str, float] = {}
    modality_name = feature_slug(modality)
    edges = np.arange(config.ulbp_points + 3) - 0.5
    for roi_id in roi_ids:
        roi_bins = bins[voxel_labels == int(roi_id)]
        hist, _ = np.histogram(roi_bins, bins=edges)
        total = hist.sum()
        normalized = hist.astype(np.float64) / float(total) if total > 0 else hist.astype(np.float64)
        for bin_index, value in enumerate(normalized):
            key = f"{modality_name}__ulbp3d__roi_{int(roi_id):02d}__bin_{bin_index:02d}"
            features[key] = float(value)
    return features
