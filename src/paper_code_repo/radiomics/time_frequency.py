from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from .feature_utils import aggregate_slice_values, axial_slices, coefficient_statistics, feature_slug
from .manual_config import HandcraftedFeatureConfig
from .transforms import dcst_spatial_subbands, dct_spatial_subbands, swt_spatial_subbands


TRANSFORM_SUBBANDS = {
    "dct": dct_spatial_subbands,
    "swt": swt_spatial_subbands,
    "dcst": dcst_spatial_subbands,
}


def extract_time_frequency_roi_features(
    volume: np.ndarray,
    roi_labels: np.ndarray,
    modality: str,
    roi_ids: Iterable[int],
    config: HandcraftedFeatureConfig | None = None,
    transforms: Iterable[str] = ("dct", "swt", "dcst"),
    max_slices: int | None = None,
) -> dict[str, float]:
    """Extract 2D slice-based DCT/SWT/DCST features from 40 anatomical ROIs.

    For every axial slice, transform subbands are isolated and transformed back
    to image space. Eleven statistics are computed inside each ROI, then each
    statistic is summarized across slices by mean, std, min, and max.
    """

    config = config or HandcraftedFeatureConfig()
    labels = np.rint(roi_labels).astype(np.int32)
    roi_ids = [int(value) for value in roi_ids]
    streams: dict[str, list[float]] = defaultdict(list)
    modality_name = feature_slug(modality)

    for _, image_slice, label_slice in axial_slices(volume, labels, axis=config.slice_axis, max_slices=max_slices):
        if not np.any(np.isin(label_slice, roi_ids)):
            continue
        for transform_name in transforms:
            subband_fn = TRANSFORM_SUBBANDS[feature_slug(transform_name).lower()]
            subbands = subband_fn(image_slice, config)
            for roi_id in roi_ids:
                roi_mask = label_slice == roi_id
                if int(roi_mask.sum()) < config.minimum_roi_voxels:
                    continue
                energies: dict[str, float] = {}
                stats_by_band: dict[str, dict[str, float]] = {}
                for band_name, band_image in subbands.items():
                    values = band_image[roi_mask]
                    stats = coefficient_statistics(values)
                    stats_by_band[band_name] = stats
                    energies[band_name] = stats["energy"]
                total_energy = float(sum(energies.values()))
                for band_name, stats in stats_by_band.items():
                    if total_energy > 0:
                        stats["relative_energy"] = float(energies[band_name] / total_energy)
                    for stat_name, value in stats.items():
                        stream_name = (
                            f"{modality_name}__tf_{feature_slug(transform_name)}"
                            f"__roi_{roi_id:02d}__band_{feature_slug(band_name)}__{feature_slug(stat_name)}"
                        )
                        streams[stream_name].append(float(value))

    features: dict[str, float] = {}
    for stream_name, values in streams.items():
        aggregated = aggregate_slice_values(values)
        for agg_name, agg_value in aggregated.items():
            features[f"{stream_name}__slice_{agg_name}"] = float(agg_value)
    return features
