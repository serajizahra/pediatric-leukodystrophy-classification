"""Training-only spatial and intensity augmentation before extracting ROI patches."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, rotate, shift, zoom


def _center_crop_or_pad(array: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    out = np.zeros(shape, dtype=array.dtype)
    source = []
    destination = []
    for old, new in zip(array.shape, shape):
        length = min(old, new)
        source.append(slice((old - length) // 2, (old - length) // 2 + length))
        destination.append(slice((new - length) // 2, (new - length) // 2 + length))
    out[tuple(destination)] = array[tuple(source)]
    return out


def augment_full_volume(
    volume: np.ndarray, labels: np.ndarray, rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same 3D geometry to T1/T2/FLAIR and the nearest-neighbor ROI map."""
    rng = rng if rng is not None else np.random.default_rng(np.random.randint(0, 2**32 - 1))
    shape = labels.shape
    volumes = [volume[i] for i in range(volume.shape[0])]
    arrays = volumes + [labels]
    orders = [1] * len(volumes) + [0]

    if rng.random() < 0.5:
        for axes in ((0, 1), (0, 2), (1, 2)):
            angle = float(rng.uniform(5.0, 15.0) * rng.choice([-1, 1]))
            arrays = [rotate(a, angle, axes=axes, reshape=False, order=o, mode="constant")
                      for a, o in zip(arrays, orders)]
    if rng.random() < 0.5:
        scale = float(rng.uniform(0.9, 1.1))
        arrays = [_center_crop_or_pad(zoom(a, scale, order=o), shape) for a, o in zip(arrays, orders)]
    if rng.random() < 0.5:
        offset = float(rng.uniform(-0.1, 0.1) * shape[0])
        arrays = [shift(a, (offset, 0, 0), order=o, mode="constant") for a, o in zip(arrays, orders)]
    if rng.random() < 0.5:
        displacement = [gaussian_filter(rng.normal(size=shape), sigma=8) * 50 for _ in range(3)]
        coordinates = np.meshgrid(*[np.arange(length) for length in shape], indexing="ij")
        warped = [grid + delta for grid, delta in zip(coordinates, displacement)]
        arrays = [map_coordinates(a, warped, order=o, mode="constant") for a, o in zip(arrays, orders)]
    if rng.random() < 0.5:
        arrays = [np.flip(a, axis=1) for a in arrays]
    if rng.random() < 0.5:
        arrays = [np.rot90(a, k=-1, axes=(0, 1)) for a in arrays]

    channels = np.stack(arrays[:-1]).astype(np.float32)
    if rng.random() < 0.5:
        # PCA intensity perturbation scales the leading inter-modal component.
        voxels = channels.reshape(channels.shape[0], -1)
        covariance = np.cov(voxels)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        component = eigenvectors[:, np.argmax(eigenvalues)] * np.sqrt(max(eigenvalues.max(), 0))
        channels += float(rng.uniform(0.3, 0.8) * rng.choice([-1, 1])) * component[:, None, None, None]
    return channels, np.rint(arrays[-1]).astype(np.int16)
