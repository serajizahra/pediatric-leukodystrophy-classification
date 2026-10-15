from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..utils.files import write_json


def _require_imaging():
    try:
        import nibabel as nib
        from scipy.ndimage import zoom

        return nib, zoom
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError("nibabel and scipy are required for deep patch preparation.") from exc


def _require_torch():
    try:
        import torch

        return torch
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError("PyTorch is required to save deep-learning patch tensors.") from exc


@dataclass(frozen=True)
class DeepPatchConfig:
    slice_count: int = 32
    image_size: int = 224
    patch_depth: int = 32
    patch_height: int = 224
    patch_width: int = 224
    roi_ids: tuple[int, ...] = tuple(range(1, 41))


def _load_volume(path: str | Path) -> np.ndarray:
    nib, _ = _require_imaging()
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)


def _robust_zscore(volume: np.ndarray) -> np.ndarray:
    finite = np.isfinite(volume)
    if not finite.any():
        return np.zeros_like(volume, dtype=np.float32)
    values = volume[finite]
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-8:
        std = 1.0
    return np.nan_to_num((volume - mean) / std).astype(np.float32)


def _take_axial_slices(volume: np.ndarray, slice_count: int) -> np.ndarray:
    z_count = volume.shape[2]
    if z_count == slice_count:
        return volume
    indices = np.linspace(0, z_count - 1, slice_count).round().astype(int)
    return volume[:, :, indices]


def _resize_volume(volume: np.ndarray, output_shape: tuple[int, int, int], order: int) -> np.ndarray:
    _, zoom = _require_imaging()
    factors = tuple(out / inp for out, inp in zip(output_shape, volume.shape, strict=False))
    return zoom(volume, factors, order=order)


def _prepare_channel(volume: np.ndarray, config: DeepPatchConfig) -> np.ndarray:
    axial = _take_axial_slices(volume, config.slice_count)
    resized = _resize_volume(axial, (config.image_size, config.image_size, config.slice_count), order=1)
    return _robust_zscore(resized)


def _prepare_labels(labels: np.ndarray, config: DeepPatchConfig) -> np.ndarray:
    axial = _take_axial_slices(labels, config.slice_count)
    resized = _resize_volume(axial, (config.image_size, config.image_size, config.slice_count), order=0)
    return np.rint(resized).astype(np.int16)


def _bbox(mask: np.ndarray) -> tuple[slice, slice, slice] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    return tuple(slice(int(start), int(stop)) for start, stop in zip(mins, maxs, strict=False))


def _resize_patch(patch: np.ndarray, config: DeepPatchConfig) -> np.ndarray:
    channels, x_size, y_size, z_size = patch.shape
    resized_channels = []
    for channel in range(channels):
        resized = _resize_volume(
            patch[channel],
            (config.patch_height, config.patch_width, config.patch_depth),
            order=1,
        )
        resized_channels.append(resized)
    stacked = np.stack(resized_channels, axis=0)
    return np.transpose(stacked, (0, 3, 1, 2)).astype(np.float32)


def stack_multimodal_volume(
    t1_path: str | Path,
    t2_path: str | Path,
    flair_path: str | Path,
    config: DeepPatchConfig,
) -> np.ndarray:
    channels = [
        _prepare_channel(_load_volume(t1_path), config),
        _prepare_channel(_load_volume(t2_path), config),
        _prepare_channel(_load_volume(flair_path), config),
    ]
    return np.stack(channels, axis=0)


def extract_roi_patches(
    t1_path: str | Path,
    t2_path: str | Path,
    flair_path: str | Path,
    roi_mask_path: str | Path,
    config: DeepPatchConfig = DeepPatchConfig(),
    augment: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    volume = stack_multimodal_volume(t1_path, t2_path, flair_path, config)
    labels = _prepare_labels(_load_volume(roi_mask_path), config)
    if augment:
        from .augmentation import augment_full_volume

        volume, labels = augment_full_volume(volume, labels)
    patches = []
    roi_metadata = []
    for roi_id in config.roi_ids:
        mask = labels == roi_id
        box = _bbox(mask)
        if box is None:
            patch = np.zeros((3, config.patch_depth, config.patch_height, config.patch_width), dtype=np.float32)
            roi_metadata.append({"roi_id": roi_id, "present": False, "voxel_count": 0})
        else:
            masked = volume * mask[np.newaxis, :, :, :]
            cropped = masked[(slice(None),) + box]
            patch = _resize_patch(cropped, config)
            roi_metadata.append({"roi_id": roi_id, "present": True, "voxel_count": int(mask.sum())})
        patches.append(patch)
    return np.stack(patches, axis=0), {
        "shape": [len(config.roi_ids), 3, config.patch_depth, config.patch_height, config.patch_width],
        "roi_count": len(config.roi_ids),
        "roi_metadata": roi_metadata,
        "slice_count": config.slice_count,
        "image_size": config.image_size,
        "patch_size": [config.patch_depth, config.patch_height, config.patch_width],
    }


def save_deep_patch_tensor(
    output_path: str | Path,
    patches: np.ndarray,
    metadata: dict[str, object],
    patient_id: str,
    label: str | None = None,
    metadata_path: str | Path | None = None,
) -> tuple[Path, Path]:
    torch = _require_torch()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tensor = torch.from_numpy(patches.astype(np.float32))
    payload = {"patches": tensor, "patient_id": patient_id, "label": label, "metadata": metadata}
    torch.save(payload, output)
    meta_path = Path(metadata_path) if metadata_path else output.with_suffix(".json")
    write_json(
        meta_path,
        {
            "patient_id": patient_id,
            "label": label,
            **metadata,
            "tensor_path": str(output),
        },
    )
    return output, meta_path
