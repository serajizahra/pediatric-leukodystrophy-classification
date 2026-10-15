from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def _display_slice(data: np.ndarray, axis: int) -> np.ndarray:
    index = data.shape[axis] // 2
    image = np.take(data, index, axis=axis)
    return np.rot90(image)


def save_montage(image_path: str | Path, output_path: str | Path, title: str) -> Path:
    data = nib.load(str(image_path)).get_fdata(dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size:
        low, high = np.percentile(finite, [1, 99])
    else:
        low, high = 0.0, 1.0
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, label, array_axis in zip(axes, ("Sagittal", "Coronal", "Axial"), (0, 1, 2), strict=True):
        axis.imshow(_display_slice(data, array_axis), cmap="gray", vmin=low, vmax=high)
        axis.set_title(label)
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output

