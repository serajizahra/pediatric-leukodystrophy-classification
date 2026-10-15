from __future__ import annotations

import math
import re
from collections.abc import Iterable

import numpy as np


STATISTIC_NAMES = (
    "mean",
    "std",
    "variance",
    "mad",
    "energy",
    "relative_energy",
    "p05",
    "p95",
    "median",
    "min",
    "max",
)

SLICE_AGGREGATORS = ("mean", "std", "min", "max")


def feature_slug(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "value"


def finite_float(value: object) -> float:
    try:
        result = float(value)
    except Exception:
        return 0.0
    return result if math.isfinite(result) else 0.0


def safe_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def entropy_from_values(values: np.ndarray, bins: int = 64) -> float:
    arr = safe_values(values)
    if arr.size == 0:
        return 0.0
    hist, _ = np.histogram(arr, bins=bins)
    total = hist.sum()
    if total <= 0:
        return 0.0
    prob = hist.astype(np.float64) / float(total)
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def coefficient_statistics(values: np.ndarray, total_energy: float | None = None) -> dict[str, float]:
    arr = safe_values(values)
    if arr.size == 0:
        return {name: 0.0 for name in STATISTIC_NAMES}
    median = float(np.median(arr))
    energy = float(np.sum(arr**2))
    denom = float(total_energy) if total_energy and total_energy > 0 else energy
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "variance": float(np.var(arr)),
        "mad": float(np.mean(np.abs(arr - np.mean(arr)))),
        "energy": energy,
        "relative_energy": float(energy / denom) if denom > 0 else 0.0,
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "median": median,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def aggregate_slice_values(values: Iterable[float]) -> dict[str, float]:
    arr = safe_values(np.asarray(list(values), dtype=np.float64))
    if arr.size == 0:
        return {name: 0.0 for name in SLICE_AGGREGATORS}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def axial_slices(volume: np.ndarray, labels: np.ndarray, axis: int = 2, max_slices: int | None = None):
    n_slices = volume.shape[axis]
    selected = range(n_slices)
    if max_slices is not None and max_slices > 0 and max_slices < n_slices:
        indices = np.linspace(0, n_slices - 1, num=max_slices, dtype=int)
        selected = sorted(set(int(index) for index in indices))
    for index in selected:
        image_slice = np.take(volume, index, axis=axis)
        label_slice = np.take(labels, index, axis=axis)
        yield index, image_slice, label_slice


def roi_ids_from_labels(labels: np.ndarray, requested: Iterable[int] | None = None) -> list[int]:
    if requested is not None:
        return [int(value) for value in requested]
    ids = np.unique(np.rint(labels).astype(np.int32))
    return [int(value) for value in ids if 1 <= int(value) <= 40]
