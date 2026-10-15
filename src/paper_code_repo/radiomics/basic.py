from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def _safe_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    return arr


def _entropy(values: np.ndarray, bins: int = 32) -> float:
    arr = _safe_array(values)
    if arr.size == 0:
        return 0.0
    hist, _ = np.histogram(arr, bins=bins, density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return 0.0
    return float(-np.sum(hist * np.log2(hist)))


def _skew(values: np.ndarray) -> float:
    arr = _safe_array(values)
    if arr.size < 3:
        return 0.0
    mean = float(arr.mean())
    std = float(arr.std())
    if std == 0:
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 3))


def _kurtosis(values: np.ndarray) -> float:
    arr = _safe_array(values)
    if arr.size < 4:
        return 0.0
    mean = float(arr.mean())
    std = float(arr.std())
    if std == 0:
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 4) - 3.0)


def extract_first_order_features(values: np.ndarray, prefix: str = "") -> dict[str, float]:
    arr = _safe_array(values)
    if arr.size == 0:
        arr = np.array([0.0], dtype=float)

    q05, q95 = np.percentile(arr, [5, 95])
    q25, q75 = np.percentile(arr, [25, 75])
    mad = float(np.median(np.abs(arr - np.median(arr))))
    energy = float(np.sum(arr ** 2))
    rel_energy = float(energy / arr.size)

    features = {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "var": float(arr.var()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "range": float(arr.max() - arr.min()),
        "mad": mad,
        "iqr": float(q75 - q25),
        "p05": float(q05),
        "p95": float(q95),
        "energy": energy,
        "relative_energy": rel_energy,
        "entropy": _entropy(arr),
        "skewness": _skew(arr),
        "kurtosis": _kurtosis(arr),
        "voxel_count": float(arr.size),
        "nonzero_fraction": float(np.count_nonzero(arr) / arr.size),
    }
    if prefix:
        return {f"{prefix}{name}": value for name, value in features.items()}
    return features


def extract_roi_statistics(image: np.ndarray, mask: np.ndarray, roi_name: str) -> dict[str, float]:
    roi_values = np.asarray(image)[np.asarray(mask) > 0]
    return extract_first_order_features(roi_values, prefix=f"{roi_name}__")


def build_feature_frame(rows: Iterable[dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))

