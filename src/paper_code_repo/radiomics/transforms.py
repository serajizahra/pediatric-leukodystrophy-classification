from __future__ import annotations

import numpy as np
import pywt
from scipy.fft import dctn, idctn

from .manual_config import HandcraftedFeatureConfig


def dct2(image: np.ndarray) -> np.ndarray:
    return dctn(np.asarray(image, dtype=np.float64), type=2, norm="ortho")


def idct2(coefficients: np.ndarray) -> np.ndarray:
    return idctn(np.asarray(coefficients, dtype=np.float64), type=2, norm="ortho")


def dcst_band_widths(length: int) -> np.ndarray:
    if length <= 1:
        return np.asarray([1], dtype=int)
    widths = 2 ** np.arange(int(np.log2(length)), dtype=int)
    covered = int(widths.sum())
    if covered < length:
        widths = np.append(widths, length - covered)
    return widths.astype(int)


def dcst1(data: np.ndarray) -> np.ndarray:
    out = dctn(np.asarray(data, dtype=np.float64), axes=(0,), type=2, norm="ortho")
    cursor = 0
    for width in dcst_band_widths(out.shape[0]):
        if width != 1:
            out[cursor : cursor + width] = idctn(out[cursor : cursor + width], axes=(0,), type=2, norm="ortho")
        cursor += int(width)
    return out


def idcst1(coefficients: np.ndarray) -> np.ndarray:
    out = np.asarray(coefficients, dtype=np.float64).copy()
    cursor = 0
    for width in dcst_band_widths(out.shape[0]):
        if width != 1:
            out[cursor : cursor + width] = dctn(out[cursor : cursor + width], axes=(0,), type=2, norm="ortho")
        cursor += int(width)
    return idctn(out, axes=(0,), type=2, norm="ortho")


def dcst2(image: np.ndarray) -> np.ndarray:
    return dcst1(dcst1(np.asarray(image, dtype=np.float64)).T).T


def idcst2(coefficients: np.ndarray) -> np.ndarray:
    return idcst1(idcst1(np.asarray(coefficients, dtype=np.float64).T).T)


def ranked_frequency_masks(shape: tuple[int, int], config: HandcraftedFeatureConfig | None = None) -> dict[str, np.ndarray]:
    config = config or HandcraftedFeatureConfig()
    total = int(shape[0] * shape[1])
    lf_count = int(round(total * config.frequency_band_fractions[0]))
    mf_count = int(round(total * (config.frequency_band_fractions[1] - config.frequency_band_fractions[0])))
    rows, cols = np.indices(shape)
    order = np.lexsort((cols.ravel(), rows.ravel(), (rows + cols).ravel()))
    flat_lf = np.zeros(total, dtype=bool)
    flat_mf = np.zeros(total, dtype=bool)
    flat_hf = np.zeros(total, dtype=bool)
    flat_lf[order[:lf_count]] = True
    flat_mf[order[lf_count : lf_count + mf_count]] = True
    flat_hf[order[lf_count + mf_count :]] = True
    return {
        "LF": flat_lf.reshape(shape),
        "MF": flat_mf.reshape(shape),
        "HF": flat_hf.reshape(shape),
    }


def dct_spatial_subbands(image: np.ndarray, config: HandcraftedFeatureConfig | None = None) -> dict[str, np.ndarray]:
    coeffs = dct2(image)
    return {
        band: idct2(np.where(mask, coeffs, 0.0))
        for band, mask in ranked_frequency_masks(coeffs.shape, config).items()
    }


def dcst_spatial_subbands(image: np.ndarray, config: HandcraftedFeatureConfig | None = None) -> dict[str, np.ndarray]:
    coeffs = dcst2(image)
    return {
        band: idcst2(np.where(mask, coeffs, 0.0))
        for band, mask in ranked_frequency_masks(coeffs.shape, config).items()
    }


def swt_spatial_subbands(image: np.ndarray, config: HandcraftedFeatureConfig | None = None) -> dict[str, np.ndarray]:
    config = config or HandcraftedFeatureConfig()
    arr = np.asarray(image, dtype=np.float64)
    pad_y = arr.shape[0] % 2
    pad_x = arr.shape[1] % 2
    padded = np.pad(arr, ((0, pad_y), (0, pad_x)), mode="edge") if pad_y or pad_x else arr
    coeffs = pywt.swt2(padded, wavelet=config.wavelet, level=1, trim_approx=False, norm=True)
    c_a, (c_h, c_v, c_d) = coeffs[0]
    zeros = np.zeros_like(c_a)
    isolated = {
        "LL": [(c_a, (zeros, zeros, zeros))],
        "LH": [(zeros, (c_h, zeros, zeros))],
        "HL": [(zeros, (zeros, c_v, zeros))],
        "HH": [(zeros, (zeros, zeros, c_d))],
    }
    subbands = {name: pywt.iswt2(value, wavelet=config.wavelet, norm=True) for name, value in isolated.items()}
    if pad_y or pad_x:
        subbands = {name: value[: arr.shape[0], : arr.shape[1]] for name, value in subbands.items()}
    return subbands
