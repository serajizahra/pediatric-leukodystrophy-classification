from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..traditional_ml import fit_classical_model, predict


def train_hybrid_classifier(
    model_name: str,
    radiomics_features: pd.DataFrame,
    deep_features: pd.DataFrame,
    labels: np.ndarray,
    params: dict[str, Any] | None = None,
):
    from .fusion import concatenate_features

    fused = concatenate_features(radiomics_features, deep_features)
    model = fit_classical_model(model_name, fused.to_numpy(), labels, params=params)
    return model, fused


def infer_hybrid_classifier(model, radiomics_features: pd.DataFrame, deep_features: pd.DataFrame) -> np.ndarray:
    from .fusion import concatenate_features

    fused = concatenate_features(radiomics_features, deep_features)
    return predict(model, fused.to_numpy())

