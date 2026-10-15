from pathlib import Path

import numpy as np
import pandas as pd

from paper_code_repo.radiomics.selection import LassoSelector, MRMRSelector, ReliefFSelector
from paper_code_repo.traditional_ml import TraditionalMLConfig, run_traditional_ml_experiment


def _synthetic_features(n_samples: int = 30, n_features: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    labels = np.repeat(["ALD", "CIM", "MLD"], n_samples // 3)
    x = rng.normal(size=(len(labels), n_features))
    for class_idx in range(3):
        x[labels == labels[class_idx * (n_samples // 3)], class_idx : class_idx + 4] += 2.0
    frame = pd.DataFrame(x, columns=[f"feature_{idx:03d}" for idx in range(n_features)])
    frame.insert(0, "label", labels)
    frame.insert(0, "patient_id", [f"P{idx:03d}" for idx in range(len(labels))])
    return frame


def test_feature_selectors_return_requested_shape():
    frame = _synthetic_features()
    x = frame.drop(columns=["patient_id", "label"]).to_numpy()
    y = pd.factorize(frame["label"])[0]
    for selector in (
        MRMRSelector(n_features_to_select=6),
        ReliefFSelector(n_features_to_select=6, n_neighbors=2),
        LassoSelector(n_features_to_select=6, max_iter=1000),
    ):
        transformed = selector.fit(x, y).transform(x)
        assert transformed.shape == (len(frame), 6)
        assert len(selector.selected_indices_) == 6


def test_traditional_ml_experiment_validation(tmp_path: Path):
    csv_path = tmp_path / "features.csv"
    out_dir = tmp_path / "traditional"
    _synthetic_features().to_csv(csv_path, index=False)

    outputs = run_traditional_ml_experiment(
        TraditionalMLConfig(
            features_csv=csv_path,
            output_dir=out_dir,
            label_column="label",
            id_column="patient_id",
            feature_sizes=(6,),
            selection_methods=("mrmr",),
            classifiers=("svm",),
            outer_folds=3,
            inner_folds=3,
            quick=True,
        )
    )

    for path in outputs.values():
        assert path.exists()
    results = pd.read_csv(outputs["aggregate_results"])
    assert len(results) == 1
    assert 0.0 <= float(results.loc[0, "mean_accuracy"]) <= 1.0
