from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .evaluation.metrics import expected_calibration_error, precision_recall_f1_specificity
from .radiomics.selection import build_feature_selector
from .utils.files import ensure_dir, write_dataframe, write_json


DEFAULT_FEATURE_SIZES = tuple(range(60, 321, 20))
DEFAULT_SELECTION_METHODS = ("mrmr", "relieff", "lasso")
DEFAULT_CLASSIFIERS = ("svm", "knn", "rf", "xgboost")


@dataclass(frozen=True)
class TraditionalMLConfig:
    features_csv: Path
    output_dir: Path
    label_column: str = "label"
    id_column: str | None = "patient_id"
    feature_sizes: tuple[int, ...] = DEFAULT_FEATURE_SIZES
    selection_methods: tuple[str, ...] = DEFAULT_SELECTION_METHODS
    classifiers: tuple[str, ...] = DEFAULT_CLASSIFIERS
    outer_folds: int = 5
    inner_folds: int = 5
    scoring: str = "accuracy"
    random_state: int = 42
    quick: bool = False
    n_jobs: int = 1
    mode: str = "radiomics"
    fold_assignments: Path | None = None
    fold_features_dir: Path | None = None


def _require_sklearn():
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
        from sklearn.svm import SVC

        return {
            "RandomForestClassifier": RandomForestClassifier,
            "SimpleImputer": SimpleImputer,
            "accuracy_score": accuracy_score,
            "GridSearchCV": GridSearchCV,
            "StratifiedGroupKFold": StratifiedGroupKFold,
            "KNeighborsClassifier": KNeighborsClassifier,
            "Pipeline": Pipeline,
            "LabelEncoder": LabelEncoder,
            "MinMaxScaler": MinMaxScaler,
            "StandardScaler": StandardScaler,
            "SVC": SVC,
        }
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "scikit-learn is required for traditional ML training. Install the project dependencies first."
        ) from exc


def _require_imblearn():
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline

        return {"SMOTE": SMOTE, "Pipeline": ImbPipeline}
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "imbalanced-learn is required for SMOTE within training folds."
        ) from exc


def _require_xgboost():
    try:
        from xgboost import XGBClassifier

        return XGBClassifier
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "xgboost is required for the XGBoost classifier. Install the project dependencies first."
        ) from exc


def _normalize_items(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(item.strip().lower() for item in items if item and item.strip())


def _safe_cv_splits(y: np.ndarray, requested: int, groups: np.ndarray | None = None) -> int:
    if groups is not None:
        group_labels = pd.DataFrame({"group": groups, "label": y}).drop_duplicates("group")
        y = group_labels["label"].to_numpy()
    _, counts = np.unique(y, return_counts=True)
    min_count = int(np.min(counts))
    folds = min(int(requested), min_count)
    if folds < 2:
        raise ValueError("At least two samples per class are required for stratified cross-validation.")
    return folds


def _read_feature_table(config: TraditionalMLConfig) -> tuple[pd.DataFrame, np.ndarray, list[str], pd.DataFrame]:
    frame = pd.read_csv(config.features_csv)
    if config.label_column not in frame.columns:
        raise ValueError(f"Label column '{config.label_column}' was not found in {config.features_csv}")

    excluded = {config.label_column}
    if config.id_column and config.id_column in frame.columns:
        excluded.add(config.id_column)

    numeric = frame.drop(columns=[col for col in excluded if col in frame.columns]).select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError("No numeric feature columns were found.")

    labels = frame[config.label_column].astype(str).to_numpy()
    return numeric, labels, numeric.columns.tolist(), frame


def _classifier_and_grid(name: str, random_state: int, quick: bool) -> tuple[Any, dict[str, list[Any]]]:
    deps = _require_sklearn()
    normalized = name.lower()
    if normalized == "svm":
        estimator = deps["SVC"](kernel="rbf", probability=True, random_state=random_state)
        grid = {
            "clf__C": [10] if quick else [0.1, 1, 10, 100],
            "clf__gamma": [0.01] if quick else [0.001, 0.01, 0.1, "scale"],
        }
    elif normalized == "knn":
        estimator = deps["KNeighborsClassifier"]()
        grid = {
            "clf__n_neighbors": [5] if quick else [3, 5, 7, 9],
            "clf__weights": ["distance"] if quick else ["uniform", "distance"],
            "clf__metric": ["euclidean"] if quick else ["euclidean", "manhattan"],
        }
    elif normalized == "rf":
        estimator = deps["RandomForestClassifier"](random_state=random_state)
        grid = {
            "clf__n_estimators": [50] if quick else [100, 200, 250, 300, 400],
            "clf__max_depth": [8] if quick else [None, 6, 8, 10],
        }
    elif normalized == "xgboost":
        XGBClassifier = _require_xgboost()
        estimator = XGBClassifier(
            random_state=random_state,
            eval_metric="mlogloss",
            tree_method="hist",
            objective="multi:softprob",
        )
        grid = {
            "clf__learning_rate": [0.1] if quick else [0.01, 0.1],
            "clf__max_depth": [5] if quick else [4, 5, 6],
            "clf__n_estimators": [50] if quick else [200, 300, 350],
        }
    else:
        raise ValueError(f"Unknown classifier: {name}")
    return estimator, grid


def _smote_neighbors(y: np.ndarray) -> int:
    _, counts = np.unique(y, return_counts=True)
    return max(1, min(5, int(np.min(counts)) - 1))


def _smote_neighbors_for_cv(y: np.ndarray, cv_splits: int) -> int:
    _, counts = np.unique(y, return_counts=True)
    min_count = int(np.min(counts))
    min_inner_train_count = min_count - int(np.ceil(min_count / cv_splits))
    return max(1, min(5, min_inner_train_count - 1))


def _build_pipeline(
    method: str,
    feature_size: int,
    classifier: str,
    config: TraditionalMLConfig,
    smote_k_neighbors: int = 5,
    feature_names: list[str] | None = None,
):
    deps = _require_sklearn()
    imblearn = _require_imblearn()
    estimator, grid = _classifier_and_grid(classifier, config.random_state, config.quick)
    if config.mode == "fused":
        from .hybrid.selection import FusedMRMRSelector

        if feature_names is None:
            raise ValueError("Fused selection requires feature names")
        radiomic_indices = tuple(i for i, name in enumerate(feature_names) if name.startswith("radiomics__"))
        deep_indices = tuple(i for i, name in enumerate(feature_names) if name.startswith("deep__"))
        if len(radiomic_indices) < 260 or len(deep_indices) < feature_size:
            raise ValueError("Fused mode requires at least 260 radiomic and the requested number of deep features")
        selector = FusedMRMRSelector(radiomic_indices, deep_indices, feature_size, 260, config.random_state)
    else:
        selector = build_feature_selector(method, feature_size, random_state=config.random_state, fast=config.quick)
    steps = [
        ("imputer", deps["SimpleImputer"](strategy="median", keep_empty_features=True)),
        ("smote", imblearn["SMOTE"](k_neighbors=smote_k_neighbors, random_state=config.random_state)),
        ("scaler", deps["MinMaxScaler"]()),
        ("selector", selector),
    ]
    if config.mode == "fused":
        steps.append(("fusion_zscore", deps["StandardScaler"]()))
    steps.append(("clf", estimator))
    return (
        imblearn["Pipeline"](steps=steps),
        grid,
    )


def _jsonable_params(params: dict[str, Any]) -> str:
    return json.dumps({key: _to_python(value) for key, value in params.items()}, sort_keys=True)


def _to_python(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _selected_features_from_pipeline(pipeline, feature_names: list[str]) -> list[str]:
    selector = pipeline.named_steps["selector"]
    return list(selector.get_feature_names_out(feature_names))


def _refit_best_row(
    row: dict[str, Any],
    config: TraditionalMLConfig,
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    cv,
    full_inner_splits: int,
    groups: np.ndarray,
):
    best_config = TraditionalMLConfig(
        features_csv=config.features_csv,
        output_dir=config.output_dir,
        label_column=config.label_column,
        id_column=config.id_column,
        feature_sizes=(int(row["requested_features"]),),
        selection_methods=(str(row["feature_selection"]),),
        classifiers=(str(row["classifier"]),),
        outer_folds=config.outer_folds,
        inner_folds=config.inner_folds,
        scoring=config.scoring,
        random_state=config.random_state,
        quick=config.quick,
        n_jobs=config.n_jobs,
        mode=config.mode,
        fold_assignments=config.fold_assignments,
        fold_features_dir=config.fold_features_dir,
    )
    final_pipeline, param_grid = _build_pipeline(
        best_config.selection_methods[0],
        best_config.feature_sizes[0],
        best_config.classifiers[0],
        best_config,
        smote_k_neighbors=_smote_neighbors_for_cv(y, full_inner_splits),
        feature_names=feature_names,
    )
    deps = _require_sklearn()
    final_search = deps["GridSearchCV"](
        estimator=final_pipeline,
        param_grid=param_grid,
        scoring=config.scoring,
        cv=cv,
        n_jobs=config.n_jobs,
        refit=True,
        error_score="raise",
    )
    final_search.fit(x, y, groups=groups)
    selected_features = _selected_features_from_pipeline(final_search.best_estimator_, feature_names)
    return final_search, selected_features


def _refit_selector_for_method_size(
    method: str,
    feature_size: int,
    config: TraditionalMLConfig,
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> list[str]:
    deps = _require_sklearn()
    imblearn = _require_imblearn()
    imputer = deps["SimpleImputer"](strategy="median", keep_empty_features=True)
    smote = imblearn["SMOTE"](k_neighbors=_smote_neighbors(y), random_state=config.random_state)
    scaler = deps["MinMaxScaler"]()
    if config.mode == "fused":
        from .hybrid.selection import FusedMRMRSelector

        radiomic_indices = tuple(i for i, name in enumerate(feature_names) if name.startswith("radiomics__"))
        deep_indices = tuple(i for i, name in enumerate(feature_names) if name.startswith("deep__"))
        selector = FusedMRMRSelector(radiomic_indices, deep_indices, feature_size, 260, config.random_state)
    else:
        selector = build_feature_selector(method, feature_size, random_state=config.random_state, fast=config.quick)
    x_imputed = imputer.fit_transform(x)
    x_resampled, y_resampled = smote.fit_resample(x_imputed, y)
    x_scaled = scaler.fit_transform(x_resampled)
    selector.fit(x_scaled, y_resampled)
    return list(selector.get_feature_names_out(feature_names))


def run_traditional_ml_experiment(config: TraditionalMLConfig) -> dict[str, Path]:
    deps = _require_sklearn()
    x_frame, labels, feature_names, original_frame = _read_feature_table(config)
    label_encoder = deps["LabelEncoder"]()
    y = label_encoder.fit_transform(labels)
    x = x_frame.to_numpy(dtype=float)
    if not config.id_column or config.id_column not in original_frame:
        raise ValueError("A patient ID column is required to keep repeat examinations in one fold")
    groups = original_frame[config.id_column].astype(str).to_numpy()
    if original_frame.groupby(config.id_column)[config.label_column].nunique().gt(1).any():
        raise ValueError("A patient cannot have conflicting class labels")
    output_dir = ensure_dir(config.output_dir)

    outer_splits = _safe_cv_splits(y, config.outer_folds, groups)
    if config.fold_features_dir and not config.fold_assignments:
        raise ValueError("Fold-specific feature tables require patient fold assignments")
    if config.mode in {"deep", "fused"} and not (config.fold_assignments and config.fold_features_dir):
        raise ValueError("Hybrid evaluation requires fold-specific CNN features and patient fold assignments")
    if config.fold_assignments:
        assignments = pd.read_csv(config.fold_assignments)
        if not {config.id_column, "fold"}.issubset(assignments.columns):
            raise ValueError("Fold assignments need patient ID and fold columns")
        assignments[config.id_column] = assignments[config.id_column].astype(str)
        if assignments.groupby(config.id_column)["fold"].nunique().gt(1).any():
            raise ValueError("A patient appears in multiple outer folds")
        fold_map = assignments.drop_duplicates(config.id_column).set_index(config.id_column)["fold"]
        assigned = original_frame[config.id_column].map(fold_map)
        if assigned.isna().any():
            raise ValueError("Every feature row must have a patient-level fold assignment")
        fold_values = sorted(assigned.unique())
        if len(fold_values) != config.outer_folds:
            raise ValueError("Expected the requested number of outer folds")
        outer_data = []
        for fold_number, fold_value in enumerate(fold_values, start=1):
            if config.fold_features_dir:
                fold_path = Path(config.fold_features_dir) / f"fold_{fold_value}.csv"
                fold_frame = pd.read_csv(fold_path)
                fold_frame[config.id_column] = fold_frame[config.id_column].astype(str)
                key = "exam_id" if "exam_id" in original_frame.columns else config.id_column
                if key not in fold_frame.columns or fold_frame[key].duplicated().any():
                    raise ValueError(f"Fold-specific table requires unique {key} values: {fold_path}")
                aligned = fold_frame.set_index(key).reindex(original_frame[key].astype(str))
                if aligned[config.label_column].isna().any() or not np.array_equal(aligned[config.label_column].astype(str), labels):
                    raise ValueError(f"Fold-specific labels or examination IDs do not match: {fold_path}")
                if key != config.id_column and not np.array_equal(aligned[config.id_column].astype(str), groups):
                    raise ValueError(f"Fold-specific patient IDs do not match: {fold_path}")
                if set(aligned.select_dtypes(include=[np.number]).columns) != set(feature_names):
                    raise ValueError(f"Fold-specific feature columns do not match: {fold_path}")
                x_fold = aligned[feature_names].to_numpy(dtype=float)
            else:
                x_fold = x
            test_idx = np.flatnonzero(assigned.to_numpy() == fold_value)
            train_idx = np.flatnonzero(assigned.to_numpy() != fold_value)
            outer_data.append((fold_number, train_idx, test_idx, x_fold))
        outer_splits = len(outer_data)
    else:
        outer_cv = deps["StratifiedGroupKFold"](n_splits=outer_splits, shuffle=True, random_state=config.random_state)
        outer_data = [
            (fold, train_idx, test_idx, x)
            for fold, (train_idx, test_idx) in enumerate(outer_cv.split(x, y, groups), start=1)
        ]
    full_inner_splits = _safe_cv_splits(y, config.inner_folds, groups)

    fold_records: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    total_jobs = len(config.selection_methods) * len(config.feature_sizes) * len(config.classifiers)
    completed = 0
    for method in config.selection_methods:
        for feature_size in config.feature_sizes:
            for classifier in config.classifiers:
                completed += 1
                print(
                    f"[{completed}/{total_jobs}] selection={method} "
                    f"features={feature_size} classifier={classifier}"
                )
                for fold_idx, train_idx, test_idx, x_fold in outer_data:
                    train_inner_splits = _safe_cv_splits(y[train_idx], config.inner_folds, groups[train_idx])
                    inner_cv = deps["StratifiedGroupKFold"](
                        n_splits=train_inner_splits,
                        shuffle=True,
                        random_state=config.random_state,
                    )
                    pipeline, param_grid = _build_pipeline(
                        method,
                        feature_size,
                        classifier,
                        config,
                        smote_k_neighbors=_smote_neighbors_for_cv(y[train_idx], train_inner_splits),
                        feature_names=feature_names,
                    )
                    search = deps["GridSearchCV"](
                        estimator=pipeline,
                        param_grid=param_grid,
                        scoring=config.scoring,
                        cv=inner_cv,
                        n_jobs=config.n_jobs,
                        refit=True,
                        error_score="raise",
                    )
                    search.fit(x_fold[train_idx], y[train_idx], groups=groups[train_idx])
                    y_pred = search.predict(x_fold[test_idx])
                    probability = search.predict_proba(x_fold[test_idx])
                    patient_votes = pd.DataFrame({"patient_id": groups[test_idx], "truth": y[test_idx], "prediction": y_pred})
                    for class_index in range(probability.shape[1]):
                        patient_votes[f"probability_{class_index}"] = probability[:, class_index]
                    patient_votes = patient_votes.groupby("patient_id", as_index=False).agg(
                        **{"truth": ("truth", "first"),
                           "prediction": ("prediction", lambda v: int(v.value_counts().index[0])),
                           **{f"probability_{i}": (f"probability_{i}", "mean") for i in range(probability.shape[1])}}
                    )
                    report = precision_recall_f1_specificity(
                        patient_votes["truth"], patient_votes["prediction"], labels=list(range(len(label_encoder.classes_))),
                    )
                    for vote in patient_votes.itertuples(index=False):
                        prediction_records.append({"feature_selection": method, "requested_features": feature_size,
                                                   "classifier": classifier, "fold": fold_idx,
                                                   **vote._asdict()})
                    selected = _selected_features_from_pipeline(search.best_estimator_, feature_names)
                    fold_records.append(
                        {
                            "feature_selection": method,
                            "requested_features": feature_size,
                            "selected_features": len(selected),
                            "classifier": classifier,
                            "fold": fold_idx,
                            "accuracy": report.accuracy,
                            "precision": report.precision,
                            "recall": report.recall,
                            "f1": report.f1,
                            "specificity": report.specificity,
                            "ece": expected_calibration_error(
                                patient_votes["truth"].to_numpy(),
                                patient_votes[[f"probability_{i}" for i in range(probability.shape[1])]].to_numpy(),
                            ),
                            "inner_best_score": float(search.best_score_),
                            "best_params": _jsonable_params(search.best_params_),
                        }
                    )

    folds = pd.DataFrame(fold_records)
    aggregate = (
        folds.groupby(["feature_selection", "requested_features", "classifier"], as_index=False)
        .agg(
            selected_features=("selected_features", "median"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_precision=("precision", "mean"),
            std_precision=("precision", "std"),
            mean_recall=("recall", "mean"),
            std_recall=("recall", "std"),
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            mean_specificity=("specificity", "mean"),
            std_specificity=("specificity", "std"),
            mean_ece=("ece", "mean"),
            mean_inner_best_score=("inner_best_score", "mean"),
        )
        .sort_values(["mean_accuracy", "mean_f1", "mean_specificity"], ascending=False)
        .reset_index(drop=True)
    )
    selection_size_summary = (
        aggregate.groupby(["feature_selection", "requested_features"], as_index=False)
        .agg(
            mean_accuracy_across_classifiers=("mean_accuracy", "mean"),
            std_accuracy_across_classifiers=("mean_accuracy", "std"),
            mean_f1_across_classifiers=("mean_f1", "mean"),
            mean_specificity_across_classifiers=("mean_specificity", "mean"),
            classifier_count=("classifier", "nunique"),
        )
        .sort_values(
            ["feature_selection", "mean_accuracy_across_classifiers", "mean_f1_across_classifiers"],
            ascending=[True, False, False],
        )
        .reset_index(drop=True)
    )
    best_size_by_selection = (
        selection_size_summary.groupby("feature_selection", as_index=False)
        .head(1)
        .sort_values("mean_accuracy_across_classifiers", ascending=False)
        .reset_index(drop=True)
    )

    full_inner_cv = deps["StratifiedGroupKFold"](
        n_splits=full_inner_splits,
        shuffle=True,
        random_state=config.random_state,
    )
    best_row = aggregate.iloc[0].to_dict()
    final_search, selected_features = _refit_best_row(
        best_row,
        config,
        x,
        y,
        feature_names,
        full_inner_cv,
        full_inner_splits,
        groups,
    )
    selected_frame = pd.DataFrame(
        {
            "rank": np.arange(1, len(selected_features) + 1),
            "feature_name": selected_features,
        }
    )
    best_by_classifier = (
        aggregate.sort_values(["classifier", "mean_accuracy", "mean_f1", "mean_specificity"], ascending=[True, False, False, False])
        .groupby("classifier", as_index=False)
        .head(1)
        .sort_values(["mean_accuracy", "mean_f1", "mean_specificity"], ascending=False)
        .reset_index(drop=True)
    )
    per_classifier_metadata: dict[str, Any] = {}
    per_classifier_selected_rows: list[dict[str, Any]] = []
    for classifier_row in best_by_classifier.to_dict(orient="records"):
        classifier_name = str(classifier_row["classifier"])
        classifier_search, classifier_features = _refit_best_row(
            classifier_row,
            config,
            x,
            y,
            feature_names,
            full_inner_cv,
            full_inner_splits,
            groups,
        )
        per_classifier_metadata[classifier_name] = {
            "best_outer_cv_summary": {key: _to_python(value) for key, value in classifier_row.items()},
            "best_refit_params": json.loads(_jsonable_params(classifier_search.best_params_)),
            "best_refit_inner_score": float(classifier_search.best_score_),
            "selected_feature_count": len(classifier_features),
        }
        for rank, feature_name in enumerate(classifier_features, start=1):
            per_classifier_selected_rows.append(
                {
                    "classifier": classifier_name,
                    "rank": rank,
                    "feature_name": feature_name,
                }
            )

    per_method_selected_rows: list[dict[str, Any]] = []
    for method_row in best_size_by_selection.to_dict(orient="records"):
        method = str(method_row["feature_selection"])
        feature_size = int(method_row["requested_features"])
        selected_for_method = _refit_selector_for_method_size(method, feature_size, config, x, y, feature_names)
        for rank, feature_name in enumerate(selected_for_method, start=1):
            per_method_selected_rows.append(
                {
                    "feature_selection": method,
                    "selected_size": feature_size,
                    "rank": rank,
                    "feature_name": feature_name,
                }
            )

    folds_path = output_dir / "traditional_fold_results.csv"
    predictions_path = output_dir / "traditional_patient_predictions.csv"
    aggregate_path = output_dir / "traditional_results.csv"
    selection_size_summary_path = output_dir / "traditional_feature_size_summary.csv"
    best_size_by_selection_path = output_dir / "traditional_best_size_by_selection_method.csv"
    selected_path = output_dir / "traditional_best_selected_features.csv"
    selected_by_method_path = output_dir / "traditional_best_selected_features_by_method.csv"
    best_by_classifier_path = output_dir / "traditional_best_by_classifier.csv"
    best_by_classifier_features_path = output_dir / "traditional_best_by_classifier_selected_features.csv"
    metadata_path = output_dir / "traditional_best_model.json"
    per_classifier_metadata_path = output_dir / "traditional_best_by_classifier.json"

    write_dataframe(folds_path, folds)
    write_dataframe(predictions_path, pd.DataFrame(prediction_records))
    write_dataframe(aggregate_path, aggregate)
    write_dataframe(selection_size_summary_path, selection_size_summary)
    write_dataframe(best_size_by_selection_path, best_size_by_selection)
    write_dataframe(selected_path, selected_frame)
    write_dataframe(selected_by_method_path, pd.DataFrame(per_method_selected_rows))
    write_dataframe(best_by_classifier_path, best_by_classifier)
    write_dataframe(best_by_classifier_features_path, pd.DataFrame(per_classifier_selected_rows))
    write_json(
        metadata_path,
        {
            "features_csv": str(config.features_csv),
            "n_samples": int(len(original_frame)),
            "n_input_features": int(len(feature_names)),
            "label_column": config.label_column,
            "id_column": config.id_column,
            "labels": dict(zip(label_encoder.classes_.tolist(), label_encoder.transform(label_encoder.classes_).tolist(), strict=False)),
            "feature_selection_methods": list(config.selection_methods),
            "feature_sizes": list(config.feature_sizes),
            "classifiers": list(config.classifiers),
            "outer_folds": outer_splits,
            "inner_folds": full_inner_splits,
            "scoring": config.scoring,
            "smote": "SMOTE applied only inside training folds; k_neighbors is capped at 5 and reduced automatically for very small folds.",
            "normalization": "MinMaxScaler fitted only inside training folds; fused mode additionally z-scores selected features",
            "grouping": "StratifiedGroupKFold on patient_id; repeated examinations majority-voted per patient",
            "mode": config.mode,
            "fold_assignments": str(config.fold_assignments) if config.fold_assignments else None,
            "fold_features_dir": str(config.fold_features_dir) if config.fold_features_dir else None,
            "best_outer_cv_summary": {key: _to_python(value) for key, value in best_row.items()},
            "feature_size_selection_rule": (
                "For each feature-selection method, average the mean outer-CV accuracy "
                "of the grid-searched SVM, KNN, RF, and XGBoost models at each feature size; "
                "select the feature size with the highest average accuracy."
            ),
            "best_size_by_selection_method": [
                {key: _to_python(value) for key, value in row.items()}
                for row in best_size_by_selection.to_dict(orient="records")
            ],
            "best_refit_params": json.loads(_jsonable_params(final_search.best_params_)),
            "best_refit_inner_score": float(final_search.best_score_),
            "selected_feature_count": len(selected_features),
        },
    )
    write_json(per_classifier_metadata_path, per_classifier_metadata)
    return {
        "fold_results": folds_path,
        "patient_predictions": predictions_path,
        "aggregate_results": aggregate_path,
        "feature_size_summary": selection_size_summary_path,
        "best_size_by_selection_method": best_size_by_selection_path,
        "selected_features": selected_path,
        "selected_features_by_method": selected_by_method_path,
        "best_by_classifier": best_by_classifier_path,
        "best_by_classifier_selected_features": best_by_classifier_features_path,
        "metadata": metadata_path,
        "best_by_classifier_metadata": per_classifier_metadata_path,
    }


def fit_classical_model(model_name: str, x_train: np.ndarray, y_train: np.ndarray, params: dict[str, Any] | None = None):
    """Backward-compatible direct model fitting helper."""

    deps = _require_sklearn()
    model_name = model_name.lower()
    params = params or {}

    if model_name == "svm":
        model = deps["Pipeline"](
            [
                ("scaler", deps["MinMaxScaler"]()),
                ("clf", deps["SVC"](kernel="rbf", probability=True, **params)),
            ]
        )
    elif model_name == "knn":
        model = deps["Pipeline"](
            [
                ("scaler", deps["MinMaxScaler"]()),
                ("clf", deps["KNeighborsClassifier"](**params)),
            ]
        )
    elif model_name == "rf":
        model = deps["RandomForestClassifier"](random_state=42, **params)
    elif model_name == "xgboost":
        XGBClassifier = _require_xgboost()
        model = XGBClassifier(
            random_state=42,
            eval_metric="mlogloss",
            tree_method="hist",
            **params,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(x_train, y_train)
    return model


def predict(model, x: np.ndarray) -> np.ndarray:
    return model.predict(x)


def predict_proba(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)
    raise AttributeError("The selected model does not support predict_proba.")
