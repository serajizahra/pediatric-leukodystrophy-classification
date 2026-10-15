from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import warnings

import numpy as np
import pandas as pd


def _require_sklearn():
    try:
        from sklearn.base import BaseEstimator, TransformerMixin
        from sklearn.feature_selection import SelectFromModel, f_classif, mutual_info_classif
        from sklearn.linear_model import LogisticRegression
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.utils.validation import check_is_fitted

        return {
            "BaseEstimator": BaseEstimator,
            "TransformerMixin": TransformerMixin,
            "SelectFromModel": SelectFromModel,
            "f_classif": f_classif,
            "mutual_info_classif": mutual_info_classif,
            "LogisticRegression": LogisticRegression,
            "ConvergenceWarning": ConvergenceWarning,
            "check_is_fitted": check_is_fitted,
        }
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "scikit-learn is required for feature selection. Install the project dependencies first."
        ) from exc


_deps = _require_sklearn()
BaseEstimator = _deps["BaseEstimator"]
TransformerMixin = _deps["TransformerMixin"]


@dataclass(frozen=True)
class SelectionSummary:
    method: str
    requested_features: int
    selected_features: int
    selected_names: list[str]


def variance_filter(frame: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    variances = frame.var(axis=0, numeric_only=True)
    keep = variances[variances > threshold].index.tolist()
    return frame.loc[:, keep]


def select_top_correlated(frame: pd.DataFrame, target: pd.Series, top_k: int = 50) -> pd.DataFrame:
    corr = frame.apply(lambda col: col.corr(target))
    keep = corr.abs().sort_values(ascending=False).head(top_k).index.tolist()
    return frame.loc[:, keep]


def _clean_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_k(requested: int, n_features: int) -> int:
    if requested < 1:
        raise ValueError("n_features_to_select must be >= 1")
    return int(min(requested, n_features))


class ReliefFSelector(BaseEstimator, TransformerMixin):
    """Deterministic multiclass ReliefF feature selector.

    The implementation is intentionally dependency-light so the repository can
    run without a specialized ReliefF package. It ranks features by comparing
    each sample with nearest same-class hits and nearest different-class misses.
    """

    def __init__(self, n_features_to_select: int = 260, n_neighbors: int = 5):
        self.n_features_to_select = n_features_to_select
        self.n_neighbors = n_neighbors

    def fit(self, x: np.ndarray, y: Iterable[int]):
        x = _clean_array(x)
        y = np.asarray(list(y))
        n_samples, n_features = x.shape
        k_features = _safe_k(self.n_features_to_select, n_features)
        k_neighbors = max(1, int(self.n_neighbors))

        mins = np.min(x, axis=0)
        ranges = np.ptp(x, axis=0)
        ranges[ranges == 0] = 1.0
        x_norm = (x - mins) / ranges

        classes, counts = np.unique(y, return_counts=True)
        priors = {label: count / n_samples for label, count in zip(classes, counts, strict=False)}
        weights = np.zeros(n_features, dtype=float)

        for row_idx in range(n_samples):
            row = x_norm[row_idx]
            row_label = y[row_idx]
            distances = np.sum(np.abs(x_norm - row), axis=1)
            distances[row_idx] = np.inf

            hit_idx = np.where(y == row_label)[0]
            hit_idx = hit_idx[hit_idx != row_idx]
            if len(hit_idx):
                nearest_hits = hit_idx[np.argsort(distances[hit_idx])[:k_neighbors]]
                weights -= np.mean(np.abs(row - x_norm[nearest_hits]), axis=0) / n_samples

            row_prior = priors[row_label]
            miss_denominator = max(1e-12, 1.0 - row_prior)
            for other_label in classes:
                if other_label == row_label:
                    continue
                miss_idx = np.where(y == other_label)[0]
                if not len(miss_idx):
                    continue
                nearest_misses = miss_idx[np.argsort(distances[miss_idx])[:k_neighbors]]
                class_weight = priors[other_label] / miss_denominator
                weights += class_weight * np.mean(np.abs(row - x_norm[nearest_misses]), axis=0) / n_samples

        order = np.argsort(weights)[::-1]
        self.scores_ = weights
        self.selected_indices_ = order[:k_features]
        self.n_features_in_ = n_features
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        _deps["check_is_fitted"](self, "selected_indices_")
        return _clean_array(x)[:, self.selected_indices_]

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        _deps["check_is_fitted"](self, "selected_indices_")
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        names = np.asarray(list(input_features), dtype=object)
        return names[self.selected_indices_]


class MRMRSelector(BaseEstimator, TransformerMixin):
    """Greedy mRMR selector using MI relevance and correlation redundancy."""

    def __init__(
        self,
        n_features_to_select: int = 260,
        random_state: int = 42,
        relevance_method: str = "mutual_info",
    ):
        self.n_features_to_select = n_features_to_select
        self.random_state = random_state
        self.relevance_method = relevance_method

    def fit(self, x: np.ndarray, y: Iterable[int]):
        x = _clean_array(x)
        y = np.asarray(list(y))
        n_features = x.shape[1]
        k_features = _safe_k(self.n_features_to_select, n_features)

        if self.relevance_method == "f_classif":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                relevance, _ = _deps["f_classif"](x, y)
            self.relevance_method_ = "f_classif"
        else:
            try:
                relevance = _deps["mutual_info_classif"](x, y, random_state=self.random_state)
                self.relevance_method_ = "mutual_info_classif"
            except ValueError:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    relevance, _ = _deps["f_classif"](x, y)
                self.relevance_method_ = "f_classif_fallback"
        relevance = np.nan_to_num(relevance, nan=0.0, posinf=0.0, neginf=0.0)
        centered = x - x.mean(axis=0)
        scale = np.linalg.norm(centered, axis=0)
        scale[scale == 0] = 1.0
        standardized = centered / scale
        redundancy_sum = np.zeros(n_features, dtype=float)

        selected: list[int] = []
        candidates = set(range(n_features))
        while candidates and len(selected) < k_features:
            candidate_list = np.asarray(sorted(candidates), dtype=int)
            if selected:
                redundancy = redundancy_sum[candidate_list] / len(selected)
                scores = relevance[candidate_list] - redundancy
            else:
                scores = relevance[candidate_list]
            best = int(candidate_list[int(np.argmax(scores))])
            selected.append(best)
            candidates.remove(best)
            redundancy_sum += np.abs(standardized.T @ standardized[:, best])

        self.scores_ = relevance
        self.selected_indices_ = np.asarray(selected, dtype=int)
        self.n_features_in_ = n_features
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        _deps["check_is_fitted"](self, "selected_indices_")
        return _clean_array(x)[:, self.selected_indices_]

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        _deps["check_is_fitted"](self, "selected_indices_")
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        names = np.asarray(list(input_features), dtype=object)
        return names[self.selected_indices_]


class LassoSelector(BaseEstimator, TransformerMixin):
    """Embedded L1 multinomial logistic selector used as the LASSO branch."""

    def __init__(self, n_features_to_select: int = 260, random_state: int = 42, max_iter: int = 2000):
        self.n_features_to_select = n_features_to_select
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, x: np.ndarray, y: Iterable[int]):
        x = _clean_array(x)
        y = np.asarray(list(y))
        n_features = x.shape[1]
        k_features = _safe_k(self.n_features_to_select, n_features)
        model = _deps["LogisticRegression"](
            solver="saga",
            C=1.0,
            l1_ratio=1.0,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", _deps["ConvergenceWarning"])
            model.fit(x, y)
        coefs = np.asarray(model.coef_)
        scores = np.mean(np.abs(coefs), axis=0)
        if np.count_nonzero(scores) < k_features:
            # Sparse L1 models can zero too many coefficients on small folds; keep
            # the requested top-k by absolute coefficient magnitude for stability.
            scores = scores + 1e-12 * np.arange(n_features)[::-1]
        order = np.argsort(scores)[::-1]
        self.estimator_ = model
        self.scores_ = scores
        self.selected_indices_ = order[:k_features]
        self.n_features_in_ = n_features
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        _deps["check_is_fitted"](self, "selected_indices_")
        return _clean_array(x)[:, self.selected_indices_]

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        _deps["check_is_fitted"](self, "selected_indices_")
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        names = np.asarray(list(input_features), dtype=object)
        return names[self.selected_indices_]


def build_feature_selector(method: str, n_features_to_select: int, random_state: int = 42, fast: bool = False):
    normalized = method.lower().replace("-", "").replace("_", "")
    if normalized == "mrmr":
        relevance_method = "f_classif" if fast else "mutual_info"
        return MRMRSelector(
            n_features_to_select=n_features_to_select,
            random_state=random_state,
            relevance_method=relevance_method,
        )
    if normalized == "relieff":
        return ReliefFSelector(n_features_to_select=n_features_to_select, n_neighbors=5)
    if normalized == "lasso":
        return LassoSelector(n_features_to_select=n_features_to_select, random_state=random_state)
    raise ValueError(f"Unknown feature selection method: {method}")


def summarize_selector(selector, input_features: Iterable[str], method: str, requested_features: int) -> SelectionSummary:
    selected = list(selector.get_feature_names_out(input_features))
    return SelectionSummary(
        method=method,
        requested_features=requested_features,
        selected_features=len(selected),
        selected_names=selected,
    )


def select_mrmr_features(frame: pd.DataFrame, target: pd.Series, top_k: int = 50) -> pd.DataFrame:
    """Select top-k columns with the implemented mRMR selector."""

    selector = MRMRSelector(n_features_to_select=top_k)
    selector.fit(frame.to_numpy(dtype=float), target.to_numpy())
    return frame.iloc[:, selector.selected_indices_]
