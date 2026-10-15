"""Fold-local mRMR selection for the 260-radiomic plus deep-feature fusion."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..radiomics.selection import MRMRSelector


class FusedMRMRSelector(BaseEstimator, TransformerMixin):
    def __init__(self, radiomic_indices, deep_indices, deep_count=300, radiomic_count=260, random_state=42):
        self.radiomic_indices = radiomic_indices
        self.deep_indices = deep_indices
        self.deep_count = deep_count
        self.radiomic_count = radiomic_count
        self.random_state = random_state

    def fit(self, x, y):
        x = np.asarray(x)
        self.radiomic_selector_ = MRMRSelector(self.radiomic_count, random_state=self.random_state)
        self.deep_selector_ = MRMRSelector(self.deep_count, random_state=self.random_state)
        self.radiomic_selector_.fit(x[:, self.radiomic_indices], y)
        self.deep_selector_.fit(x[:, self.deep_indices], y)
        self.selected_indices_ = np.r_[
            np.asarray(self.radiomic_indices)[self.radiomic_selector_.selected_indices_],
            np.asarray(self.deep_indices)[self.deep_selector_.selected_indices_],
        ]
        self.n_features_in_ = x.shape[1]
        return self

    def transform(self, x):
        check_is_fitted(self, "selected_indices_")
        return np.asarray(x)[:, self.selected_indices_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "selected_indices_")
        names = np.asarray(input_features if input_features is not None else [f"x{i}" for i in range(self.n_features_in_)])
        return names[self.selected_indices_]
