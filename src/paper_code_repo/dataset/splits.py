from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FoldAssignment:
    patient_id: str
    class_label: str
    fold: int


def stratified_kfold_assignments(
    frame: pd.DataFrame,
    label_column: str = "class_label",
    id_column: str = "patient_id",
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    from sklearn.model_selection import StratifiedGroupKFold

    if frame.groupby(id_column)[label_column].nunique().gt(1).any():
        raise ValueError("A patient must have one class label across all examinations")
    group_counts = frame[[id_column, label_column]].drop_duplicates().groupby(label_column).size()
    if group_counts.min() < n_splits:
        raise ValueError("Each class needs at least one distinct patient per fold")
    assignments = frame[[id_column, label_column]].copy()
    assignments["fold"] = -1
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, test_index) in enumerate(splitter.split(frame, frame[label_column], frame[id_column])):
        assignments.iloc[test_index, assignments.columns.get_loc("fold")] = fold
    if assignments["fold"].lt(0).any():
        raise RuntimeError("Some examinations were not assigned to a fold")
    if assignments.groupby(id_column)["fold"].nunique().gt(1).any():
        raise RuntimeError("Examinations of one patient were split across folds")
    assignments = assignments.sort_values([label_column, id_column]).reset_index(drop=True)
    return assignments


def summarize_folds(
    frame: pd.DataFrame, label_column: str = "class_label", fold_column: str = "fold", id_column: str = "patient_id",
) -> pd.DataFrame:
    summary = (
        frame.groupby([fold_column, label_column])[id_column]
        .nunique()
        .reset_index(name="count")
        .sort_values([fold_column, label_column])
    )
    return summary
