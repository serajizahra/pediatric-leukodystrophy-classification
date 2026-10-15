from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClassificationReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    specificity: float


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int] | None = None) -> np.ndarray:
    if labels is None:
        labels = sorted(set(map(int, np.unique(np.concatenate([y_true, y_pred])))))
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for truth, pred in zip(y_true, y_pred, strict=False):
        matrix[index[int(truth)], index[int(pred)]] += 1
    return matrix


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def precision_recall_f1_specificity(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[int] | None = None,
) -> ClassificationReport:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = labels or sorted(set(map(int, np.unique(np.concatenate([y_true, y_pred])))))
    cm = confusion_matrix(y_true, y_pred, labels)

    precisions = []
    recalls = []
    f1s = []
    specificities = []

    total = cm.sum()
    for i in range(len(labels)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        specificities.append(specificity)

    return ClassificationReport(
        accuracy=accuracy_score(y_true, y_pred),
        precision=float(np.mean(precisions)),
        recall=float(np.mean(recalls)),
        f1=float(np.mean(f1s)),
        specificity=float(np.mean(specificities)),
    )


def confidence_interval(values: list[float] | np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    if len(arr) < 2:
        return mean, mean
    std = float(np.std(arr, ddof=1))
    sem = std / math.sqrt(len(arr))
    from scipy.stats import t

    critical = float(t.ppf((1 + confidence) / 2, df=len(arr) - 1))
    return mean - critical * sem, mean + critical * sem


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = predicted == np.asarray(y_true)
    error = 0.0
    for start, end in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        selected = (confidence > start) & (confidence <= end) if start else (confidence >= start) & (confidence <= end)
        if selected.any():
            error += float(selected.mean() * abs(correct[selected].mean() - confidence[selected].mean()))
    return error


def paired_recall_t_test(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    from scipy.stats import ttest_rel

    result = ttest_rel(first, second)
    return float(result.statistic), float(result.pvalue)


def exact_mcnemar(y_true: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    from scipy.stats import binomtest

    first_correct = np.asarray(first) == np.asarray(y_true)
    second_correct = np.asarray(second) == np.asarray(y_true)
    discordant_first = int(np.sum(first_correct & ~second_correct))
    discordant_second = int(np.sum(~first_correct & second_correct))
    total = discordant_first + discordant_second
    return float(binomtest(min(discordant_first, discordant_second), total, 0.5).pvalue) if total else 1.0


def classification_summary(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    report = precision_recall_f1_specificity(y_true, y_pred)
    return pd.DataFrame(
        [
            {
                "accuracy": report.accuracy,
                "precision": report.precision,
                "recall": report.recall,
                "f1": report.f1,
                "specificity": report.specificity,
            }
        ]
    )

