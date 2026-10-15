"""Out-of-fold patient-level metrics, paired comparisons, and calibration."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.files import ensure_dir, write_dataframe
from .metrics import (
    confidence_interval, exact_mcnemar, expected_calibration_error,
    paired_recall_t_test, precision_recall_f1_specificity,
)


def _selected_predictions(name: str, outputs: dict[str, Path]) -> list[pd.DataFrame]:
    predictions = pd.read_csv(outputs["patient_predictions"])
    best = pd.read_csv(outputs["best_by_classifier"])
    selected = []
    for row in best.itertuples(index=False):
        matched = predictions[
            (predictions.classifier == row.classifier)
            & (predictions.feature_selection == row.feature_selection)
            & (predictions.requested_features == row.requested_features)
        ].copy()
        if matched.empty:
            raise ValueError(f"No held-out predictions for {name}/{row.classifier}")
        matched["model"] = f"{name}_{row.classifier}"
        selected.append(matched)
    return selected


def _calibration_rows(frame: pd.DataFrame, bins: int = 10) -> list[dict[str, object]]:
    probability_columns = sorted((col for col in frame if col.startswith("probability_")),
                                 key=lambda col: int(col.rsplit("_", 1)[1]))
    probabilities = frame[probability_columns].to_numpy(dtype=float)
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == frame.truth.to_numpy(dtype=int)
    rows = []
    for bin_index, (start, end) in enumerate(zip(np.linspace(0, 1, bins + 1)[:-1],
                                                  np.linspace(0, 1, bins + 1)[1:])):
        selected = (confidence >= start) & (confidence <= end) if bin_index == 0 else (confidence > start) & (confidence <= end)
        if selected.any():
            rows.append({"model": frame.model.iloc[0], "bin": bin_index + 1,
                         "count": int(selected.sum()), "mean_confidence": float(confidence[selected].mean()),
                         "observed_accuracy": float(correct[selected].mean())})
    return rows


def summarize_study(
    model_outputs: dict[str, dict[str, Path]],
    deep_predictions: dict[str, list[Path]],
    output_dir: str | Path,
    folds: int,
) -> dict[str, Path]:
    root = ensure_dir(output_dir)
    prediction_tables = [table for name, outputs in model_outputs.items()
                         for table in _selected_predictions(name, outputs)]
    for backbone, paths in deep_predictions.items():
        for fold, path in enumerate(paths, start=1):
            frame = pd.read_csv(path)
            frame["fold"] = fold
            frame["model"] = f"{backbone}_end_to_end"
            prediction_tables.append(frame)
    all_predictions = pd.concat(prediction_tables, ignore_index=True)
    probability_columns = sorted((col for col in all_predictions if col.startswith("probability_")),
                                 key=lambda col: int(col.rsplit("_", 1)[1]))
    if all_predictions[probability_columns].isna().any().any():
        raise ValueError("Every model must supply probabilities for all diagnostic classes")
    for name, frame in all_predictions.groupby("model"):
        if frame.patient_id.duplicated().any() or frame.fold.nunique() != folds:
            raise ValueError(f"Expected one held-out prediction per patient and all folds for {name}")
    prediction_path = root / "model_patient_predictions.csv"
    write_dataframe(prediction_path, all_predictions)

    fold_rows = []
    summary_rows = []
    calibration_rows = []
    for name, model_frame in all_predictions.groupby("model"):
        for fold, frame in model_frame.groupby("fold"):
            scores = precision_recall_f1_specificity(
                frame.truth.to_numpy(dtype=int), frame.prediction.to_numpy(dtype=int),
                labels=list(range(len(probability_columns))),
            )
            fold_rows.append({"model": name, "fold": int(fold), **scores.__dict__,
                              "ece": expected_calibration_error(frame.truth.to_numpy(dtype=int),
                                                                  frame[probability_columns].to_numpy(dtype=float))})
        calibration_rows.extend(_calibration_rows(model_frame))
    fold_metrics = pd.DataFrame(fold_rows)
    for name, frame in fold_metrics.groupby("model"):
        row = {"model": name}
        for metric in ("accuracy", "precision", "recall", "f1", "specificity", "ece"):
            values = frame.sort_values("fold")[metric].to_numpy(dtype=float)
            low, high = confidence_interval(values)
            row.update({f"{metric}_mean": float(values.mean()),
                        f"{metric}_sd": float(values.std(ddof=1)),
                        f"{metric}_ci95_low": low, f"{metric}_ci95_high": high})
        summary_rows.append(row)

    pair_rows = []
    by_model = {name: frame for name, frame in all_predictions.groupby("model")}
    for first_name, second_name in combinations(sorted(by_model), 2):
        first = by_model[first_name].set_index("patient_id")
        second = by_model[second_name].set_index("patient_id")
        shared = first.index.intersection(second.index)
        if len(shared) != len(first) or len(shared) != len(second):
            raise ValueError("Paired comparisons require the same patients in every model")
        first = first.loc[shared].sort_values("fold")
        second = second.loc[first.index]
        if not np.array_equal(first.truth.to_numpy(), second.truth.to_numpy()):
            raise ValueError("Paired models have different true labels")
        first_recall = fold_metrics[fold_metrics.model == first_name].sort_values("fold").recall.to_numpy()
        second_recall = fold_metrics[fold_metrics.model == second_name].sort_values("fold").recall.to_numpy()
        statistic, p_recall = paired_recall_t_test(first_recall, second_recall)
        pair_rows.append({"first_model": first_name, "second_model": second_name,
                          "paired_recall_t": statistic, "paired_recall_p": p_recall,
                          "exact_mcnemar_p": exact_mcnemar(first.truth.to_numpy(dtype=int),
                                                             first.prediction.to_numpy(dtype=int),
                                                             second.prediction.to_numpy(dtype=int))})
    paths = {
        "patient_predictions": prediction_path,
        "fold_metrics": root / "model_fold_metrics.csv",
        "summary": root / "model_summary.csv",
        "paired_statistics": root / "pairwise_statistics.csv",
        "calibration_bins": root / "calibration_bins.csv",
        "calibration_plot": root / "calibration_curves.png",
    }
    write_dataframe(paths["fold_metrics"], fold_metrics)
    write_dataframe(paths["summary"], pd.DataFrame(summary_rows))
    write_dataframe(paths["paired_statistics"], pd.DataFrame(pair_rows))
    calibration = pd.DataFrame(calibration_rows)
    write_dataframe(paths["calibration_bins"], calibration)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 8))
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", label="Perfect calibration")
    for name, frame in calibration.groupby("model"):
        axis.plot(frame.mean_confidence, frame.observed_accuracy, marker="o", label=name)
    axis.set(xlabel="Mean predicted confidence", ylabel="Observed accuracy",
             xlim=(0, 1), ylim=(0, 1), title="Held-out patient calibration")
    axis.legend(fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(1.02, 1))
    figure.tight_layout()
    figure.savefig(paths["calibration_plot"], dpi=180, bbox_inches="tight")
    plt.close(figure)
    return paths
