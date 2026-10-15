from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils.files import ensure_dir, write_dataframe, write_json
from .dataset import build_dataset_class, collate_deep_patch_batch, samples_from_manifest
from .finetuning import apply_fine_tuning_strategy
from .models import BACKBONE_SPECS, build_roi_attention_model


def _require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader

        return torch, DataLoader
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError("PyTorch is required for deep feature extraction.") from exc


def _load_model_checkpoint(model, checkpoint_path: str | Path, backbone: str, num_classes: int, fine_tuning: str):
    torch, _ = _require_torch()
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Feature extraction requires a task-specific checkpoint from train-deep-attention.")
    config = checkpoint.get("config", {})
    if config.get("backbone", "").lower() != backbone.lower() or config.get("num_classes") != num_classes:
        raise ValueError("Model checkpoint backbone or number of classes does not match feature extraction arguments.")
    if config.get("fine_tuning") != fine_tuning:
        raise ValueError("Model checkpoint fine-tuning strategy does not match feature extraction arguments.")
    pretraining = checkpoint.get("pretraining")
    expected_source = "MedicalNet 23-dataset" if backbone.lower().startswith("resnet") else "RadImageNet DenseNet121 2D"
    if not isinstance(pretraining, dict) or pretraining.get("source") != expected_source:
        raise ValueError(f"Checkpoint does not document required {expected_source} pretraining.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return pretraining


def extract_deep_features_from_manifest(
    manifest_csv: str | Path,
    output_dir: str | Path,
    backbone: str,
    fine_tuning: str = "linear_probe",
    model_checkpoint: str | Path | None = None,
    batch_size: int = 1,
    num_classes: int = 5,
    patch_column: str = "patch_path",
    label_column: str = "label",
    id_column: str = "patient_id",
    seed: int = 42,
) -> dict[str, Path]:
    if model_checkpoint is None:
        raise ValueError("Hybrid feature extraction requires --model-checkpoint from task-specific fine-tuning.")
    torch, DataLoader = _require_torch()
    torch.manual_seed(seed)
    output = ensure_dir(output_dir)
    samples, label_map = samples_from_manifest(manifest_csv, patch_column, label_column, id_column)
    Dataset = build_dataset_class()
    dataset = Dataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_deep_patch_batch)

    model = build_roi_attention_model(
        backbone,
        in_channels=3,
        num_classes=num_classes,
    )
    freeze_summary = apply_fine_tuning_strategy(model, backbone, fine_tuning)
    pretraining = _load_model_checkpoint(model, model_checkpoint, backbone, num_classes, fine_tuning)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    feature_rows: list[dict[str, object]] = []
    attention_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    embedding_dim = BACKBONE_SPECS[backbone.lower()].embedding_dim

    with torch.no_grad():
        for batch in loader:
            patches = batch["patches"].to(device)
            outputs = model(patches, return_features=True)
            embeddings = outputs["patient_embedding"].detach().cpu().numpy()
            attention = outputs["attention_weights"].detach().cpu().numpy()
            probabilities = torch.softmax(outputs["logits"], dim=1).detach().cpu().numpy()
            predictions = probabilities.argmax(axis=1)
            for row_idx, patient_id in enumerate(batch["patient_id"]):
                sample = samples[len(feature_rows)]
                feature_row = {"patient_id": patient_id}
                if label_column:
                    feature_row[label_column] = str((sample.metadata or {}).get("label", sample.label))
                for feature_idx in range(embedding_dim):
                    feature_row[f"{backbone}_deep_{feature_idx:04d}"] = float(embeddings[row_idx, feature_idx])
                feature_rows.append(feature_row)

                attention_row = {"patient_id": patient_id}
                for roi_idx, value in enumerate(attention[row_idx], start=1):
                    attention_row[f"roi_{roi_idx:02d}_attention"] = float(value)
                attention_rows.append(attention_row)

                prediction_row = {"patient_id": patient_id, "predicted_label_index": int(predictions[row_idx])}
                for class_idx, value in enumerate(probabilities[row_idx]):
                    prediction_row[f"class_{class_idx}_probability"] = float(value)
                prediction_rows.append(prediction_row)

    features_path = output / f"{backbone}_deep_features.csv"
    attention_path = output / f"{backbone}_attention_weights.csv"
    predictions_path = output / f"{backbone}_end_to_end_predictions.csv"
    metadata_path = output / f"{backbone}_deep_feature_metadata.json"

    feature_frame = pd.DataFrame(feature_rows)
    if feature_frame.groupby("patient_id")[label_column].nunique().gt(1).any():
        raise ValueError("Repeated examinations of one patient have conflicting labels")
    feature_columns = [column for column in feature_frame if column not in {"patient_id", label_column}]
    patient_features = feature_frame.groupby("patient_id", as_index=False).agg(
        {label_column: "first", **{column: "mean" for column in feature_columns}}
    )
    attention_frame = pd.DataFrame(attention_rows).groupby("patient_id", as_index=False).mean(numeric_only=True)
    prediction_frame = pd.DataFrame(prediction_rows)
    probability_columns = [column for column in prediction_frame if column.endswith("_probability")]
    patient_predictions = prediction_frame.groupby("patient_id", as_index=False)[probability_columns].mean()
    patient_predictions["predicted_label_index"] = patient_predictions[probability_columns].to_numpy().argmax(axis=1)
    write_dataframe(features_path, patient_features)
    write_dataframe(attention_path, attention_frame)
    write_dataframe(predictions_path, patient_predictions)
    write_json(
        metadata_path,
        {
            "manifest_csv": str(manifest_csv),
            "backbone": backbone,
            "embedding_dim": embedding_dim,
            "fine_tuning": fine_tuning,
            "freeze_summary": freeze_summary,
            "label_map": label_map,
            "model_checkpoint": str(model_checkpoint),
            "pretraining": pretraining,
            "feature_usage": "Save one full-cohort feature CSV per outer-fold checkpoint and use train-hybrid-ml with matching fold assignments.",
        },
    )
    return {
        "features": features_path,
        "attention": attention_path,
        "predictions": predictions_path,
        "metadata": metadata_path,
    }
