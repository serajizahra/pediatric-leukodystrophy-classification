from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..utils.files import ensure_dir, write_json
from .finetuning import apply_fine_tuning_strategy
from .models import build_roi_attention_model


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader

        return torch, nn, DataLoader
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError("PyTorch is required for deep-learning training.") from exc


@dataclass(frozen=True)
class DeepTrainingConfig:
    backbone: str
    fine_tuning: str
    num_classes: int = 5
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 8
    imbalance_mode: str = "both"
    seed: int = 42
    medicalnet_checkpoint: str | None = None
    radimagenet_checkpoint: str | None = None
    output_dir: Path | None = None


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_loss: float
    best_metric: float
    checkpoint_path: str | None
    history_path: str | None


def _batch_to_device(batch, device):
    return batch["patches"].to(device), batch["label"].to(device)


def build_configured_model(config: DeepTrainingConfig):
    torch, _, _ = _require_torch()
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if config.backbone.lower() in {"resnet18", "resnet50"} and not config.medicalnet_checkpoint:
        raise ValueError(f"{config.backbone} requires its official MedicalNet 23-dataset checkpoint.")
    if config.backbone.lower() == "densenet121" and not config.radimagenet_checkpoint:
        raise ValueError("DenseNet121 requires RadImageNet pretrained weights.")
    model = build_roi_attention_model(
        config.backbone,
        in_channels=3,
        num_classes=config.num_classes,
        medicalnet_checkpoint=config.medicalnet_checkpoint,
        radimagenet_checkpoint=config.radimagenet_checkpoint,
    )
    freeze_summary = apply_fine_tuning_strategy(model, config.backbone, config.fine_tuning)
    return model, freeze_summary


def train_deep_attention_model(model, train_loader, val_loader, config: DeepTrainingConfig) -> TrainingResult:
    import random

    import numpy as np
    import pandas as pd

    torch, nn, _ = _require_torch()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    expected_source = "MedicalNet 23-dataset" if config.backbone.lower().startswith("resnet") else "RadImageNet DenseNet121 2D"
    if not isinstance(getattr(model, "pretraining", None), dict) or model.pretraining.get("source") != expected_source:
        raise ValueError(f"Training {config.backbone} requires loaded {expected_source} pretrained weights.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )
    if config.imbalance_mode not in {"baseline", "augmentation", "weighted_loss", "both"}:
        raise ValueError("Unknown augmentation/imbalance ablation mode")
    if config.imbalance_mode in {"weighted_loss", "both"}:
        patient_labels = {}
        for sample in train_loader.dataset.samples:
            if sample.patient_id in patient_labels and patient_labels[sample.patient_id] != sample.label:
                raise ValueError("A patient has conflicting training labels")
            patient_labels[sample.patient_id] = sample.label
        labels = list(patient_labels.values())
        counts = torch.bincount(torch.tensor(labels), minlength=config.num_classes)
        if torch.any(counts == 0):
            raise ValueError("Every class must occur in the training fold")
        weights = 1.0 / counts.float()
        weights *= len(labels) / weights[torch.tensor(labels)].sum()
        criterion = nn.CrossEntropyLoss(weight=weights.to(device), reduction="none")
    else:
        criterion = nn.CrossEntropyLoss(reduction="none")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6,
    )
    output_dir = ensure_dir(config.output_dir) if config.output_dir else None
    checkpoint_path = output_dir / "best_deep_attention.pt" if output_dir else None
    history_path = output_dir / "deep_training_history.json" if output_dir else None

    best_loss = float("inf")
    best_metric = 0.0
    best_epoch = 0
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm3d):
                module.eval()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader, start=1):
            inputs, targets = _batch_to_device(batch, device)
            logits = model(inputs)
            loss = criterion(logits, targets).mean()
            effective_batch = min(config.batch_size, len(train_loader) - ((batch_index - 1) // config.batch_size) * config.batch_size)
            (loss / effective_batch).backward()
            if batch_index % config.batch_size == 0 or batch_index == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            train_loss += float(loss.item())
            train_correct += int((logits.argmax(dim=1) == targets).sum().item())
            train_total += int(targets.numel())

        model.eval()
        val_loss = 0.0
        val_rows = []
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = _batch_to_device(batch, device)
                logits = model(inputs)
                loss = criterion(logits, targets).mean()
                val_loss += float(loss.item())
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                for patient_id, truth, scores in zip(batch["patient_id"], targets.cpu().tolist(), probabilities):
                    val_rows.append({"patient_id": patient_id, "truth": truth,
                                     **{f"probability_{i}": float(p) for i, p in enumerate(scores)}})

        mean_train_loss = train_loss / max(len(train_loader), 1)
        mean_val_loss = val_loss / max(len(val_loader), 1)
        scheduler.step(mean_val_loss)
        val_frame = pd.DataFrame(val_rows)
        if val_frame.empty or val_frame.groupby("patient_id")["truth"].nunique().gt(1).any():
            raise ValueError("Validation data are empty or contain conflicting patient labels")
        probability_columns = [f"probability_{i}" for i in range(config.num_classes)]
        patient_frame = val_frame.groupby("patient_id", as_index=False).agg(
            {"truth": "first", **{column: "mean" for column in probability_columns}}
        )
        accuracy = float(np.mean(patient_frame[probability_columns].to_numpy().argmax(axis=1) == patient_frame["truth"].to_numpy()))
        row = {
            "epoch": epoch,
            "train_loss": mean_train_loss,
            "train_accuracy": train_correct / train_total if train_total else 0.0,
            "val_loss": mean_val_loss,
            "val_accuracy": accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        if accuracy > best_metric or (accuracy == best_metric and mean_val_loss < best_loss):
            best_loss = mean_val_loss
            best_metric = accuracy
            best_epoch = epoch
            if checkpoint_path:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": mean_val_loss,
                        "val_accuracy": accuracy,
                        "config": {
                            **config.__dict__,
                            "medicalnet_checkpoint": str(config.medicalnet_checkpoint) if config.medicalnet_checkpoint else None,
                            "radimagenet_checkpoint": str(config.radimagenet_checkpoint) if config.radimagenet_checkpoint else None,
                            "output_dir": str(config.output_dir) if config.output_dir else None,
                        },
                        "pretraining": model.pretraining,
                    },
                    checkpoint_path,
                )

    if history_path:
        write_json(history_path, {"history": history})
    return TrainingResult(
        best_epoch=best_epoch,
        best_loss=best_loss,
        best_metric=best_metric,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        history_path=str(history_path) if history_path else None,
    )


def evaluate_held_out_patients(model, loader, checkpoint_path: str | Path, output_dir: str | Path):
    """Evaluate an untouched test fold after validation selected the checkpoint."""
    import pandas as pd

    from ..evaluation.metrics import expected_calibration_error, precision_recall_f1_specificity
    from ..utils.files import write_dataframe

    torch, _, _ = _require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            probability = torch.softmax(model(batch["patches"].to(device)), dim=1).cpu().numpy()
            for patient_id, truth, scores in zip(batch["patient_id"], batch["label"].tolist(), probability):
                rows.append({"patient_id": patient_id, "truth": truth, **{f"probability_{i}": float(p) for i, p in enumerate(scores)}})
    examinations = pd.DataFrame(rows)
    if examinations.empty:
        raise ValueError("Held-out test manifest is empty")
    if examinations.groupby("patient_id")["truth"].nunique().gt(1).any():
        raise ValueError("A patient has conflicting test labels")
    probability_columns = [column for column in examinations if column.startswith("probability_")]
    patients = examinations.groupby("patient_id", as_index=False).agg(
        {"truth": "first", **{column: "mean" for column in probability_columns}}
    )
    patients["prediction"] = patients[probability_columns].to_numpy().argmax(axis=1)
    report = precision_recall_f1_specificity(
        patients["truth"], patients["prediction"], labels=list(range(len(probability_columns))),
    )
    output = ensure_dir(output_dir)
    write_dataframe(output / "held_out_patient_predictions.csv", patients)
    write_json(output / "held_out_metrics.json", {
        **report.__dict__,
        "ece": expected_calibration_error(patients["truth"].to_numpy(), patients[probability_columns].to_numpy()),
    })
    return report
