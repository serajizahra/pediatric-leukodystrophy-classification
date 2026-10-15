from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DeepPatchSample:
    patch_path: str
    label: int
    patient_id: str
    metadata: dict[str, Any] | None = None


def _require_torch():
    try:
        import torch
        from torch.utils.data import Dataset

        return torch, Dataset
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError("PyTorch is required for the deep-learning dataset.") from exc


def samples_from_manifest(
    manifest_csv: str | Path,
    patch_column: str = "patch_path",
    label_column: str = "label",
    id_column: str = "patient_id",
    label_map: dict[str, int] | None = None,
) -> tuple[list[DeepPatchSample], dict[str, int]]:
    frame = pd.read_csv(manifest_csv)
    for column in (patch_column, label_column, id_column):
        if column not in frame.columns:
            raise ValueError(f"Column '{column}' was not found in {manifest_csv}")
    labels = sorted(frame[label_column].astype(str).unique().tolist())
    label_map = label_map or {label: idx for idx, label in enumerate(labels)}
    if not set(labels).issubset(label_map):
        raise ValueError("Manifest contains a label absent from the training label map")
    samples = [
        DeepPatchSample(
            patch_path=str(row[patch_column]),
            label=label_map[str(row[label_column])],
            patient_id=str(row[id_column]),
            metadata={"source_row": int(index), "label": str(row[label_column])},
        )
        for index, row in frame.iterrows()
    ]
    return samples, label_map


def build_dataset_class():
    torch, Dataset = _require_torch()

    class DeepPatchDataset(Dataset):
        def __init__(self, samples: list[DeepPatchSample], transform=None, augment: bool = False) -> None:
            self.samples = samples
            self.transform = transform
            self.augment = augment

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int):
            sample = self.samples[index]
            payload = torch.load(sample.patch_path, map_location="cpu")
            patches = payload["patches"] if isinstance(payload, dict) and "patches" in payload else payload
            if self.augment:
                from .patches import DeepPatchConfig, extract_roi_patches

                sources = payload.get("metadata", {}).get("source_volumes", {}) if isinstance(payload, dict) else {}
                if not all(name in sources for name in ("t1", "t2", "flair", "roi_mask")):
                    raise ValueError("Training augmentation requires source_volumes in the patch metadata")
                augmented, _ = extract_roi_patches(
                    sources["t1"], sources["t2"], sources["flair"], sources["roi_mask"],
                    config=DeepPatchConfig(**payload["metadata"]["patch_config"]), augment=True,
                )
                patches = torch.from_numpy(augmented)
            if self.transform is not None:
                patches = self.transform(patches)
            return {
                "patches": patches.float(),
                "label": torch.tensor(sample.label, dtype=torch.long),
                "patient_id": sample.patient_id,
            }

    return DeepPatchDataset


def collate_deep_patch_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    torch, _ = _require_torch()
    return {
        "patches": torch.stack([item["patches"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
        "patient_id": [item["patient_id"] for item in batch],
    }
