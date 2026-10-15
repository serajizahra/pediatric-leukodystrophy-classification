from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np


ROI_NAMES = {
    1: "L-WM", 2: "R-WM", 3: "L-Cerebellar-WM", 4: "R-Cerebellar-WM",
    5: "L-Cerebellar-Cortex", 6: "R-Cerebellar-Cortex", 7: "L-Thalamus", 8: "R-Thalamus",
    9: "L-Caudate", 10: "R-Caudate", 11: "L-Putamen", 12: "R-Putamen",
    13: "L-Pallidum", 14: "R-Pallidum", 15: "L-Hippocampus", 16: "R-Hippocampus",
    17: "L-Amygdala", 18: "R-Amygdala", 19: "L-Ventral-Striatum", 20: "R-Ventral-Striatum",
    21: "L-Frontal", 22: "R-Frontal", 23: "L-Parietal", 24: "R-Parietal",
    25: "L-Temporal", 26: "R-Temporal", 27: "L-Occipital", 28: "R-Occipital",
    29: "L-Cingulate", 30: "R-Cingulate", 31: "L-Insula", 32: "R-Insula",
    33: "L-Precentral", 34: "R-Precentral", 35: "L-Postcentral", 36: "R-Postcentral",
    37: "Brainstem", 38: "Ventricles", 39: "Extra-Cerebral-CSF", 40: "Corpus-Callosum",
}


def _hemi(labels: set[int], right: bool) -> set[int]:
    return {label + (1000 if right else 0) for label in labels}


FRONTAL = {1003, 1012, 1014, 1017, 1018, 1019, 1020, 1027, 1028, 1032}
PARIETAL = {1008, 1025, 1029, 1031}
TEMPORAL = {1001, 1006, 1007, 1009, 1015, 1016, 1030, 1033, 1034}
OCCIPITAL = {1005, 1011, 1013, 1021}
CINGULATE = {1002, 1010, 1023, 1026}


SOURCE_LABELS: dict[int, set[int]] = {
    1: {2}, 2: {41}, 3: {7}, 4: {46}, 5: {8}, 6: {47},
    7: {10}, 8: {49}, 9: {11}, 10: {50}, 11: {12}, 12: {51},
    13: {13}, 14: {52}, 15: {17}, 16: {53}, 17: {18}, 18: {54},
    19: {26, 28}, 20: {58, 60}, 21: FRONTAL, 22: _hemi(FRONTAL, True),
    23: PARIETAL, 24: _hemi(PARIETAL, True), 25: TEMPORAL, 26: _hemi(TEMPORAL, True),
    27: OCCIPITAL, 28: _hemi(OCCIPITAL, True), 29: CINGULATE, 30: _hemi(CINGULATE, True),
    31: {1035}, 32: {2035}, 33: {1024}, 34: {2024}, 35: {1022}, 36: {2022},
    37: {16}, 38: {4, 5, 14, 15, 43, 44, 72}, 39: {24}, 40: {251, 252, 253, 254, 255},
}


def remap_synthseg_to_40_rois(
    segmentation_path: str | Path,
    output_path: str | Path,
    table_path: str | Path,
) -> Path:
    image = nib.load(str(segmentation_path))
    labels = np.rint(image.get_fdata()).astype(np.int32)
    output_labels = np.zeros(labels.shape, dtype=np.uint8)
    rows: list[dict[str, object]] = []

    for roi_id in range(1, 41):
        source = SOURCE_LABELS[roi_id]
        roi_mask = np.isin(labels, list(source))
        derivation = "SynthSeg labels"
        if roi_id == 40 and not roi_mask.any():
            # SynthSeg 2.0 does not always emit FreeSurfer labels 251-255.
            # In MNI space, recover a conservative midsagittal callosal WM ROI.
            i, j, k = np.indices(labels.shape, sparse=True)
            affine = image.affine
            x = affine[0, 0] * i + affine[0, 1] * j + affine[0, 2] * k + affine[0, 3]
            y = affine[1, 0] * i + affine[1, 1] * j + affine[1, 2] * k + affine[1, 3]
            z = affine[2, 0] * i + affine[2, 1] * j + affine[2, 2] * k + affine[2, 3]
            white_matter = np.isin(labels, [2, 41])
            roi_mask = (
                white_matter
                & (np.abs(x) <= 6.0)
                & (y >= -50.0)
                & (y <= 35.0)
                & (z >= 0.0)
                & (z <= 50.0)
            )
            derivation = "MNI midsagittal white-matter fallback"
        output_labels[roi_mask] = roi_id
        rows.append(
            {
                "roi_id": roi_id,
                "roi_name": ROI_NAMES[roi_id],
                "source_labels": ";".join(str(value) for value in sorted(source)),
                "derivation": derivation,
                "voxel_count": int(roi_mask.sum()),
            }
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(output_labels, image.affine, image.header), str(output))
    table = Path(table_path)
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output
