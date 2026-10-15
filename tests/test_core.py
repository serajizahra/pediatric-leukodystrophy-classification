from pathlib import Path

import numpy as np
import pandas as pd

from paper_code_repo.dataset.discovery import discover_dataset
from paper_code_repo.dataset.splits import stratified_kfold_assignments
from paper_code_repo.evaluation.metrics import precision_recall_f1_specificity
from paper_code_repo.preprocessing.inventory import DicomSeries, score_series, select_modalities
from paper_code_repo.preprocessing.rois import remap_synthseg_to_40_rois
from paper_code_repo.radiomics.time_frequency import extract_time_frequency_roi_features
from paper_code_repo.radiomics.transforms import dcst2, idcst2
from paper_code_repo.radiomics.ulbp3d import extract_ulbp3d_features


def test_discover_dataset(tmp_path: Path):
    class_dir = tmp_path / "ALD"
    patient_dir = class_dir / "patient_a"
    patient_dir.mkdir(parents=True)
    (patient_dir / "1.dcm").write_text("dummy", encoding="utf-8")
    (patient_dir / "2.jpg").write_text("dummy", encoding="utf-8")

    records = discover_dataset(tmp_path)
    assert len(records) == 1
    record = records[0]
    assert record.class_label == "ALD"
    assert record.dicom_count == 1
    assert record.image_count == 1


def test_stratified_folds_have_rows():
    frame = pd.DataFrame(
        {
            "class_label": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "patient_id": [f"p{i}" for i in range(9)],
        }
    )
    assignments = stratified_kfold_assignments(frame, n_splits=3, seed=1)
    assert len(assignments) == 9
    assert set(assignments["fold"]) == {0, 1, 2}


def test_classification_metrics_bounds():
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0])
    report = precision_recall_f1_specificity(y_true, y_pred)
    assert 0.0 <= report.accuracy <= 1.0
    assert 0.0 <= report.precision <= 1.0
    assert 0.0 <= report.recall <= 1.0
    assert 0.0 <= report.f1 <= 1.0
    assert 0.0 <= report.specificity <= 1.0


def test_modality_selector_rejects_projection_and_dwi():
    t1 = DicomSeries(
        uid="t1",
        files=[Path(f"t1_{index}.dcm") for index in range(120)],
        series_description="sT1W_3D_TFE",
        image_type=("ORIGINAL", "PRIMARY"),
        slice_thickness=1.0,
    )
    t2 = DicomSeries(
        uid="t2",
        files=[Path(f"t2_{index}.dcm") for index in range(180)],
        series_description="3D_Brain_VIEW_T2",
        image_type=("ORIGINAL", "PRIMARY"),
        slice_thickness=1.0,
    )
    flair = DicomSeries(
        uid="flair",
        files=[Path(f"flair_{index}.dcm") for index in range(40)],
        series_description="3D_FLAIR_3mm_Tra",
        image_type=("ORIGINAL", "PRIMARY"),
        slice_thickness=3.0,
    )
    projection = DicomSeries(
        uid="projection",
        files=[Path(f"p_{index}.dcm") for index in range(200)],
        series_description="T2 projection",
        image_type=("ORIGINAL", "PRIMARY", "PROJECTION IMAGE"),
    )
    dwi = DicomSeries(
        uid="dwi",
        files=[Path(f"dwi_{index}.dcm") for index in range(100)],
        series_description="DWI ADC",
        image_type=("ORIGINAL", "PRIMARY"),
    )

    selected = select_modalities([projection, dwi, t1, t2, flair])
    assert selected["T1"].uid == "t1"
    assert selected["T2"].uid == "t2"
    assert selected["FLAIR"].uid == "flair"
    assert score_series(projection, "T2") < score_series(t2, "T2")


def test_synthseg_labels_remap_to_40_regions(tmp_path: Path):
    import nibabel as nib

    labels = np.zeros((4, 4, 4), dtype=np.int16)
    labels[0, 0, 0] = 2
    labels[1, 1, 1] = 41
    labels[2, 2, 2] = 1024
    labels[3, 3, 3] = 255
    source = tmp_path / "synthseg.nii.gz"
    output = tmp_path / "regions_40.nii.gz"
    table = tmp_path / "regions_40.csv"
    nib.save(nib.Nifti1Image(labels, np.eye(4)), source)

    remap_synthseg_to_40_rois(source, output, table)
    remapped = np.rint(nib.load(output).get_fdata()).astype(int)
    assert remapped[0, 0, 0] == 1
    assert remapped[1, 1, 1] == 2
    assert remapped[2, 2, 2] == 33
    assert remapped[3, 3, 3] == 40
    assert table.exists()


def test_dcst_roundtrip_is_close():
    rng = np.random.default_rng(42)
    image = rng.normal(size=(16, 16))
    reconstructed = idcst2(dcst2(image))
    assert np.allclose(image, reconstructed, atol=1e-10)


def test_ulbp3d_histograms_are_l1_normalized():
    volume = np.arange(6 * 6 * 6, dtype=float).reshape(6, 6, 6)
    labels = np.zeros_like(volume, dtype=int)
    labels[1:4, 1:4, 1:4] = 1
    features = extract_ulbp3d_features(volume, labels, modality="T1", roi_ids=[1])
    bins = [value for key, value in features.items() if "__bin_" in key]
    assert len(bins) == 18
    assert np.isclose(sum(bins), 1.0)


def test_time_frequency_roi_features_are_finite():
    rng = np.random.default_rng(7)
    volume = rng.normal(size=(16, 16, 4))
    labels = np.zeros((16, 16, 4), dtype=int)
    labels[2:8, 2:8, :] = 1
    labels[9:14, 9:14, :] = 2
    features = extract_time_frequency_roi_features(volume, labels, modality="T1", roi_ids=[1, 2])
    assert features
    values = np.asarray(list(features.values()), dtype=float)
    assert np.isfinite(values).all()
    assert any("__tf_dct__" in key for key in features)
    assert any("__tf_swt__" in key for key in features)
    assert any("__tf_dcst__" in key for key in features)
