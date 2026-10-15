# Illustrative DICOM slices

`T1_001.dcm`, `T2_001.dcm`, and `FLAIR_001.dcm` are deidentified, single-slice, axial MR DICOM examples corresponding to the three panels in `../sample_axial_mri.png`. They contain derived 8-bit pixel data, new DICOM UIDs, generic patient identifiers, and no source patient, institution, study-date, or private-tag metadata.

These files demonstrate the DICOM format. The `run-study` workflow requires complete T1, T2, and FLAIR series with at least 20 slices per modality and patient-level class labels.
