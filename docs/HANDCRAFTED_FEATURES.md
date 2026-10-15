# Handcrafted Feature Extraction

`run-study` applies this extractor to T1, T2, and FLAIR separately for every examination, concatenates their modality-prefixed descriptors, and writes one fold-specific examination feature table. The standalone `extract-multimodal-handcrafted` command accepts `--t1`, `--t2`, `--flair`, `--roi-mask`, and `--brain-mask` for the same operation on one examination.

This module extracts handcrafted features from anatomical MRI regions.

## Implemented Families

| Feature family | Implementation |
|---|---|
| 3D shape radiomics | PyRadiomics shape descriptors per anatomical ROI |
| 3D texture radiomics | PyRadiomics GLCM, GLRLM, GLSZM, NGTDM, and GLDM per anatomical ROI |
| 3D ULBP | Uniform local binary pattern with `P=16`, `R=1`, spherical neighbors, trilinear interpolation, and L1-normalized histograms |
| Gabor | 32 2D filters per volume: 4 frequencies x 4 azimuths x 2 phase offsets; mean, std, energy, entropy |
| DCT time-frequency | 2D DCT per axial slice, LF/MF/HF coefficient bands, inverse DCT per band, ROI statistics, slice aggregation |
| SWT time-frequency | 2D stationary wavelet transform per axial slice, LL/LH/HL/HH bands, inverse SWT per band, ROI statistics, slice aggregation |
| DCST time-frequency | DCT-based discrete Stockwell transform per axial slice, LF/MF/HF bands, inverse DCST per band, ROI statistics, slice aggregation |

## Time-Frequency Statistics

For each slice, band, and ROI, the code computes 11 statistics:

- mean
- standard deviation
- variance
- mean absolute deviation
- energy
- relative energy
- 5th percentile
- 95th percentile
- median
- minimum
- maximum

For each statistic, the final examination-level feature is summarized across slices by:

- slice mean
- slice standard deviation
- slice minimum
- slice maximum

## Command

To extract all three modalities from one co-registered examination:

```powershell
.\.venv\Scripts\python.exe -m paper_code_repo extract-multimodal-handcrafted `
  --t1 "C:\path\to\07_normalized\T1.nii.gz" `
  --t2 "C:\path\to\07_normalized\T2.nii.gz" `
  --flair "C:\path\to\07_normalized\FLAIR.nii.gz" `
  --roi-mask "C:\path\to\09_rois\regions_40.nii.gz" `
  --brain-mask "C:\path\to\06_brain_extracted\brain_mask.nii.gz" `
  --patient-id "patient_001" `
  --output "C:\path\to\features\patient_001_multimodal.csv"
```

The individual-modality command is also available:

```powershell
.\.venv\Scripts\python.exe -m paper_code_repo extract-handcrafted `
  --image "C:\path\to\07_normalized\T1.nii.gz" `
  --roi-mask "C:\path\to\09_rois\regions_40.nii.gz" `
  --brain-mask "C:\path\to\06_brain_extracted\brain_mask.nii.gz" `
  --patient-id "patient_001" `
  --modality "T1" `
  --output "C:\path\to\features\patient_001_T1_handcrafted.csv"
```

The command writes:

- a feature CSV
- a metadata JSON file
- a validation JSON file checking finite numeric values

For a limited-slice quality-control run, use `--max-slices N`. Omit it for full-volume feature extraction.

## Dependency Note

The repository targets Python 3.11.9, PyRadiomics 3.1.0 (official `v3.1.0` Git tag), PyWavelets 1.9.0, and DIPY 1.12.0.
