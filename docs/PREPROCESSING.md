# Preprocessing Pipeline

## Processing steps

| Operation | Implementation |
|---|---|
| DICOM to NIfTI | `dcm2niix` with anonymized BIDS sidecars |
| RAS+ orientation | `nibabel.as_closest_canonical` |
| 1 mm isotropic resampling | SimpleITK Lanczos windowed-sinc interpolation |
| T2 and FLAIR to T1 registration | independent ANTs 2.6.5 rigid registrations with Mattes mutual information; both images resampled to the native T1 grid |
| T1 to MNI152 registration | ANTs 2.6.5 container rigid registration, Mattes MI (32 bins, 25% regular sampling) |
| T2/FLAIR transform propagation | the forward T1-to-MNI transform applied after T1 co-registration, with linear interpolation |
| Rician non-local means | DIPY 1.12.0 `nlmeans(..., rician=True, patch_radius=2, block_radius=4)` with estimated noise sigma |
| Skull stripping | FreeSurfer 8.2.0 `mri_synthstrip` |
| T1 normalization | gray-matter reference mode, WhiteStripe quantile width 0.05 on each side |
| T2/FLAIR normalization | masked quantile histogram matching to same-modality references |
| Final intensity scaling | z-score within the propagated brain mask |
| Anatomical parcellation | SynthSeg with cortical parcellation enabled (`--parc`) |
| Study atlas | explicit SynthSeg/Desikan-Killiany mapping to 40 regions |

SynthSeg 2.0 may omit FreeSurfer corpus-callosum labels 251-255. When they are absent, ROI 40 is derived conservatively from midsagittal white matter in MNI space (`|x| <= 6 mm`, `-50 <= y <= 35 mm`, `0 <= z <= 50 mm`). The ROI CSV records whether direct labels or this fallback were used. Ventral-striatum ROIs merge the adjacent Accumbens and VentralDC labels.

## Output tree

```text
patient_output/
|-- 00_inventory/
|-- 01_nifti/
|-- 02_ras/
|-- 03_resampled_1mm/
|-- 03_coregistered_t1/
|-- 04_registered_mni/
|-- 05_denoised/
|-- 06_brain_extracted/
|-- 07_normalized/
|-- 08_segmentation/
|-- 09_rois/
|-- logs/
|-- qc/
|-- resources/
`-- provenance.json
```

The final model inputs are `07_normalized/T1.nii.gz`, `T2.nii.gz`, and `FLAIR.nii.gz`. The final ROI image is `09_rois/regions_40.nii.gz`.

## Histogram-reference rule

T2 and FLAIR histogram matching uses same-modality reference images supplied as command-line inputs. In cross-validation runs, these references are selected or estimated from the training fold only to prevent information leakage.

`run-study` creates a reference from one training examination per outer fold, then uses that fold's reference images for all examinations in the corresponding fold-specific feature table. `--allow-self-reference` is available for the reference-generation examination and single-subject installation checks; its use is recorded in `provenance.json`.

## Reproducibility and privacy

Every run records selected DICOM series, package versions, external commands, transform files, estimated noise sigma, image dimensions, voxel spacing, and intensity summaries. DICOM and NIfTI files are ignored by Git. Patient DICOM metadata, generated sidecars, and patient-specific QC figures remain in the local data workspace. The repository includes one deidentified illustrative PNG.

## External backends

The default Windows setup uses these pinned containers:

- `freesurfer/freesurfer:8.2.0`, containing `mri_synthstrip` and `mri_synthseg`
- `antsx/ants:v2.6.5`, containing `antsRegistration` and `antsApplyTransforms`

When `mri_synthstrip` or `mri_synthseg` is installed locally, use `--synthstrip-backend local` or `--synthseg-backend local`.

Standard SynthSeg cortical parcellation can exceed the default WSL memory limit. On a 16GB workstation, the validated Windows configuration was:

```ini
[wsl2]
memory=12GB
processors=8
swap=8GB
```

Store this in `%USERPROFILE%\.wslconfig`, run `wsl --shutdown`, and restart Docker Desktop. Exit code `137` in `logs/synthseg.json` indicates that the container was killed because the memory limit was too low.
