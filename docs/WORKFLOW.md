# Processing and Classification Workflow

The `run-study` command executes all study stages from a DICOM cohort to out-of-fold patient-level evaluation. The required data layout and checkpoint arguments are in the [README](../README.md).

## Sequence

1. Discover DICOM studies in the ALD, CIM, MLC, MLD, and PMDL groups. Select axial T1, T2, and FLAIR acquisitions when available; a non-axial 3D acquisition is reformatted to axial after conversion when needed. Exclude studies without all three modalities.
2. Assign patients to five stratified outer folds. Repeat examinations share their patient's fold.
3. For each outer fold, choose a histogram reference examination from training patients. Preprocess every examination with that fold's T2 and FLAIR references.
4. Convert the selected DICOM series to NIfTI, reorient to RAS, resample to 1 mm isotropic voxels, rigidly register T2 and FLAIR separately to T1, and register the co-registered set to MNI152. Denoise, extract the brain, normalize intensities, segment anatomy, and map the parcellation to 40 regions.
5. Extract 3D radiomics, uniform local binary patterns, Gabor, and DCT/SWT/DCST descriptors from T1, T2, and FLAIR. Save one feature row per examination for each outer fold.
6. Build 40 ROI-specific T1/T2/FLAIR 3D patches per examination. Split the outer training patients into training and validation partitions; keep the outer test patients untouched during fine-tuning selection.
7. Fine-tune MedicalNet ResNet18, MedicalNet ResNet50, and inflated RadImageNet DenseNet121 with five strategies each. Select each backbone's strategy using validation accuracy and validation loss. Evaluate selected checkpoints on the outer test patients, then extract deep representations with each fold's selected checkpoint.
8. Evaluate the four traditional classifiers with radiomics-only features, deep features, and fused radiomics/deep features. Fit imputation, SMOTE, scaling, mRMR/ReliefF/LASSO selection, and hyperparameter search within the training partitions. The fused branch selects 260 radiomics features plus 20 to 400 deep features.
9. Aggregate repeat-examination predictions to one result per patient. Export fold metrics, mean and 95% confidence interval, paired recall tests, exact McNemar tests, expected calibration error, and calibration curves.

The DenseNet121 ablation trains `baseline`, `augmentation`, `weighted_loss`, and `both` modes on the selected fine-tuning strategy, recording validation results separately.

## Output tree

```text
results/
  cohort.csv
  excluded_studies.csv
  fold_assignments.csv
  fold_summary.csv
  folds/fold_0/ ... folds/fold_4/
    preprocessing/reference_subject/
    preprocessing/exams/<exam_id>/
    features/radiomics.csv
    features/patch_manifest.csv
    manifests/{train,val,test,all}_manifest.csv
    deep/<backbone>/
  features/radiomics/fold_<k>.csv
  features/deep/<backbone>/fold_<k>.csv
  features/fused/<backbone>/fold_<k>.csv
  models/
  evaluation/
    model_patient_predictions.csv
    model_fold_metrics.csv
    model_summary.csv
    pairwise_statistics.csv
    calibration_bins.csv
    calibration_curves.png
  study_manifest.json
```

The output directory contains medical images and derived patient data and should be managed under the dataset's access controls. Source DICOM, pretrained checkpoints, and generated results are excluded from this repository.

## Individual stages

The CLI also exposes `inspect-series`, `preprocess-patient`, `extract-multimodal-handcrafted`, `prepare-deep-patches`, `train-deep-attention`, `extract-deep-features`, `train-traditional`, `train-hybrid-ml`, `fuse-features`, and `gradcam-densenet` for stage-level analyses. `run-study` supplies the shared fold assignments, fold-specific feature tables, and training-only histogram references automatically.
