# Traditional Radiomics Machine Learning

The connected `run-study` workflow extracts T1, T2, and FLAIR radiomics features for every examination and supplies one complete feature table per outer fold. The same patient assignments are used for radiomics, deep, and fused branches.

This module trains the radiomics-only machine-learning classifiers.

## Pipeline

1. Read an examination-level or patient-level feature table with a patient ID.
2. Encode the five leukodystrophy subtype labels.
3. Use five-fold stratified patient-grouped outer cross-validation for model evaluation.
4. Inside each training fold only:
   - median imputation,
   - SMOTE oversampling,
   - min-max normalization,
   - feature selection using mRMR, ReliefF, or LASSO,
   - classifier hyperparameter tuning with GridSearchCV.
5. Evaluate SVM-RBF, KNN, RF, and XGBoost on the held-out fold; majority-vote repeat examinations to one patient prediction.
6. For each feature-selection method and feature size, average the mean outer-CV accuracy of the four grid-searched classifiers.
7. Select the best feature size separately for each feature-selection method using this average accuracy.
8. Select the best overall feature-selection/model configuration and refit it on all available rows.
9. Also refit and save the best configuration separately for each classifier family: SVM, KNN, RF, and XGBoost.

The default selected-feature counts cover 60 to 320 features:

```text
60, 80, 100, ..., 320
```

Use `--feature-sizes` to evaluate a different range.

## Input CSV

The input CSV may contain multiple examinations per patient; all examinations with the same `patient_id` stay in the same fold. A minimal table looks like:

```text
patient_id,label,feature_001,feature_002,...
P001,ALD,0.12,1.90,...
P002,MLD,0.44,1.51,...
```

All numeric columns except `label` and `patient_id` are treated as candidate features. An optional `exam_id` column identifies repeat examinations when aligning fold-specific tables through `--fold-features-dir` and `--fold-assignments`.

## Command

```powershell
.\.venv\Scripts\python.exe -m paper_code_repo train-traditional `
  --features "D:\features\radiomics_patient_features.csv" `
  --label-column "label" `
  --id-column "patient_id" `
  --output-dir "D:\results\traditional_ml"
```

To evaluate a custom feature-size range:

```powershell
.\.venv\Scripts\python.exe -m paper_code_repo train-traditional `
  --features "D:\features\radiomics_patient_features.csv" `
  --output-dir "D:\results\traditional_ml_200_320" `
  --feature-sizes "200,240,260,300,320"
```

For a reduced-runtime check, add `--quick`. This uses one parameter value per classifier and an ANOVA relevance score for the mRMR ranking step. Omit `--quick` for the full grid search.

## Outputs

```text
traditional_fold_results.csv
traditional_patient_predictions.csv
traditional_results.csv
traditional_feature_size_summary.csv
traditional_best_size_by_selection_method.csv
traditional_best_selected_features.csv
traditional_best_selected_features_by_method.csv
traditional_best_by_classifier.csv
traditional_best_by_classifier_selected_features.csv
traditional_best_model.json
traditional_best_by_classifier.json
```

- `traditional_fold_results.csv`: one row per outer fold and model/feature-selection/feature-size combination.
- `traditional_patient_predictions.csv`: held-out predictions and probabilities, one row per patient and configuration.
- `traditional_results.csv`: mean and standard deviation of accuracy, precision, recall, F1, and specificity.
- `traditional_feature_size_summary.csv`: classifier-averaged accuracy for each feature-selection method and feature size.
- `traditional_best_size_by_selection_method.csv`: best feature size for each selection method based on average accuracy across SVM, KNN, RF, and XGBoost.
- `traditional_best_selected_features.csv`: selected features after refitting the best configuration on all rows.
- `traditional_best_selected_features_by_method.csv`: selected features for each selection method at its best classifier-averaged feature size.
- `traditional_best_by_classifier.csv`: best feature-selection method and feature count for each classifier family.
- `traditional_best_by_classifier_selected_features.csv`: selected features after refitting each classifier family's best configuration.
- `traditional_best_model.json`: label map, CV settings, selected best configuration, and final grid-search parameters.
- `traditional_best_by_classifier.json`: final refit grid-search parameters for each classifier family.

## Leakage control

SMOTE, min-max scaling, feature selection, and grid search are fitted only inside each training fold. Both outer and inner folds group by patient ID. The held-out outer fold is transformed using the fitted training-fold pipeline and is never used during feature selection or hyperparameter tuning. SMOTE uses up to `k_neighbors=5` and automatically lowers this value in very small folds. Full-data refit feature lists are descriptive exports and must not be reused for cross-validation of a fused model.
