# Deep Transfer Learning and Hybrid Deep Features

This module provides end-to-end 3D classification and deep-feature hybrid classification.

The full cohort analysis is run with `paper_code_repo run-study` as documented in the [README](../README.md). The commands below expose individual stages for focused analyses. The complete workflow creates fold-specific training, validation, and test manifests, selects fine-tuning strategies by validation results, and evaluates only selected checkpoints on held-out patients.

Use Python 3.11.9. For CUDA 12.1, install the official PyTorch 2.4.1 and torchvision 0.19.1 wheels from the [PyTorch version archive](https://pytorch.org/get-started/previous-versions/):

```powershell
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

## Branch 1: End-to-End ROI Attention Classification

Each subject is represented by 40 ROI-specific 3D patches derived from co-registered T1, T2, and FLAIR MRI.

Default input representation:

```text
ROI patches per subject: 40
Channels: 3 (T1, T2, FLAIR)
Axial slices: 32
In-plane size: 224 x 224
Tensor shape: 40 x 3 x 32 x 224 x 224
```

The model processes each ROI patch independently using a 3D backbone and then aggregates ROI embeddings using gated attention.

Implemented backbones:

```text
ResNet18    -> 512-dimensional ROI embedding
ResNet50    -> 2048-dimensional ROI embedding
DenseNet121 -> 1024-dimensional ROI embedding
```

ResNet18 and ResNet50 require the official [MedicalNet 23-dataset checkpoints](https://github.com/Tencent/MedicalNet): `resnet_18_23dataset.pth` and `resnet_50_23dataset.pth`. Their 3D backbone matches the official stem, shortcut and dilation layout. The one-channel input convolution is repeated across T1/T2/FLAIR and divided by three. DenseNet121 requires the PyTorch [RadImageNet DenseNet121 weights](https://github.com/BMEII-AI/RadImageNet). Every 2D convolution kernel is repeated across its 3D depth and divided by that depth; the 2D classifier is discarded. Training rejects a missing, mismatched or incomplete source checkpoint and records the source file SHA-256 in output metadata.

## Fine-Tuning Strategies

ResNet strategies:

```text
linear_probe
from_layer2
from_layer3
from_layer4
full
```

DenseNet strategies:

```text
linear_probe
from_denseblock2
from_denseblock3
from_denseblock4
full
```

Training defaults:

```text
epochs: 100
optimizer: AdamW (betas 0.9, 0.999)
learning_rate: 1e-4
weight_decay: 1e-4
batch_size: 16 for ResNet18, 8 for ResNet50 and DenseNet121
loss: inverse-frequency weighted cross-entropy
dropout: 0.3
scheduler: ReduceLROnPlateau (factor 0.5, patience 10, min_lr 1e-6)
augmentation: full preprocessed volume before ROI cropping, training only
seed: 42
```

The batch sizes are logical patient batches. The loader processes one patient at a time and accumulates gradients across 16 or 8 patients; each patient's 40 ROI patches are encoded sequentially to fit volumetric data in memory. Batch-normalization running statistics remain fixed during fine-tuning; affine parameters still follow the chosen freeze strategy.

## Prepare ROI Patches

```powershell
.\.venv\Scripts\python.exe -m paper_code_repo prepare-deep-patches `
  --t1 "D:\preprocessed\P001\07_normalized\T1.nii.gz" `
  --t2 "D:\preprocessed\P001\07_normalized\T2.nii.gz" `
  --flair "D:\preprocessed\P001\07_normalized\FLAIR.nii.gz" `
  --roi-mask "D:\preprocessed\P001\09_rois\regions_40.nii.gz" `
  --patient-id "P001" `
  --label "ALD" `
  --output "D:\deep_patches\P001.pt"
```

Create a manifest CSV:

```text
patient_id,label,patch_path
P001,ALD,D:\deep_patches\P001.pt
P002,MLD,D:\deep_patches\P002.pt
```

Each patch tensor records its four source-volume paths so training-only augmentation can transform the complete 3D volume and ROI map before cropping.

## Train End-to-End

```powershell
.\.venv\Scripts\python.exe -m paper_code_repo train-deep-attention `
  --train-manifest "D:\splits\train.csv" `
  --val-manifest "D:\splits\val.csv" `
  --test-manifest "D:\splits\test.csv" `
  --backbone "resnet18" `
  --fine-tuning "from_layer3" `
  --medicalnet-checkpoint "D:\MedicalNet\resnet_18_23dataset.pth" `
  --epochs 100 `
  --output-dir "D:\results\deep_resnet18_layer3"
```

Repeat for all five fine-tuning strategies per backbone.
The default `--imbalance-mode both` applies training-only augmentation and weighted loss. Use `baseline`, `augmentation`, or `weighted_loss` for the other three ablation settings. Training, validation, and test manifests must have disjoint patient IDs. The held-out test is evaluated after selecting the best validation checkpoint.

For ResNet50, set `--backbone resnet50` and `--medicalnet-checkpoint "D:\MedicalNet\resnet_50_23dataset.pth"`. For DenseNet121, set `--backbone densenet121` and `--radimagenet-checkpoint "D:\RadImageNet\densenet121_torch.pt"` (replace this example path with the downloaded official PyTorch DenseNet121 file). The selected 2D checkpoint must contain torchvision-compatible DenseNet121 feature keys. Source checkpoints are external files and are not bundled with this repository.

After all strategies have been trained, select the best strategy for each backbone:

```powershell
.\.venv\Scripts\python.exe -m paper_code_repo select-deep-finetuning `
  --results-root "D:\results\deep_finetuning_runs" `
  --output-dir "D:\results\deep_finetuning_selection"
```

The selection rule is highest validation accuracy, with lower validation loss used to break ties. Use the selected checkpoint for subsequent deep feature extraction.

## Branch 2: Deep-Feature Hybrid ML

For the hybrid branch, use the selected task-specific ROI attention checkpoint from end-to-end training as a feature extractor. `--model-checkpoint` is required and loaded strictly; its saved backbone, class count and pretraining source are checked. The attention-pooled patient embedding is saved as a CSV and then passed to supervised ML classifiers after mRMR feature selection.

The default deep-feature counts cover 20 to 400 features:

```text
20, 40, 60, ..., 400
```

```powershell
.\.venv\Scripts\python.exe -m paper_code_repo extract-deep-features `
  --manifest "D:\splits\all_subjects_patches.csv" `
  --backbone "densenet121" `
  --fine-tuning "from_denseblock3" `
  --model-checkpoint "D:\results\deep_densenet121_block3\best_deep_attention.pt" `
  --output-dir "D:\features\deep_densenet121"
```

Repeat extraction with each of the five outer-fold checkpoints. For fold `k`, save the resulting full-cohort deep feature table as `D:\features\deep_by_fold\fold_k.csv`. The fold `k` checkpoint must have been trained without any patient assigned to test fold `k`. The accompanying fold assignment CSV must contain `patient_id,fold` with the same IDs as the feature tables.

Deep-feature-only hybrid classification:

```powershell
.\.venv\Scripts\python.exe -m paper_code_repo train-hybrid-ml `
  --features "D:\features\deep_by_fold\fold_0.csv" `
  --fold-assignments "D:\results\splits\fold_assignments.csv" `
  --fold-features-dir "D:\features\deep_by_fold" `
  --classifiers "svm,knn,rf,xgboost" `
  --output-dir "D:\results\hybrid_deep_only_densenet121"
```

Radiomics + selected deep fused hybrid classification: merge the complete feature tables by patient ID for each outer fold, using that fold's deep feature table. Save the five merged tables as `fold_0.csv` through `fold_4.csv`. The supervised mRMR selectors run separately inside each training fold and select 260 radiomic features plus 20–400 deep features. A fold-fitted standard scaler then z-scores the selected fused vector. The best feature count is chosen separately for each classifier.

When a patient has repeated examinations, the deep embedding extractor and fusion utility average numeric features across that patient's examinations; the traditional radiomics branch retains examination rows and majority-votes their predictions.

```powershell
.\.venv\Scripts\python.exe -m paper_code_repo fuse-features `
  --radiomics "D:\features\radiomics_patient_features.csv" `
  --deep-features "D:\features\deep_by_fold\fold_0.csv" `
  --output "D:\features\fused_by_fold\fold_0.csv"

.\.venv\Scripts\python.exe -m paper_code_repo train-hybrid-ml `
  --features "D:\features\fused_by_fold\fold_0.csv" `
  --fold-assignments "D:\results\splits\fold_assignments.csv" `
  --fold-features-dir "D:\features\fused_by_fold" `
  --mode fused `
  --classifiers "svm,knn,rf,xgboost" `
  --output-dir "D:\results\hybrid_fused_densenet121"
```

`fuse-selected-features` remains an export utility for a final refit feature list; it is not an evaluation input because selecting features on all patients before cross-validation would leak test information.

`train-hybrid-ml` saves the same model-selection outputs as the traditional ML command, including the best overall hybrid model and the best SVM, KNN, RF, and XGBoost configurations separately.

## DenseNet121 Grad-CAM

For a held-out DenseNet121 case, run `gradcam-densenet --model-checkpoint CHECKPOINT --patch CASE.pt --output-prefix CASE_gradcam`. The command saves normalized 3D heatmaps for all 40 ROIs and a FLAIR middle-slice montage. Select representative cases and interpret the resulting maps separately.

## Model Validation

```powershell
.\.venv\Scripts\python.exe -m paper_code_repo validate-deep-models `
  --output-dir "D:\results\deep_model_validation"
```

This command checks tensor shapes, attention aggregation, embedding dimensions, and fine-tuning freeze rules for the configured 3D backbones.
