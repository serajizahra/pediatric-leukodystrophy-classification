# Pediatric Leukodystrophy MRI Classification

This repository provides a connected DICOM-to-evaluation workflow for five pediatric leukodystrophy classes: ALD, mitochondrial complex I deficiency (CIM), MLC, MLD, and PMDL. It uses axial T1, T2, and FLAIR images throughout preprocessing, handcrafted feature extraction, 3D convolutional representation learning, and classification.

## Input data

Arrange DICOM studies in class folders. A patient may have multiple examinations; the workflow identifies examinations by DICOM `StudyInstanceUID` and keeps all examinations of one patient in the same outer fold.

```text
dataset_root/
  ALD/patient_001/**/*.dcm
  CIM/patient_002/**/*.dcm
  MLC/patient_003/**/*.dcm
  MLD/patient_004/**/*.dcm
  PMDL/patient_005/**/*.dcm
```

The five class folders may also be replaced by patient folders whose names end in a class abbreviation, such as `patient_001 - MLD`. The input dataset remains local. A deidentified illustrative image is available at [data/sample_axial_mri.png](data/sample_axial_mri.png), with one derived axial DICOM slice per modality in [data/sample_dicom](data/sample_dicom). These single-slice examples are for viewing the file format; the workflow requires complete series.

## Dependencies and pretrained weights

Install Python 3.11 and the project dependencies, then install the external preprocessing tools:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[preprocessing,handcrafted,traditional,deep]"
.\scripts\setup_preprocessing.ps1
.\scripts\pull_preprocessing_containers.ps1
.\.venv\Scripts\python.exe -m paper_code_repo preprocessing-doctor
```

Docker Desktop supplies the pinned ANTs, SynthStrip, and SynthSeg containers. The three source checkpoints are provided as local files:

| Backbone | Medical source weights | Argument |
| --- | --- | --- |
| 3D ResNet18 | [MedicalNet](https://github.com/Tencent/MedicalNet) 23-dataset `resnet_18_23dataset.pth` | `--medicalnet18` |
| 3D ResNet50 | [MedicalNet](https://github.com/Tencent/MedicalNet) 23-dataset `resnet_50_23dataset.pth` | `--medicalnet50` |
| 3D DenseNet121 | [RadImageNet](https://github.com/BMEII-AI/RadImageNet) PyTorch DenseNet121 2D weights, inflated to 3D | `--radimagenet121` |

## Run the study

```powershell
.\.venv\Scripts\python.exe -m paper_code_repo run-study `
  --data-root "D:\dataset_root" `
  --output-dir "D:\leukodystrophy_results" `
  --medicalnet18 "D:\weights\resnet_18_23dataset.pth" `
  --medicalnet50 "D:\weights\resnet_50_23dataset.pth" `
  --radimagenet121 "D:\weights\radimagenet_densenet121.pth"
```

The workflow discovers complete T1/T2/FLAIR studies, creates five patient-stratified folds, and independently registers T2 and FLAIR to T1 before mapping all three images to MNI space. Fold-specific T2 and FLAIR histogram references are drawn from the training patients. Each examination contributes handcrafted features from all three modalities and a 40-region, three-channel 3D patch tensor.

The classification branches cover radiomics-only models, end-to-end ROI-attention 3D CNNs, deep-feature hybrid models, and radiomics-plus-deep hybrid models. ResNet18, ResNet50, and DenseNet121 each use five fine-tuning strategies; the DenseNet branch also records four augmentation and class-weighting ablations. Traditional and hybrid models apply fold-contained SMOTE, normalization, feature selection, and grid search. The final evaluation exports patient-level out-of-fold predictions, fold metrics, 95% confidence intervals, paired tests, expected calibration error, and calibration curves.

Outputs are organized under `folds/`, `features/`, `models/`, and `evaluation/` in the requested output directory. The full sequence and individual commands are described in [docs/WORKFLOW.md](docs/WORKFLOW.md). Method details are in [docs/PREPROCESSING.md](docs/PREPROCESSING.md), [docs/HANDCRAFTED_FEATURES.md](docs/HANDCRAFTED_FEATURES.md), [docs/TRADITIONAL_ML.md](docs/TRADITIONAL_ML.md), and [docs/DEEP_TRANSFER_AND_HYBRID.md](docs/DEEP_TRANSFER_AND_HYBRID.md).

## Citation

Cite the exact repository commit or tagged release used for analysis; see [CITATION.cff](CITATION.cff).
