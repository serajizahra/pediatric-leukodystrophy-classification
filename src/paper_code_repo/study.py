"""Patient-grouped DICOM-to-evaluation workflow for the five diagnostic classes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .utils.files import ensure_dir, write_dataframe, write_json


BACKBONES = ("resnet18", "resnet50", "densenet121")
CLASSIFIERS = ("svm", "knn", "rf", "xgboost")


@dataclass(frozen=True)
class StudyConfig:
    data_root: Path
    output_dir: Path
    medicalnet18: Path
    medicalnet50: Path
    radimagenet121: Path
    folds: int = 5
    inner_folds: int = 5
    epochs: int = 100
    seed: int = 42
    n_jobs: int = 1
    min_series_slices: int = 20
    dcm2niix_path: Path | None = None
    mni_template: Path | None = None
    synthstrip_backend: str = "auto"
    synthseg_backend: str = "auto"
    synthstrip_image: str = "freesurfer/freesurfer:8.2.0"
    synthseg_image: str = "freesurfer/freesurfer:8.2.0"
    ants_image: str = "antsx/ants:v2.6.5"


def _preprocessing_config(config: StudyConfig, record: dict, output: Path, **references):
    from .preprocessing.config import PreprocessingConfig

    return PreprocessingConfig(
        input_dir=Path(record["input_dir"]), output_dir=output,
        study_uid=str(record["study_uid"]),
        min_series_slices=config.min_series_slices,
        dcm2niix_path=config.dcm2niix_path,
        mni_template=config.mni_template,
        synthstrip_backend=config.synthstrip_backend,
        synthseg_backend=config.synthseg_backend,
        synthstrip_image=config.synthstrip_image,
        synthseg_image=config.synthseg_image,
        ants_image=config.ants_image,
        **references,
    )


def _preprocess_fold(config: StudyConfig, cohort: pd.DataFrame, fold: int, train_ids: set[str], root: Path):
    from .preprocessing.pipeline import run_patient_preprocessing

    training = cohort[cohort.patient_id.isin(train_ids)].sort_values("exam_id")
    reference_record = training.iloc[0].to_dict()
    reference_root = root / "reference_subject"
    run_patient_preprocessing(_preprocessing_config(
        config, reference_record, reference_root, allow_self_reference=True,
    ))
    references = {
        "t2_reference": reference_root / "06_brain_extracted" / "T2.nii.gz",
        "flair_reference": reference_root / "06_brain_extracted" / "FLAIR.nii.gz",
        "t2_reference_mask": reference_root / "06_brain_extracted" / "brain_mask.nii.gz",
        "flair_reference_mask": reference_root / "06_brain_extracted" / "brain_mask.nii.gz",
    }
    write_json(root / "training_reference.json", {
        "fold": fold, "reference_exam_id": reference_record["exam_id"],
        "training_patient_count": len(train_ids),
        "references": {key: str(value) for key, value in references.items()},
    })
    output_rows = []
    for record in cohort.to_dict(orient="records"):
        exam_root = root / "exams" / record["exam_id"]
        outputs = run_patient_preprocessing(_preprocessing_config(config, record, exam_root, **references))
        output_rows.append({
            "patient_id": record["patient_id"], "exam_id": record["exam_id"],
            "label": record["label"],
            **{name.lower(): str(outputs[name]) for name in ("T1", "T2", "FLAIR")},
            "brain_mask": str(outputs["brain_mask"]), "roi_mask": str(outputs["regions_40"]),
        })
    return pd.DataFrame(output_rows)


def _build_features_and_patches(exams: pd.DataFrame, root: Path) -> tuple[Path, Path]:
    from .deep_learning.patches import DeepPatchConfig, extract_roi_patches, save_deep_patch_tensor
    from .radiomics.handcrafted import extract_multimodal_handcrafted_features

    patch_config = DeepPatchConfig()
    radiomics_rows = []
    patch_rows = []
    for exam in exams.itertuples(index=False):
        images = {"T1": exam.t1, "T2": exam.t2, "FLAIR": exam.flair}
        features, metadata = extract_multimodal_handcrafted_features(
            images, exam.roi_mask, exam.brain_mask, exam.patient_id,
        )
        feature_row = features.iloc[0].to_dict()
        feature_row.update({"exam_id": exam.exam_id, "label": exam.label})
        radiomics_rows.append(feature_row)
        write_json(root / "radiomics_metadata" / f"{exam.exam_id}.json", metadata)
        patches, patch_metadata = extract_roi_patches(
            exam.t1, exam.t2, exam.flair, exam.roi_mask, config=patch_config,
        )
        patch_metadata["patch_config"] = {
            "slice_count": patch_config.slice_count, "image_size": patch_config.image_size,
            "patch_depth": patch_config.patch_depth, "patch_height": patch_config.patch_height,
            "patch_width": patch_config.patch_width, "roi_ids": list(patch_config.roi_ids),
        }
        patch_metadata["source_volumes"] = {
            "t1": exam.t1, "t2": exam.t2, "flair": exam.flair, "roi_mask": exam.roi_mask,
        }
        patch_path, _ = save_deep_patch_tensor(
            root / "patches" / f"{exam.exam_id}.pt", patches, patch_metadata,
            patient_id=exam.patient_id, label=exam.label,
        )
        patch_rows.append({
            "patient_id": exam.patient_id, "exam_id": exam.exam_id,
            "label": exam.label, "patch_path": str(patch_path),
        })
    radiomics_path = root / "radiomics.csv"
    manifest_path = root / "patch_manifest.csv"
    write_dataframe(radiomics_path, pd.DataFrame(radiomics_rows))
    write_dataframe(manifest_path, pd.DataFrame(patch_rows))
    return radiomics_path, manifest_path


def _split_deep_manifests(manifest: pd.DataFrame, assignments: pd.DataFrame, fold: int, root: Path,
                          seed: int, inner_folds: int):
    from sklearn.model_selection import StratifiedKFold

    patients = assignments[["patient_id", "label", "fold"]].drop_duplicates("patient_id")
    train_pool = patients[patients.fold != fold].reset_index(drop=True)
    test_ids = set(patients.loc[patients.fold == fold, "patient_id"])
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    train_idx, val_idx = next(inner.split(train_pool.patient_id, train_pool.label))
    train_ids = set(train_pool.iloc[train_idx].patient_id)
    val_ids = set(train_pool.iloc[val_idx].patient_id)
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Patient-level partitions overlap")
    paths = {}
    for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids),
                      ("all", set(patients.patient_id))):
        path = root / f"{name}_manifest.csv"
        write_dataframe(path, manifest[manifest.patient_id.isin(ids)].copy())
        paths[name] = path
    return paths


def _train_backbone(config: StudyConfig, backbone: str, paths: dict[str, Path], root: Path):
    from torch.utils.data import DataLoader

    from .deep_learning.dataset import build_dataset_class, collate_deep_patch_batch, samples_from_manifest
    from .deep_learning.features import extract_deep_features_from_manifest
    from .deep_learning.finetuning import available_strategies
    from .deep_learning.train import (
        DeepTrainingConfig, build_configured_model,
        evaluate_held_out_patients, train_deep_attention_model,
    )

    checkpoint = {
        "resnet18": config.medicalnet18, "resnet50": config.medicalnet50,
        "densenet121": config.radimagenet121,
    }[backbone]
    train_samples, label_map = samples_from_manifest(paths["train"])
    val_samples, _ = samples_from_manifest(paths["val"], label_map=label_map)
    test_samples, _ = samples_from_manifest(paths["test"], label_map=label_map)
    if len(label_map) != 5:
        raise ValueError("Every deep-learning training fold requires all five classes")
    Dataset = build_dataset_class()
    loaders = {
        "train": DataLoader(Dataset(train_samples, augment=True), batch_size=1, shuffle=True,
                            collate_fn=collate_deep_patch_batch),
        "val": DataLoader(Dataset(val_samples), batch_size=1, shuffle=False,
                          collate_fn=collate_deep_patch_batch),
        "test": DataLoader(Dataset(test_samples), batch_size=1, shuffle=False,
                           collate_fn=collate_deep_patch_batch),
    }
    results = []
    chosen = None
    for strategy in available_strategies(backbone):
        strategy_root = root / strategy
        training = DeepTrainingConfig(
            backbone=backbone, fine_tuning=strategy, num_classes=len(label_map),
            epochs=config.epochs, batch_size=16 if backbone == "resnet18" else 8,
            imbalance_mode="both", seed=config.seed,
            medicalnet_checkpoint=str(checkpoint) if backbone.startswith("resnet") else None,
            radimagenet_checkpoint=str(checkpoint) if backbone == "densenet121" else None,
            output_dir=strategy_root,
        )
        model, freeze_summary = build_configured_model(training)
        result = train_deep_attention_model(model, loaders["train"], loaders["val"], training)
        if not result.checkpoint_path:
            raise RuntimeError("Validation did not save a checkpoint")
        row = {"backbone": backbone, "strategy": strategy, "validation_accuracy": result.best_metric,
               "validation_loss": result.best_loss, "best_epoch": result.best_epoch,
               "checkpoint": result.checkpoint_path, **freeze_summary}
        results.append(row)
        if chosen is None or (result.best_metric, -result.best_loss) > (chosen["validation_accuracy"], -chosen["validation_loss"]):
            chosen = row
    write_dataframe(root / "fine_tuning_results.csv", pd.DataFrame(results))
    selected_strategy = chosen["strategy"]
    selected_checkpoint = chosen["checkpoint"]
    selected_model, _ = build_configured_model(DeepTrainingConfig(
        backbone=backbone, fine_tuning=selected_strategy, num_classes=len(label_map),
        medicalnet_checkpoint=str(checkpoint) if backbone.startswith("resnet") else None,
        radimagenet_checkpoint=str(checkpoint) if backbone == "densenet121" else None,
    ))
    evaluate_held_out_patients(selected_model, loaders["test"], selected_checkpoint, root / "held_out")
    if backbone == "densenet121":
        _run_imbalance_ablation(config, backbone, selected_strategy, checkpoint,
                                train_samples, val_samples, root / "imbalance_ablation")
    features = extract_deep_features_from_manifest(
        paths["all"], root / "features", backbone, selected_strategy,
        selected_checkpoint, num_classes=len(label_map), seed=config.seed,
    )
    write_json(root / "selection.json", {"selected": chosen, "label_map": label_map,
                                         "test_predictions": str(root / "held_out" / "held_out_patient_predictions.csv")})
    return features["features"], root / "held_out" / "held_out_patient_predictions.csv"


def _run_imbalance_ablation(config, backbone, strategy, checkpoint, train_samples, val_samples, root):
    from torch.utils.data import DataLoader

    from .deep_learning.dataset import build_dataset_class, collate_deep_patch_batch
    from .deep_learning.train import DeepTrainingConfig, build_configured_model, train_deep_attention_model

    Dataset = build_dataset_class()
    val_loader = DataLoader(Dataset(val_samples), batch_size=1, shuffle=False,
                            collate_fn=collate_deep_patch_batch)
    rows = []
    for mode in ("baseline", "augmentation", "weighted_loss", "both"):
        train_loader = DataLoader(Dataset(train_samples, augment=mode in {"augmentation", "both"}),
                                  batch_size=1, shuffle=True, collate_fn=collate_deep_patch_batch)
        training = DeepTrainingConfig(
            backbone=backbone, fine_tuning=strategy, num_classes=5, epochs=config.epochs,
            batch_size=8, imbalance_mode=mode, seed=config.seed,
            radimagenet_checkpoint=str(checkpoint), output_dir=root / mode,
        )
        model, _ = build_configured_model(training)
        result = train_deep_attention_model(model, train_loader, val_loader, training)
        rows.append({"mode": mode, "validation_accuracy": result.best_metric,
                     "validation_loss": result.best_loss, "checkpoint": result.checkpoint_path})
    write_dataframe(root / "validation_results.csv", pd.DataFrame(rows))


def run_study(config: StudyConfig) -> dict[str, Path]:
    """Run discovery, fold-local preprocessing, three model branches and evaluation."""
    from .dataset.cohort import discover_cohort
    from .dataset.splits import stratified_kfold_assignments, summarize_folds
    from .evaluation.study import summarize_study
    from .hybrid.fusion import merge_feature_tables
    from .traditional_ml import TraditionalMLConfig, run_traditional_ml_experiment

    for checkpoint in (config.medicalnet18, config.medicalnet50, config.radimagenet121):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    root = ensure_dir(config.output_dir)
    cohort, excluded = discover_cohort(config.data_root, config.min_series_slices)
    write_dataframe(root / "cohort.csv", cohort)
    write_dataframe(root / "excluded_studies.csv", excluded)
    assignments = stratified_kfold_assignments(
        cohort[["patient_id", "label"]].drop_duplicates(),
        label_column="label", n_splits=config.folds, seed=config.seed,
    )
    assignments = assignments[["patient_id", "label", "fold"]]
    assignment_path = root / "fold_assignments.csv"
    write_dataframe(assignment_path, assignments)
    write_dataframe(root / "fold_summary.csv", summarize_folds(assignments, label_column="label"))
    radiomics_dir = ensure_dir(root / "features" / "radiomics")
    deep_dirs = {name: ensure_dir(root / "features" / "deep" / name) for name in BACKBONES}
    fused_dirs = {name: ensure_dir(root / "features" / "fused" / name) for name in BACKBONES}
    deep_predictions: dict[str, list[Path]] = {name: [] for name in BACKBONES}
    for fold in sorted(assignments.fold.unique()):
        fold = int(fold)
        fold_root = ensure_dir(root / "folds" / f"fold_{fold}")
        train_ids = set(assignments.loc[assignments.fold != fold, "patient_id"])
        exams = _preprocess_fold(config, cohort, fold, train_ids, fold_root / "preprocessing")
        radiomics_path, manifest_path = _build_features_and_patches(exams, fold_root / "features")
        radiomics = pd.read_csv(radiomics_path)
        write_dataframe(radiomics_dir / f"fold_{fold}.csv", radiomics)
        manifests = _split_deep_manifests(pd.read_csv(manifest_path), assignments,
                                          fold, fold_root / "manifests", config.seed, config.inner_folds)
        for backbone in BACKBONES:
            deep_path, prediction_path = _train_backbone(config, backbone, manifests,
                                                          fold_root / "deep" / backbone)
            deep = pd.read_csv(deep_path)
            write_dataframe(deep_dirs[backbone] / f"fold_{fold}.csv", deep)
            write_dataframe(fused_dirs[backbone] / f"fold_{fold}.csv",
                            merge_feature_tables(radiomics, deep))
            deep_predictions[backbone].append(prediction_path)
    common = dict(output_dir=root / "models" / "radiomics", fold_assignments=assignment_path,
                  outer_folds=config.folds, inner_folds=config.inner_folds,
                  random_state=config.seed, n_jobs=config.n_jobs)
    radiomics_outputs = run_traditional_ml_experiment(TraditionalMLConfig(
        features_csv=radiomics_dir / "fold_0.csv", fold_features_dir=radiomics_dir, **common,
    ))
    model_outputs = {"radiomics": radiomics_outputs}
    for backbone in BACKBONES:
        for mode, feature_dir, feature_sizes in (
            ("deep", deep_dirs[backbone], tuple(range(20, 401, 20))),
            ("fused", fused_dirs[backbone], tuple(range(20, 401, 20))),
        ):
            model_outputs[f"{backbone}_{mode}"] = run_traditional_ml_experiment(TraditionalMLConfig(
                features_csv=feature_dir / "fold_0.csv", output_dir=root / "models" / backbone / mode,
                fold_assignments=assignment_path, fold_features_dir=feature_dir,
                outer_folds=config.folds, inner_folds=config.inner_folds,
                random_state=config.seed, n_jobs=config.n_jobs,
                mode=mode, selection_methods=("mrmr",),
                feature_sizes=feature_sizes, classifiers=CLASSIFIERS,
            ))
    evaluation = summarize_study(model_outputs, deep_predictions, root / "evaluation", config.folds)
    write_json(root / "study_manifest.json", {
        "cohort": str(root / "cohort.csv"), "fold_assignments": str(assignment_path),
        "feature_directories": {"radiomics": str(radiomics_dir),
                                "deep": {key: str(value) for key, value in deep_dirs.items()},
                                "fused": {key: str(value) for key, value in fused_dirs.items()}},
        "evaluation": {key: str(value) for key, value in evaluation.items()},
    })
    return evaluation
