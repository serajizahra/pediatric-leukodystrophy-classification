from __future__ import annotations

import argparse
from pathlib import Path

from .dataset.discovery import discover_dataset
from .dataset.manifest import save_manifest
from .dataset.splits import stratified_kfold_assignments, summarize_folds
from .utils.files import ensure_dir, write_dataframe, write_json
from .utils.logging import configure_logging


def _cmd_scan(args: argparse.Namespace) -> int:
    records = discover_dataset(args.data_root)
    output_root = ensure_dir(args.output_root)
    csv_path = output_root / "manifest.csv"
    json_path = output_root / "manifest.json"
    frame = save_manifest(records, csv_path=csv_path, json_path=json_path)
    print(f"Saved manifest to {csv_path}")
    print(f"Saved manifest to {json_path}")
    print(f"Discovered {len(frame)} patient folders across {frame['class_label'].nunique()} classes")
    return 0


def _cmd_split(args: argparse.Namespace) -> int:
    import pandas as pd

    manifest = pd.read_csv(args.manifest)
    assignments = stratified_kfold_assignments(
        manifest, label_column=args.label_column, id_column=args.id_column,
        n_splits=args.folds, seed=args.seed,
    )
    write_dataframe(args.output, assignments)
    summary = summarize_folds(assignments, label_column=args.label_column, id_column=args.id_column)
    summary_path = Path(args.output).with_name(Path(args.output).stem + "_summary.csv")
    write_dataframe(summary_path, summary)
    print(f"Saved fold assignments to {args.output}")
    print(f"Saved fold summary to {summary_path}")
    return 0


def _cmd_inspect_series(args: argparse.Namespace) -> int:
    from .preprocessing.inventory import scan_dicom_series, select_modalities

    series = scan_dicom_series(args.input_dir)
    selected = select_modalities(series, min_slices=args.min_slices)
    for modality, item in selected.items():
        print(
            f"{modality}: {item.series_description or item.protocol_name} | "
            f"slices={item.count} | UID={item.uid}"
        )
    if args.output:
        write_json(
            args.output,
            {
                "selected": {name: item.public_dict() for name, item in selected.items()},
                "all_series": [item.public_dict() for item in series],
            },
        )
    return 0


def _cmd_preprocess_patient(args: argparse.Namespace) -> int:
    from .preprocessing.config import PreprocessingConfig
    from .preprocessing.pipeline import run_patient_preprocessing

    config = PreprocessingConfig(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        study_uid=args.study_uid,
        mni_template=Path(args.mni_template) if args.mni_template else None,
        dcm2niix_path=Path(args.dcm2niix_path) if args.dcm2niix_path else None,
        t2_reference=Path(args.t2_reference) if args.t2_reference else None,
        flair_reference=Path(args.flair_reference) if args.flair_reference else None,
        t2_reference_mask=Path(args.t2_reference_mask) if args.t2_reference_mask else None,
        flair_reference_mask=Path(args.flair_reference_mask) if args.flair_reference_mask else None,
        allow_self_reference=args.allow_self_reference,
        synthstrip_backend=args.synthstrip_backend,
        synthseg_backend=args.synthseg_backend,
        synthstrip_image=args.synthstrip_image,
        synthseg_image=args.synthseg_image,
        ants_image=args.ants_image,
        min_series_slices=args.min_slices,
        force=args.force,
        keep_all_converted_series=args.keep_all_converted_series,
    )
    outputs = run_patient_preprocessing(config)
    print("Preprocessing completed:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return 0


def _cmd_preprocessing_doctor(args: argparse.Namespace) -> int:
    import importlib.util
    import shutil

    from .preprocessing.external import docker_ready

    modules = (
        "pydicom",
        "nibabel",
        "scipy",
        "skimage",
        "dipy",
        "SimpleITK",
        "matplotlib",
        "nilearn",
        "pywt",
        "radiomics",
    )
    failed = False
    for module in modules:
        available = importlib.util.find_spec(module) is not None
        print(f"{'OK' if available else 'MISSING'}  Python module: {module}")
        failed = failed or not available
    dcm2niix = shutil.which("dcm2niix")
    local_tool = Path(__file__).resolve().parents[2] / ".tools" / "dcm2niix" / "dcm2niix.exe"
    available_dcm2niix = bool(dcm2niix or local_tool.exists())
    print(f"{'OK' if available_dcm2niix else 'MISSING'}  dcm2niix")
    available_docker = docker_ready()
    print(f"{'OK' if available_docker else 'MISSING'}  Docker engine")
    return 1 if failed or not available_dcm2niix or not available_docker else 0


def _cmd_extract_handcrafted(args: argparse.Namespace) -> int:
    from .radiomics.handcrafted import (
        extract_handcrafted_features_for_image,
        save_handcrafted_features,
        validate_feature_frame,
    )
    from .radiomics.manual_config import HandcraftedFeatureConfig
    from .utils.files import write_json

    roi_ids = None
    if args.roi_ids:
        roi_ids = [int(value) for value in args.roi_ids.split(",") if value.strip()]
    config = HandcraftedFeatureConfig(
        pyradiomics_bin_width=args.bin_width,
        wavelet=args.wavelet,
        minimum_roi_voxels=args.minimum_roi_voxels,
    )
    frame, metadata = extract_handcrafted_features_for_image(
        image_path=args.image,
        roi_mask_path=args.roi_mask,
        brain_mask_path=args.brain_mask,
        modality=args.modality,
        patient_id=args.patient_id,
        config=config,
        roi_ids=roi_ids,
        include_pyradiomics=not args.skip_pyradiomics,
        include_ulbp=not args.skip_ulbp,
        include_gabor=not args.skip_gabor,
        include_time_frequency=not args.skip_time_frequency,
        max_slices=args.max_slices,
    )
    save_handcrafted_features(args.output, frame, metadata, args.metadata)
    validation = validate_feature_frame(frame)
    validation_path = Path(args.output).with_name(Path(args.output).stem + "_validation.json")
    write_json(validation_path, validation)
    print(f"Saved handcrafted features to {args.output}")
    print(f"Saved metadata to {args.metadata or Path(args.output).with_suffix('.json')}")
    print(f"Saved validation to {validation_path}")
    print(validation)
    return 0 if validation["all_numeric_features_finite"] else 1


def _cmd_extract_multimodal_handcrafted(args: argparse.Namespace) -> int:
    from .radiomics.handcrafted import (
        extract_multimodal_handcrafted_features,
        save_handcrafted_features,
        validate_feature_frame,
    )

    frame, metadata = extract_multimodal_handcrafted_features(
        {"T1": args.t1, "T2": args.t2, "FLAIR": args.flair},
        args.roi_mask,
        args.brain_mask,
        patient_id=args.patient_id,
    )
    save_handcrafted_features(args.output, frame, metadata)
    validation = validate_feature_frame(frame)
    write_json(Path(args.output).with_name(Path(args.output).stem + "_validation.json"), validation)
    return 0 if validation["all_numeric_features_finite"] else 1


def _parse_csv_items(value: str, cast=str) -> tuple:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def _cmd_train_traditional(args: argparse.Namespace) -> int:
    from .traditional_ml import TraditionalMLConfig, run_traditional_ml_experiment

    config = TraditionalMLConfig(
        features_csv=Path(args.features),
        output_dir=Path(args.output_dir),
        label_column=args.label_column,
        id_column=args.id_column,
        feature_sizes=_parse_csv_items(args.feature_sizes, int),
        selection_methods=_parse_csv_items(args.selection_methods, str),
        classifiers=_parse_csv_items(args.classifiers, str),
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        scoring=args.scoring,
        random_state=args.seed,
        quick=args.quick,
        n_jobs=args.n_jobs,
        fold_assignments=Path(args.fold_assignments) if args.fold_assignments else None,
        fold_features_dir=Path(args.fold_features_dir) if args.fold_features_dir else None,
    )
    outputs = run_traditional_ml_experiment(config)
    print("Traditional ML experiment completed:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return 0


def _cmd_train_hybrid_ml(args: argparse.Namespace) -> int:
    from .traditional_ml import TraditionalMLConfig, run_traditional_ml_experiment

    config = TraditionalMLConfig(
        features_csv=Path(args.features),
        output_dir=Path(args.output_dir),
        label_column=args.label_column,
        id_column=args.id_column,
        feature_sizes=_parse_csv_items(args.feature_sizes, int),
        selection_methods=("mrmr",),
        classifiers=_parse_csv_items(args.classifiers, str),
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        scoring=args.scoring,
        random_state=args.seed,
        quick=args.quick,
        n_jobs=args.n_jobs,
        mode=args.mode,
        fold_assignments=Path(args.fold_assignments) if args.fold_assignments else None,
        fold_features_dir=Path(args.fold_features_dir) if args.fold_features_dir else None,
    )
    outputs = run_traditional_ml_experiment(config)
    print("Hybrid ML experiment completed:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return 0


def _parse_int_triplet(value: str) -> tuple[int, int, int]:
    parts = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) != 3:
        raise ValueError("Expected three comma-separated integers, e.g. 32,224,224")
    return tuple(parts)  # type: ignore[return-value]


def _cmd_prepare_deep_patches(args: argparse.Namespace) -> int:
    from .deep_learning.patches import DeepPatchConfig, extract_roi_patches, save_deep_patch_tensor

    patch_depth, patch_height, patch_width = _parse_int_triplet(args.patch_size)
    roi_ids = tuple(int(value) for value in args.roi_ids.split(",") if value.strip()) if args.roi_ids else tuple(range(1, 41))
    config = DeepPatchConfig(
        slice_count=args.slice_count,
        image_size=args.image_size,
        patch_depth=patch_depth,
        patch_height=patch_height,
        patch_width=patch_width,
        roi_ids=roi_ids,
    )
    patches, metadata = extract_roi_patches(args.t1, args.t2, args.flair, args.roi_mask, config=config)
    metadata["patch_config"] = {
        "slice_count": config.slice_count, "image_size": config.image_size,
        "patch_depth": config.patch_depth, "patch_height": config.patch_height,
        "patch_width": config.patch_width, "roi_ids": list(config.roi_ids),
    }
    metadata["source_volumes"] = {
        "t1": str(Path(args.t1).resolve()), "t2": str(Path(args.t2).resolve()),
        "flair": str(Path(args.flair).resolve()), "roi_mask": str(Path(args.roi_mask).resolve()),
    }
    tensor_path, metadata_path = save_deep_patch_tensor(
        args.output,
        patches,
        metadata,
        patient_id=args.patient_id,
        label=args.label,
        metadata_path=args.metadata,
    )
    print(f"Saved deep ROI patch tensor to {tensor_path}")
    print(f"Saved metadata to {metadata_path}")
    print(metadata)
    return 0


def _cmd_extract_deep_features(args: argparse.Namespace) -> int:
    from .deep_learning.features import extract_deep_features_from_manifest

    outputs = extract_deep_features_from_manifest(
        manifest_csv=args.manifest,
        output_dir=args.output_dir,
        backbone=args.backbone,
        fine_tuning=args.fine_tuning,
        model_checkpoint=args.model_checkpoint,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        patch_column=args.patch_column,
        label_column=args.label_column,
        id_column=args.id_column,
        seed=args.seed,
    )
    print("Deep feature extraction completed:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return 0


def _cmd_train_deep_attention(args: argparse.Namespace) -> int:
    from torch.utils.data import DataLoader

    from .deep_learning.dataset import build_dataset_class, collate_deep_patch_batch, samples_from_manifest
    from .deep_learning.train import DeepTrainingConfig, build_configured_model, evaluate_held_out_patients, train_deep_attention_model
    from .utils.files import write_json

    train_samples, train_label_map = samples_from_manifest(
        args.train_manifest,
        patch_column=args.patch_column,
        label_column=args.label_column,
        id_column=args.id_column,
    )
    val_samples, val_label_map = samples_from_manifest(
        args.val_manifest,
        patch_column=args.patch_column,
        label_column=args.label_column,
        id_column=args.id_column,
        label_map=train_label_map,
    )
    if set(sample.patient_id for sample in train_samples) & set(sample.patient_id for sample in val_samples):
        raise ValueError("A patient cannot occur in both training and validation manifests")
    test_samples, _ = samples_from_manifest(
        args.test_manifest, patch_column=args.patch_column, label_column=args.label_column,
        id_column=args.id_column, label_map=train_label_map,
    )
    train_ids = {sample.patient_id for sample in train_samples}
    val_ids = {sample.patient_id for sample in val_samples}
    test_ids = {sample.patient_id for sample in test_samples}
    if (train_ids & test_ids) or (val_ids & test_ids):
        raise ValueError("Train, validation, and held-out test patients must be disjoint")
    if args.imbalance_mode in {"augmentation", "both"}:
        from pathlib import Path as _Path

        for sample in train_samples:
            if not _Path(sample.patch_path).is_file():
                raise FileNotFoundError(sample.patch_path)
    Dataset = build_dataset_class()
    train_loader = DataLoader(
        Dataset(train_samples, augment=args.imbalance_mode in {"augmentation", "both"}),
        batch_size=1,
        shuffle=True,
        collate_fn=collate_deep_patch_batch,
    )
    val_loader = DataLoader(
        Dataset(val_samples),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_deep_patch_batch,
    )
    test_loader = DataLoader(
        Dataset(test_samples), batch_size=1,
        shuffle=False, collate_fn=collate_deep_patch_batch,
    )
    config = DeepTrainingConfig(
        backbone=args.backbone,
        fine_tuning=args.fine_tuning,
        num_classes=len(train_label_map),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size or (16 if args.backbone == "resnet18" else 8),
        imbalance_mode=args.imbalance_mode,
        seed=args.seed,
        medicalnet_checkpoint=args.medicalnet_checkpoint,
        radimagenet_checkpoint=args.radimagenet_checkpoint,
        output_dir=Path(args.output_dir),
    )
    model, freeze_summary = build_configured_model(config)
    result = train_deep_attention_model(model, train_loader, val_loader, config)
    if result.checkpoint_path is None:
        raise RuntimeError("The validation phase did not save a checkpoint")
    test_report = evaluate_held_out_patients(model, test_loader, result.checkpoint_path, args.output_dir)
    metadata_path = Path(args.output_dir) / "deep_training_metadata.json"
    write_json(
        metadata_path,
        {
            "backbone": args.backbone,
            "fine_tuning": args.fine_tuning,
            "label_map": train_label_map,
            "freeze_summary": freeze_summary,
            "pretraining": model.pretraining,
            "result": result.__dict__,
            "held_out_test": test_report.__dict__,
            "optimization": {
                "optimizer": "AdamW",
                "betas": [0.9, 0.999],
                "scheduler": {"name": "ReduceLROnPlateau", "factor": 0.5, "patience": 10, "min_lr": 1e-6},
                "weighted_cross_entropy": args.imbalance_mode in {"weighted_loss", "both"},
                "augmentation": args.imbalance_mode in {"augmentation", "both"},
                "dropout": 0.3,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "batch_size": args.batch_size or (16 if args.backbone == "resnet18" else 8),
            },
        },
    )
    print(f"Training completed. Metadata: {metadata_path}")
    print(result)
    return 0


def _cmd_select_deep_finetuning(args: argparse.Namespace) -> int:
    from .deep_learning.selection import summarize_deep_fine_tuning_results

    outputs = summarize_deep_fine_tuning_results(args.results_root, args.output_dir)
    print("Deep fine-tuning selection completed:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return 0


def _cmd_fuse_features(args: argparse.Namespace) -> int:
    from .hybrid.fusion import save_fused_features

    output, metadata = save_fused_features(
        radiomics_csv=args.radiomics,
        deep_features_csv=args.deep_features,
        output_csv=args.output,
        id_column=args.id_column,
        label_column=args.label_column,
        metadata_path=args.metadata,
    )
    print(f"Saved fused features to {output}")
    print(f"Saved metadata to {metadata}")
    return 0


def _cmd_fuse_selected_features(args: argparse.Namespace) -> int:
    from .hybrid.fusion import save_selected_fused_features

    output, metadata = save_selected_fused_features(
        radiomics_csv=args.radiomics,
        deep_features_csv=args.deep_features,
        radiomics_selection_csv=args.radiomics_selection,
        deep_selection_csv=args.deep_selection,
        output_csv=args.output,
        id_column=args.id_column,
        label_column=args.label_column,
        classifier=args.classifier,
        metadata_path=args.metadata,
    )
    print(f"Saved selected-feature fusion table to {output}")
    print(f"Saved metadata to {metadata}")
    return 0


def _cmd_validate_deep_models(args: argparse.Namespace) -> int:
    from .deep_learning.finetuning import apply_fine_tuning_strategy, available_strategies
    from .deep_learning.models import BACKBONE_SPECS, build_roi_attention_model
    from .utils.files import write_json

    import torch

    output_dir = ensure_dir(args.output_dir)
    patch_depth, patch_height, patch_width = _parse_int_triplet(args.patch_size)
    rows = []
    torch.manual_seed(args.seed)
    for backbone in _parse_csv_items(args.backbones, str):
        model = build_roi_attention_model(backbone, in_channels=3, num_classes=args.num_classes)
        model.eval()
        x = torch.randn(args.batch_size, args.roi_count, 3, patch_depth, patch_height, patch_width)
        with torch.no_grad():
            outputs = model(x, return_features=True)
        rows.append(
            {
                "backbone": backbone,
                "logits_shape": list(outputs["logits"].shape),
                "patient_embedding_shape": list(outputs["patient_embedding"].shape),
                "roi_embedding_shape": list(outputs["roi_embeddings"].shape),
                "attention_shape": list(outputs["attention_weights"].shape),
                "expected_embedding_dim": BACKBONE_SPECS[backbone].embedding_dim,
            }
        )
        for strategy in available_strategies(backbone):
            freeze_summary = apply_fine_tuning_strategy(model, backbone, strategy)
            rows.append({"backbone": backbone, "strategy": strategy, **freeze_summary})
    output = Path(output_dir) / "deep_model_validation.json"
    write_json(output, {"model_validation": rows})
    print(f"Saved deep model validation report to {output}")
    return 0


def _cmd_gradcam_densenet(args: argparse.Namespace) -> int:
    from .deep_learning.gradcam import save_densenet_gradcam

    array_path, figure_path = save_densenet_gradcam(
        args.model_checkpoint, args.patch, args.output_prefix, args.target_class,
    )
    print(f"Saved Grad-CAM arrays to {array_path} and figure to {figure_path}")
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    import pandas as pd

    manifest = pd.read_csv(args.manifest)
    print(manifest.groupby("class_label").size().to_string())
    print()
    print(manifest[["dicom_count", "image_count", "nested_folder_count"]].describe().to_string())
    return 0


def _cmd_run_study(args: argparse.Namespace) -> int:
    from .study import StudyConfig, run_study

    config = StudyConfig(
        data_root=Path(args.data_root), output_dir=Path(args.output_dir),
        medicalnet18=Path(args.medicalnet18), medicalnet50=Path(args.medicalnet50),
        radimagenet121=Path(args.radimagenet121), folds=args.folds,
        inner_folds=args.inner_folds, epochs=args.epochs, seed=args.seed,
        n_jobs=args.n_jobs, min_series_slices=args.min_slices,
        dcm2niix_path=Path(args.dcm2niix_path) if args.dcm2niix_path else None,
        mni_template=Path(args.mni_template) if args.mni_template else None,
        synthstrip_backend=args.synthstrip_backend,
        synthseg_backend=args.synthseg_backend,
    )
    outputs = run_study(config)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paper-code-repo", description="Pediatric leukodystrophy MRI workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="Discover dataset folders and build a manifest")
    scan.add_argument("--data-root", required=True)
    scan.add_argument("--output-root", required=True)
    scan.set_defaults(func=_cmd_scan)

    split = subparsers.add_parser("split", help="Create stratified patient-level folds")
    split.add_argument("--manifest", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--folds", type=int, default=5)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--id-column", default="patient_id")
    split.add_argument("--label-column", default="class_label")
    split.set_defaults(func=_cmd_split)

    inspect_series = subparsers.add_parser("inspect-series", help="Select volumetric T1, T2, and FLAIR DICOM series")
    inspect_series.add_argument("--input-dir", required=True)
    inspect_series.add_argument("--output")
    inspect_series.add_argument("--min-slices", type=int, default=20)
    inspect_series.set_defaults(func=_cmd_inspect_series)

    preprocess = subparsers.add_parser("preprocess-patient", help="Run the complete MRI preprocessing pipeline")
    preprocess.add_argument("--input-dir", required=True)
    preprocess.add_argument("--output-dir", required=True)
    preprocess.add_argument("--study-uid", help="Restrict conversion and selection to one DICOM study")
    preprocess.add_argument("--mni-template")
    preprocess.add_argument("--dcm2niix-path")
    preprocess.add_argument("--t2-reference")
    preprocess.add_argument("--flair-reference")
    preprocess.add_argument("--t2-reference-mask")
    preprocess.add_argument("--flair-reference-mask")
    preprocess.add_argument("--allow-self-reference", action="store_true")
    preprocess.add_argument("--synthstrip-backend", choices=("auto", "local", "docker"), default="auto")
    preprocess.add_argument("--synthseg-backend", choices=("auto", "local", "docker"), default="auto")
    preprocess.add_argument("--synthstrip-image", default="freesurfer/freesurfer:8.2.0")
    preprocess.add_argument("--synthseg-image", default="freesurfer/freesurfer:8.2.0")
    preprocess.add_argument("--ants-image", default="antsx/ants:v2.6.5")
    preprocess.add_argument("--min-slices", type=int, default=20)
    preprocess.add_argument("--force", action="store_true")
    preprocess.add_argument("--keep-all-converted-series", action="store_true")
    preprocess.set_defaults(func=_cmd_preprocess_patient)

    doctor = subparsers.add_parser("preprocessing-doctor", help="Check preprocessing dependencies")
    doctor.set_defaults(func=_cmd_preprocessing_doctor)

    handcrafted = subparsers.add_parser(
        "extract-handcrafted",
        help="Extract manual 3D radiomics, ULBP, Gabor, and DCT/SWT/DCST ROI features",
    )
    handcrafted.add_argument("--image", required=True)
    handcrafted.add_argument("--roi-mask", required=True)
    handcrafted.add_argument("--brain-mask", required=True)
    handcrafted.add_argument("--output", required=True)
    handcrafted.add_argument("--metadata")
    handcrafted.add_argument("--patient-id", default="patient")
    handcrafted.add_argument("--modality", default="T1")
    handcrafted.add_argument("--roi-ids", help="Comma-separated ROI IDs. Defaults to all labels 1..40 in the mask.")
    handcrafted.add_argument("--bin-width", type=float, default=25.0)
    handcrafted.add_argument("--wavelet", default="haar")
    handcrafted.add_argument("--minimum-roi-voxels", type=int, default=8)
    handcrafted.add_argument("--max-slices", type=int, help="Optional slice limit for quality-control runs; omit for full-volume features.")
    handcrafted.add_argument("--skip-pyradiomics", action="store_true")
    handcrafted.add_argument("--skip-ulbp", action="store_true")
    handcrafted.add_argument("--skip-gabor", action="store_true")
    handcrafted.add_argument("--skip-time-frequency", action="store_true")
    handcrafted.set_defaults(func=_cmd_extract_handcrafted)

    multimodal = subparsers.add_parser(
        "extract-multimodal-handcrafted",
        help="Extract T1, T2 and FLAIR features into one examination row",
    )
    for modality in ("t1", "t2", "flair"):
        multimodal.add_argument(f"--{modality}", required=True)
    multimodal.add_argument("--roi-mask", required=True)
    multimodal.add_argument("--brain-mask", required=True)
    multimodal.add_argument("--patient-id", required=True)
    multimodal.add_argument("--output", required=True)
    multimodal.set_defaults(func=_cmd_extract_multimodal_handcrafted)

    traditional = subparsers.add_parser(
        "train-traditional",
        help="Run radiomics traditional ML with SMOTE, mRMR/ReliefF/LASSO, and grid search",
    )
    traditional.add_argument("--features", required=True, help="CSV containing patient IDs, labels and numeric features.")
    traditional.add_argument("--output-dir", required=True)
    traditional.add_argument("--label-column", default="label")
    traditional.add_argument("--id-column", default="patient_id")
    traditional.add_argument(
        "--feature-sizes",
        default="60,80,100,120,140,160,180,200,220,240,260,280,300,320",
        help="Comma-separated selected-feature counts. Default: 60..320 in steps of 20.",
    )
    traditional.add_argument("--selection-methods", default="mrmr,relieff,lasso")
    traditional.add_argument("--classifiers", default="svm,knn,rf,xgboost")
    traditional.add_argument("--outer-folds", type=int, default=5)
    traditional.add_argument("--inner-folds", type=int, default=5)
    traditional.add_argument("--scoring", default="accuracy")
    traditional.add_argument("--seed", type=int, default=42)
    traditional.add_argument("--n-jobs", type=int, default=1)
    traditional.add_argument("--fold-assignments", help="Patient-level outer fold assignments shared with deep and hybrid models")
    traditional.add_argument("--fold-features-dir", help="Directory containing fold_<number>.csv examination feature tables")
    traditional.add_argument("--quick", action="store_true", help="Use a reduced parameter grid for a faster check.")
    traditional.set_defaults(func=_cmd_train_traditional)

    hybrid_ml = subparsers.add_parser(
        "train-hybrid-ml",
        help="Run deep-feature or fused hybrid ML with mRMR feature selection and grid-searched classifiers",
    )
    hybrid_ml.add_argument("--features", required=True, help="Deep-only or radiomics+deep fused feature CSV.")
    hybrid_ml.add_argument("--output-dir", required=True)
    hybrid_ml.add_argument("--label-column", default="label")
    hybrid_ml.add_argument("--id-column", default="patient_id")
    hybrid_ml.add_argument(
        "--feature-sizes",
        default="20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400",
        help="Comma-separated deep feature counts. Hybrid default: 20..400 in steps of 20.",
    )
    hybrid_ml.add_argument("--classifiers", default="svm,knn,rf,xgboost")
    hybrid_ml.add_argument("--mode", choices=("deep", "fused"), default="deep")
    hybrid_ml.add_argument("--fold-assignments", required=True, help="Patient-ID-to-fold CSV for independent outer evaluation")
    hybrid_ml.add_argument("--fold-features-dir", required=True, help="Directory containing fold_0.csv ... fold_4.csv from fold-specific CNN checkpoints")
    hybrid_ml.add_argument("--outer-folds", type=int, default=5)
    hybrid_ml.add_argument("--inner-folds", type=int, default=5)
    hybrid_ml.add_argument("--scoring", default="accuracy")
    hybrid_ml.add_argument("--seed", type=int, default=42)
    hybrid_ml.add_argument("--n-jobs", type=int, default=1)
    hybrid_ml.add_argument("--quick", action="store_true", help="Use lightweight parameter grids for reduced-runtime validation.")
    hybrid_ml.set_defaults(func=_cmd_train_hybrid_ml)

    deep_patches = subparsers.add_parser(
        "prepare-deep-patches",
        help="Create 40 ROI-specific 3D T1/T2/FLAIR patch tensors for deep models",
    )
    deep_patches.add_argument("--t1", required=True)
    deep_patches.add_argument("--t2", required=True)
    deep_patches.add_argument("--flair", required=True)
    deep_patches.add_argument("--roi-mask", required=True)
    deep_patches.add_argument("--patient-id", required=True)
    deep_patches.add_argument("--label")
    deep_patches.add_argument("--output", required=True)
    deep_patches.add_argument("--metadata")
    deep_patches.add_argument("--slice-count", type=int, default=32)
    deep_patches.add_argument("--image-size", type=int, default=224)
    deep_patches.add_argument("--patch-size", default="32,224,224", help="depth,height,width")
    deep_patches.add_argument("--roi-ids", help="Comma-separated ROI IDs. Defaults to 1..40.")
    deep_patches.set_defaults(func=_cmd_prepare_deep_patches)

    deep_features = subparsers.add_parser(
        "extract-deep-features",
        help="Extract attention-pooled patient-level deep embeddings from ROI patch tensors",
    )
    deep_features.add_argument("--manifest", required=True)
    deep_features.add_argument("--output-dir", required=True)
    deep_features.add_argument("--backbone", choices=("resnet18", "resnet50", "densenet121"), required=True)
    deep_features.add_argument("--fine-tuning", default="linear_probe")
    deep_features.add_argument("--model-checkpoint", required=True, help="Fine-tuned end-to-end checkpoint for this backbone and strategy")
    deep_features.add_argument("--batch-size", type=int, default=1)
    deep_features.add_argument("--num-classes", type=int, default=5)
    deep_features.add_argument("--patch-column", default="patch_path")
    deep_features.add_argument("--label-column", default="label")
    deep_features.add_argument("--id-column", default="patient_id")
    deep_features.add_argument("--seed", type=int, default=42)
    deep_features.set_defaults(func=_cmd_extract_deep_features)

    deep_train = subparsers.add_parser(
        "train-deep-attention",
        help="Train 3D ROI attention with MedicalNet ResNets or inflated RadImageNet DenseNet121",
    )
    deep_train.add_argument("--train-manifest", required=True)
    deep_train.add_argument("--val-manifest", required=True)
    deep_train.add_argument("--test-manifest", required=True, help="Untouched patient-disjoint outer fold")
    deep_train.add_argument("--output-dir", required=True)
    deep_train.add_argument("--backbone", choices=("resnet18", "resnet50", "densenet121"), required=True)
    deep_train.add_argument("--fine-tuning", required=True)
    deep_train.add_argument(
        "--medicalnet-checkpoint",
        help="Official resnet_18_23dataset.pth or resnet_50_23dataset.pth for the selected ResNet",
    )
    deep_train.add_argument(
        "--radimagenet-checkpoint",
        help="Official PyTorch RadImageNet DenseNet121 2D checkpoint; kernels are inflated to 3D",
    )
    deep_train.add_argument("--epochs", type=int, default=100)
    deep_train.add_argument("--learning-rate", type=float, default=1e-4)
    deep_train.add_argument("--weight-decay", type=float, default=1e-4)
    deep_train.add_argument("--batch-size", type=int, help="Default: 16 for ResNet18, 8 otherwise")
    deep_train.add_argument("--imbalance-mode", choices=("baseline", "augmentation", "weighted_loss", "both"), default="both")
    deep_train.add_argument("--patch-column", default="patch_path")
    deep_train.add_argument("--label-column", default="label")
    deep_train.add_argument("--id-column", default="patient_id")
    deep_train.add_argument("--seed", type=int, default=42)
    deep_train.set_defaults(func=_cmd_train_deep_attention)

    deep_select = subparsers.add_parser(
        "select-deep-finetuning",
        help="Select the best fine-tuning strategy for each backbone from saved deep training metadata",
    )
    deep_select.add_argument("--results-root", required=True)
    deep_select.add_argument("--output-dir", required=True)
    deep_select.set_defaults(func=_cmd_select_deep_finetuning)

    fuse = subparsers.add_parser(
        "fuse-features",
        help="Merge radiomics and deep feature CSV files by patient_id for the fused hybrid branch",
    )
    fuse.add_argument("--radiomics", required=True)
    fuse.add_argument("--deep-features", required=True)
    fuse.add_argument("--output", required=True)
    fuse.add_argument("--metadata")
    fuse.add_argument("--id-column", default="patient_id")
    fuse.add_argument("--label-column", default="label")
    fuse.set_defaults(func=_cmd_fuse_features)

    selected_fuse = subparsers.add_parser(
        "fuse-selected-features",
        help="Fuse selected radiomics features with selected deep features for the second hybrid strategy",
    )
    selected_fuse.add_argument("--radiomics", required=True)
    selected_fuse.add_argument("--deep-features", required=True)
    selected_fuse.add_argument("--radiomics-selection", required=True)
    selected_fuse.add_argument("--deep-selection", required=True)
    selected_fuse.add_argument("--output", required=True)
    selected_fuse.add_argument("--metadata")
    selected_fuse.add_argument("--classifier", help="Optional classifier name for best-by-classifier selection CSVs.")
    selected_fuse.add_argument("--id-column", default="patient_id")
    selected_fuse.add_argument("--label-column", default="label")
    selected_fuse.set_defaults(func=_cmd_fuse_selected_features)

    deep_validation = subparsers.add_parser(
        "validate-deep-models",
        help="Run tensor-shape and fine-tuning checks for 3D CNN backbones",
    )
    deep_validation.add_argument("--output-dir", required=True)
    deep_validation.add_argument("--backbones", default="resnet18,resnet50,densenet121")
    deep_validation.add_argument("--roi-count", type=int, default=4)
    deep_validation.add_argument("--patch-size", default="16,32,32", help="depth,height,width")
    deep_validation.add_argument("--batch-size", type=int, default=1)
    deep_validation.add_argument("--num-classes", type=int, default=5)
    deep_validation.add_argument("--seed", type=int, default=42)
    deep_validation.set_defaults(func=_cmd_validate_deep_models)

    gradcam = subparsers.add_parser("gradcam-densenet", help="Save DenseNet121 ROI Grad-CAM for a held-out case")
    gradcam.add_argument("--model-checkpoint", required=True)
    gradcam.add_argument("--patch", required=True)
    gradcam.add_argument("--output-prefix", required=True)
    gradcam.add_argument("--target-class", type=int, help="Defaults to the predicted class")
    gradcam.set_defaults(func=_cmd_gradcam_densenet)

    summary = subparsers.add_parser("summary", help="Print a quick dataset summary")
    summary.add_argument("--manifest", required=True)
    summary.set_defaults(func=_cmd_summary)

    study = subparsers.add_parser("run-study", help="Run the connected DICOM-to-evaluation study")
    study.add_argument("--data-root", required=True)
    study.add_argument("--output-dir", required=True)
    study.add_argument("--medicalnet18", required=True, help="Official resnet_18_23dataset.pth")
    study.add_argument("--medicalnet50", required=True, help="Official resnet_50_23dataset.pth")
    study.add_argument("--radimagenet121", required=True, help="RadImageNet DenseNet121 2D weights")
    study.add_argument("--folds", type=int, default=5)
    study.add_argument("--inner-folds", type=int, default=5)
    study.add_argument("--epochs", type=int, default=100)
    study.add_argument("--seed", type=int, default=42)
    study.add_argument("--n-jobs", type=int, default=1)
    study.add_argument("--min-slices", type=int, default=20)
    study.add_argument("--dcm2niix-path")
    study.add_argument("--mni-template")
    study.add_argument("--synthstrip-backend", choices=("auto", "local", "docker"), default="auto")
    study.add_argument("--synthseg-backend", choices=("auto", "local", "docker"), default="auto")
    study.set_defaults(func=_cmd_run_study)

    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
