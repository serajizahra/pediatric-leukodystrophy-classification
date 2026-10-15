from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import nibabel as nib

from ..utils.files import ensure_dir, write_json
from .config import PreprocessingConfig
from .external import run_command, run_synthseg, run_synthstrip
from .image_ops import (
    apply_brain_mask,
    histogram_match_masked,
    image_metadata,
    materialize_mni152_template,
    reorient_to_ras,
    resample_isotropic,
    rigid_register_modalities_to_t1,
    rician_nlmeans_denoise,
    rigid_register_to_template,
    whitestripe_t1,
    zscore_within_mask,
)
from .inventory import DicomSeries, scan_dicom_series, select_modalities
from .qc import save_montage
from .rois import remap_synthseg_to_40_rois


def _resolve_dcm2niix(config: PreprocessingConfig) -> Path:
    candidates: list[Path] = []
    if config.dcm2niix_path is not None:
        candidates.append(config.dcm2niix_path)
    command = shutil.which("dcm2niix")
    if command:
        candidates.append(Path(command))
    repository_root = Path(__file__).resolve().parents[3]
    candidates.append(repository_root / ".tools" / "dcm2niix" / "dcm2niix.exe")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise RuntimeError(
        "dcm2niix was not found. Run scripts/setup_preprocessing.ps1 or pass --dcm2niix-path."
    )


def _run_dcm2niix(
    config: PreprocessingConfig,
    selected: dict[str, DicomSeries],
    output_dir: Path,
    log_dir: Path,
) -> dict[str, Path]:
    all_dir = ensure_dir(output_dir / "all_series")
    expected = {modality: output_dir / f"{modality}.nii.gz" for modality in selected}
    if all(path.exists() for path in expected.values()) and not config.force:
        return expected

    executable = _resolve_dcm2niix(config)
    with tempfile.TemporaryDirectory(prefix="selected_dicom_", dir=output_dir) as temporary:
        selected_dir = Path(temporary)
        for modality, series in selected.items():
            for index, source in enumerate(series.files):
                destination = selected_dir / f"{modality}_{index:06d}.dcm"
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
        command = [
            str(executable), "-z", "y", "-b", "y", "-ba", "y",
            "-f", "%j", "-o", str(all_dir), str(selected_dir),
        ]
        run_command(command, log_dir / "dcm2niix.json")

    outputs: dict[str, Path] = {}
    for modality, series in selected.items():
        candidates = sorted(all_dir.glob(f"{series.uid}*.nii*"), key=lambda path: path.stat().st_size, reverse=True)
        if not candidates:
            raise RuntimeError(f"dcm2niix did not create an output for selected {modality} series {series.uid}")
        source = candidates[0]
        destination = output_dir / f"{modality}.nii.gz"
        shutil.copy2(source, destination)
        sidecar = source.with_suffix("").with_suffix(".json") if source.name.endswith(".nii.gz") else source.with_suffix(".json")
        if sidecar.exists():
            shutil.copy2(sidecar, output_dir / f"{modality}.json")
        outputs[modality] = destination

    if not config.keep_all_converted_series:
        shutil.rmtree(all_dir, ignore_errors=True)
    return outputs


def _versions(config: PreprocessingConfig) -> dict[str, object]:
    versions: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    versions["ANTs"] = config.ants_image
    for module_name in ("nibabel", "numpy", "pydicom", "dipy", "SimpleITK", "nilearn"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[module_name] = f"unavailable: {exc}"
    try:
        result = subprocess.run(
            [str(_resolve_dcm2niix(config)), "--version"], capture_output=True, text=True, check=False
        )
        versions["dcm2niix"] = (result.stdout or result.stderr).strip()
    except Exception as exc:
        versions["dcm2niix"] = f"unavailable: {exc}"
    return versions


def _record_images(paths: dict[str, Path]) -> dict[str, object]:
    return {name: image_metadata(path) for name, path in paths.items()}


def _require_references(config: PreprocessingConfig) -> None:
    if config.allow_self_reference:
        return
    missing = []
    if config.t2_reference is None:
        missing.append("--t2-reference")
    if config.flair_reference is None:
        missing.append("--flair-reference")
    if missing:
        raise ValueError(
            "Histogram matching needs same-modality training references: " + ", ".join(missing)
            + ". Use --allow-self-reference only for a single-subject installation check."
        )


def run_patient_preprocessing(config: PreprocessingConfig) -> dict[str, Path]:
    if not config.input_dir.exists():
        raise FileNotFoundError(config.input_dir)
    _require_references(config)
    root = ensure_dir(config.output_dir)
    logs = ensure_dir(root / "logs")
    qc_dir = ensure_dir(root / "qc")
    provenance_path = root / "provenance.json"
    provenance: dict[str, object] = {
        "config": config.to_dict(),
        "versions": _versions(config),
        "started_at_unix": time.time(),
        "steps": [],
        "warnings": [],
    }
    if config.allow_self_reference:
        provenance["warnings"].append(
            "T2/FLAIR histogram matching used each image as its own reference for a single-subject installation check."
        )

    def record(name: str, started: float, outputs: dict[str, Path], extra: dict[str, object] | None = None) -> None:
        step = {
            "name": name,
            "duration_seconds": round(time.time() - started, 3),
            "outputs": _record_images(outputs),
        }
        if extra:
            step.update(extra)
        provenance["steps"].append(step)
        write_json(provenance_path, provenance)

    started = time.time()
    series = scan_dicom_series(config.input_dir, study_uid=config.study_uid)
    selected = select_modalities(series, min_slices=config.min_series_slices)
    inventory_dir = ensure_dir(root / "00_inventory")
    write_json(inventory_dir / "series_inventory.json", [item.public_dict() for item in series])
    write_json(
        inventory_dir / "selected_series.json",
        {modality: item.public_dict() for modality, item in selected.items()},
    )
    provenance["selected_series"] = {modality: item.public_dict() for modality, item in selected.items()}
    provenance["steps"].append(
        {"name": "series_selection", "duration_seconds": round(time.time() - started, 3)}
    )
    write_json(provenance_path, provenance)

    started = time.time()
    nifti = _run_dcm2niix(config, selected, ensure_dir(root / "01_nifti"), logs)
    record("dicom_to_nifti", started, nifti)

    started = time.time()
    ras = {
        modality: reorient_to_ras(path, root / "02_ras" / f"{modality}.nii.gz")
        for modality, path in nifti.items()
    }
    record("reorient_ras", started, ras)

    started = time.time()
    resampled = {
        modality: resample_isotropic(
            path, root / "03_resampled_1mm" / f"{modality}.nii.gz", config.target_spacing
        )
        for modality, path in ras.items()
    }
    record("resample_1mm", started, resampled)

    started = time.time()
    coregistered, modality_transforms = rigid_register_modalities_to_t1(
        resampled["T1"], resampled, root / "03_coregistered_t1", config.ants_image
    )
    record(
        "rigid_registration_t2_flair_to_t1",
        started,
        coregistered,
        {"transforms": [str(path) for path in modality_transforms]},
    )

    template = config.mni_template
    if template is None:
        template = materialize_mni152_template(root / "resources" / "MNI152_T1_1mm.nii.gz")
    started = time.time()
    registered, transforms = rigid_register_to_template(
        coregistered["T1"], coregistered, template, root / "04_registered_mni", config.ants_image
    )
    record(
        "rigid_registration_to_mni",
        started,
        registered,
        {"transforms": [str(path) for path in transforms], "template": str(template)},
    )

    started = time.time()
    denoised: dict[str, Path] = {}
    sigmas: dict[str, float] = {}
    for modality, path in registered.items():
        output, sigma = rician_nlmeans_denoise(path, root / "05_denoised" / f"{modality}.nii.gz")
        denoised[modality] = output
        sigmas[modality] = sigma
    record("rician_non_local_means", started, denoised, {"estimated_sigma": sigmas})

    started = time.time()
    stripped_dir = ensure_dir(root / "06_brain_extracted")
    t1_brain = stripped_dir / "T1.nii.gz"
    brain_mask = stripped_dir / "brain_mask.nii.gz"
    run_synthstrip(
        denoised["T1"],
        t1_brain,
        brain_mask,
        root,
        config.synthstrip_backend,
        config.synthstrip_image,
        logs / "synthstrip.json",
    )
    stripped = {"T1": t1_brain}
    for modality in ("T2", "FLAIR"):
        stripped[modality] = apply_brain_mask(
            denoised[modality], brain_mask, stripped_dir / f"{modality}.nii.gz"
        )
    record("synthstrip_and_mask_propagation", started, {**stripped, "brain_mask": brain_mask})

    started = time.time()
    normalized_dir = ensure_dir(root / "07_normalized")
    t1_white = whitestripe_t1(stripped["T1"], brain_mask, normalized_dir / "T1_whitestripe.nii.gz")
    t2_reference = stripped["T2"] if config.allow_self_reference else config.t2_reference
    flair_reference = stripped["FLAIR"] if config.allow_self_reference else config.flair_reference
    matched_t2 = histogram_match_masked(
        stripped["T2"], brain_mask, t2_reference, normalized_dir / "T2_histogram_matched.nii.gz", config.t2_reference_mask
    )
    matched_flair = histogram_match_masked(
        stripped["FLAIR"], brain_mask, flair_reference, normalized_dir / "FLAIR_histogram_matched.nii.gz", config.flair_reference_mask
    )
    normalized = {
        "T1": zscore_within_mask(t1_white, brain_mask, normalized_dir / "T1.nii.gz"),
        "T2": zscore_within_mask(matched_t2, brain_mask, normalized_dir / "T2.nii.gz"),
        "FLAIR": zscore_within_mask(matched_flair, brain_mask, normalized_dir / "FLAIR.nii.gz"),
    }
    record(
        "modality_normalization_and_zscore",
        started,
        normalized,
        {"t2_reference": str(t2_reference), "flair_reference": str(flair_reference)},
    )

    started = time.time()
    segmentation_dir = ensure_dir(root / "08_segmentation")
    synthseg = segmentation_dir / "synthseg_parc.nii.gz"
    run_synthseg(
        normalized["T1"],
        synthseg,
        root,
        config.synthseg_backend,
        config.synthseg_image,
        logs / "synthseg.json",
    )
    record("synthseg_parcellation", started, {"synthseg_parc": synthseg})

    started = time.time()
    roi_dir = ensure_dir(root / "09_rois")
    roi_path = remap_synthseg_to_40_rois(
        synthseg, roi_dir / "regions_40.nii.gz", roi_dir / "regions_40.csv"
    )
    record("merge_to_40_regions", started, {"regions_40": roi_path})

    for stage_name, paths in (
        ("01_nifti", nifti),
        ("03_resampled", resampled),
        ("03_coregistered_t1", coregistered),
        ("04_registered", registered),
        ("05_denoised", denoised),
        ("06_brain_extracted", stripped),
        ("07_normalized", normalized),
    ):
        for modality, path in paths.items():
            save_montage(path, qc_dir / f"{stage_name}_{modality}.png", f"{stage_name}: {modality}")
    save_montage(roi_path, qc_dir / "09_regions_40.png", "SynthSeg merged to 40 regions")

    final_outputs = {
        "T1": normalized["T1"],
        "T2": normalized["T2"],
        "FLAIR": normalized["FLAIR"],
        "brain_mask": brain_mask,
        "synthseg": synthseg,
        "regions_40": roi_path,
        "provenance": provenance_path,
    }
    provenance["completed_at_unix"] = time.time()
    provenance["final_outputs"] = {key: str(value) for key, value in final_outputs.items()}
    write_json(provenance_path, provenance)
    return final_outputs
