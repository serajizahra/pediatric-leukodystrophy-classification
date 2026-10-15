from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from nibabel.processing import resample_from_to

from .external import run_command


def image_metadata(path: str | Path) -> dict[str, object]:
    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float32)
    finite = data[np.isfinite(data)]
    return {
        "path": str(path),
        "shape": list(image.shape),
        "spacing": [float(value) for value in image.header.get_zooms()[:3]],
        "orientation": list(nib.aff2axcodes(image.affine)),
        "minimum": float(finite.min()) if finite.size else None,
        "maximum": float(finite.max()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
        "nonzero_voxels": int(np.count_nonzero(data)),
    }


def reorient_to_ras(input_path: str | Path, output_path: str | Path) -> Path:
    image = nib.load(str(input_path))
    canonical = nib.as_closest_canonical(image)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(canonical, str(output))
    return output


def resample_isotropic(
    input_path: str | Path,
    output_path: str | Path,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Path:
    import SimpleITK as sitk

    image = sitk.ReadImage(str(input_path))
    old_spacing = image.GetSpacing()
    size = [max(1, int(round(n * old / new))) for n, old, new in zip(image.GetSize(), old_spacing, spacing)]
    resampled = sitk.Resample(
        image, size, sitk.Transform(), sitk.sitkLanczosWindowedSinc,
        image.GetOrigin(), spacing, image.GetDirection(), 0.0, image.GetPixelID(),
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(resampled, str(output))
    return output


def materialize_mni152_template(output_path: str | Path) -> Path:
    from nilearn.datasets import load_mni152_template

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    template = load_mni152_template(resolution=1)
    nib.save(template, str(output))
    return output


def rigid_register_modalities_to_t1(
    t1_path: str | Path,
    modality_paths: dict[str, Path],
    output_dir: str | Path,
    ants_image: str = "antsx/ants:v2.6.5",
) -> tuple[dict[str, Path], list[Path]]:
    """Register T2 and FLAIR independently to the native T1 grid."""
    import shutil

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    t1 = Path(t1_path).resolve()
    moving_dirs = {Path(path).resolve().parent for path in modality_paths.values()}
    if len(moving_dirs) != 1 or t1.parent not in moving_dirs:
        raise ValueError("T1, T2 and FLAIR must share one input directory")
    fixed = f"/moving/{t1.name}"
    mounts = [
        "docker", "run", "--rm",
        "-v", f"{t1.parent}:/moving:ro",
        "-v", f"{output_root.resolve()}:/output",
        ants_image,
    ]
    t1_output = output_root / "T1.nii.gz"
    shutil.copy2(t1, t1_output)
    outputs = {"T1": t1_output}
    transforms: list[Path] = []
    for modality in ("T2", "FLAIR"):
        if modality not in modality_paths:
            raise ValueError(f"Missing {modality} image")
        moving = f"/moving/{Path(modality_paths[modality]).name}"
        prefix = f"/output/{modality.lower()}_to_t1_"
        run_command(mounts + [
            "antsRegistration", "--dimensionality", "3", "--float", "1",
            "--output", f"[{prefix},/output/{modality}_preview.nii.gz]",
            "--interpolation", "Linear", "--use-histogram-matching", "0",
            "--initial-moving-transform", f"[{fixed},{moving},1]",
            "--transform", "Rigid[0.1]",
            "--metric", f"Mattes[{fixed},{moving},1,32,Regular,0.25]",
            "--convergence", "[1000x500x250x0,1e-6,10]",
            "--shrink-factors", "8x4x2x1",
            "--smoothing-sigmas", "3x2x1x0vox",
        ], output_root / f"ants_registration_{modality}.json")
        transform = output_root / f"{modality.lower()}_to_t1_0GenericAffine.mat"
        if not transform.is_file():
            raise RuntimeError(f"ANTs registration did not produce the {modality}-to-T1 transform")
        destination = output_root / f"{modality}.nii.gz"
        run_command(mounts + [
            "antsApplyTransforms", "-d", "3", "-i", moving,
            "-r", fixed, "-o", f"/output/{destination.name}",
            "-n", "Linear", "-t", f"/output/{transform.name}",
        ], output_root / f"ants_apply_{modality}.json")
        outputs[modality] = destination
        transforms.append(transform)
    return outputs, transforms


def rigid_register_to_template(
    t1_path: str | Path,
    modality_paths: dict[str, Path],
    template_path: str | Path,
    output_dir: str | Path,
    ants_image: str = "antsx/ants:v2.6.5",
) -> tuple[dict[str, Path], list[Path]]:
    """Run the pinned ANTs 2.6.5 rigid registration in its official container."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    fixed_path = Path(template_path).resolve()
    t1 = Path(t1_path).resolve()
    mounts = ["docker", "run", "--rm", "-v", f"{fixed_path.parent}:/fixed:ro",
              "-v", f"{output_root.resolve()}:/output"]
    moving_dirs = {Path(path).resolve().parent for path in modality_paths.values()}
    if len(moving_dirs) != 1 or t1.parent not in moving_dirs:
        raise ValueError("All reoriented MRI modalities must share the same input directory")
    mounts.extend(["-v", f"{t1.parent}:/moving:ro", ants_image])
    fixed = f"/fixed/{fixed_path.name}"
    moving = f"/moving/{t1.name}"
    prefix = "/output/t1_to_mni_"
    run_command(mounts + [
        "antsRegistration", "--dimensionality", "3", "--float", "1",
        "--output", f"[{prefix},/output/registration_preview.nii.gz]",
        "--interpolation", "Linear", "--use-histogram-matching", "0",
        "--initial-moving-transform", f"[{fixed},{moving},1]",
        "--transform", "Rigid[0.1]",
        "--metric", f"Mattes[{fixed},{moving},1,32,Regular,0.25]",
        "--convergence", "[1000x500x250x0,1e-6,10]",
        "--shrink-factors", "8x4x2x1", "--smoothing-sigmas", "3x2x1x0vox",
    ], output_root / "ants_registration.json")
    transform = output_root / "t1_to_mni_0GenericAffine.mat"
    if not transform.is_file():
        raise RuntimeError("ANTs registration did not produce its rigid affine transform")
    outputs: dict[str, Path] = {}
    for modality, path in modality_paths.items():
        destination = output_root / f"{modality}.nii.gz"
        run_command(mounts + [
            "antsApplyTransforms", "-d", "3", "-i", f"/moving/{Path(path).name}",
            "-r", fixed, "-o", f"/output/{destination.name}", "-n", "Linear",
            "-t", f"/output/{transform.name}",
        ], output_root / f"ants_apply_{modality}.json")
        outputs[modality] = destination
    return outputs, [transform]


def rician_nlmeans_denoise(input_path: str | Path, output_path: str | Path) -> tuple[Path, float]:
    image = nib.load(str(input_path))
    data = image.get_fdata(dtype=np.float32)
    finite = np.isfinite(data)
    positive = data[finite & (data > 0)]
    threshold = float(np.percentile(positive, 1.0)) if positive.size else 0.0
    mask = finite & (data > threshold)
    sigma_values = estimate_sigma(data, N=0)
    sigma = float(np.mean(np.asarray(sigma_values)))
    denoised = nlmeans(data, sigma=sigma, mask=mask, rician=True, patch_radius=2, block_radius=4)
    denoised[~mask] = 0
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(denoised.astype(np.float32), image.affine, image.header), str(output))
    return output, sigma


def apply_brain_mask(input_path: str | Path, mask_path: str | Path, output_path: str | Path) -> Path:
    image = nib.load(str(input_path))
    mask_image = nib.load(str(mask_path))
    data = image.get_fdata(dtype=np.float32)
    mask = mask_image.get_fdata() > 0
    if data.shape != mask.shape:
        raise ValueError(f"Image and mask shapes differ: {data.shape} vs {mask.shape}")
    data[~mask] = 0
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, image.affine, image.header), str(output))
    return output


def whitestripe_t1(input_path: str | Path, mask_path: str | Path, output_path: str | Path) -> Path:
    from nilearn.datasets import load_mni152_gm_mask

    image = nib.load(str(input_path))
    mask = nib.load(str(mask_path))
    data = image.get_fdata(dtype=np.float32)
    brain = mask.get_fdata() > 0
    gm_reference = load_mni152_gm_mask(resolution=1)
    gm = resample_from_to(gm_reference, image, order=0).get_fdata() > 0
    values = data[brain & gm & np.isfinite(data)]
    if values.size < 100:
        raise ValueError("Insufficient gray-matter voxels for T1 WhiteStripe normalization")
    density, edges = np.histogram(values, bins=256)
    mode = (edges[density.argmax()] + edges[density.argmax() + 1]) / 2
    mode_quantile = float(np.mean(values <= mode))
    low, high = np.quantile(values, [max(0.0, mode_quantile - 0.05), min(1.0, mode_quantile + 0.05)])
    stripe = values[(values >= low) & (values <= high)]
    if stripe.size < 2 or float(stripe.std()) == 0:
        raise ValueError("Gray-matter WhiteStripe interval has insufficient variance")
    normalized = np.zeros_like(data, dtype=np.float32)
    normalized[brain] = (data[brain] - float(stripe.mean())) / float(stripe.std())
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(normalized, image.affine, image.header), str(output))
    return output


def histogram_match_masked(
    input_path: str | Path,
    mask_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    reference_mask_path: str | Path | None = None,
    quantiles: int = 1024,
) -> Path:
    image = nib.load(str(input_path))
    data = image.get_fdata(dtype=np.float32)
    mask = nib.load(str(mask_path)).get_fdata() > 0
    reference = nib.load(str(reference_path)).get_fdata(dtype=np.float32)
    if reference_mask_path is not None:
        reference_mask = nib.load(str(reference_mask_path)).get_fdata() > 0
    else:
        reference_mask = np.isfinite(reference) & (reference != 0)

    source_values = data[mask & np.isfinite(data)]
    reference_values = reference[reference_mask & np.isfinite(reference)]
    if source_values.size < 10 or reference_values.size < 10:
        raise ValueError("Histogram matching requires non-empty source and reference masks")

    probabilities = np.linspace(0.0, 1.0, quantiles)
    source_quantiles = np.quantile(source_values, probabilities)
    reference_quantiles = np.quantile(reference_values, probabilities)
    source_quantiles, unique_indices = np.unique(source_quantiles, return_index=True)
    reference_quantiles = reference_quantiles[unique_indices]
    matched = np.zeros_like(data, dtype=np.float32)
    matched[mask] = np.interp(data[mask], source_quantiles, reference_quantiles).astype(np.float32)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(matched, image.affine, image.header), str(output))
    return output


def zscore_within_mask(input_path: str | Path, mask_path: str | Path, output_path: str | Path) -> Path:
    image = nib.load(str(input_path))
    data = image.get_fdata(dtype=np.float32)
    mask = nib.load(str(mask_path)).get_fdata() > 0
    values = data[mask & np.isfinite(data)]
    if values.size == 0:
        raise ValueError("Brain mask contains no valid voxels")
    mean = float(values.mean())
    std = float(values.std())
    if std == 0:
        raise ValueError("Cannot z-score an image with zero variance inside the brain mask")
    normalized = np.zeros_like(data, dtype=np.float32)
    normalized[mask] = (data[mask] - mean) / std
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(normalized, image.affine, image.header), str(output))
    return output

