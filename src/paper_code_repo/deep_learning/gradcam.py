"""DenseNet121 ROI Grad-CAM for held-out patient examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_densenet_gradcam(checkpoint_path: str | Path, patch_path: str | Path, output_prefix: str | Path,
                          target_class: int | None = None) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as functional

    from .models import build_roi_attention_model

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("config", {}).get("backbone", "").lower() != "densenet121":
        raise ValueError("Grad-CAM is configured for the study's best DenseNet121 model")
    model = build_roi_attention_model("densenet121", num_classes=checkpoint["config"]["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    payload = torch.load(patch_path, map_location="cpu", weights_only=False)
    patches = payload["patches"].float()
    if patches.ndim != 5 or patches.shape[0] != 40 or patches.shape[1] != 3:
        raise ValueError("Expected 40 T1/T2/FLAIR ROI patches with shape [40,3,D,H,W]")
    with torch.no_grad():
        base_embeddings = torch.stack([model.backbone(patches[i:i + 1])[0] for i in range(40)], dim=0).unsqueeze(0)
        patient_embedding, _ = model.attention(base_embeddings)
        base_logits = model.classifier(patient_embedding)
    class_index = int(base_logits.argmax(dim=1).item()) if target_class is None else int(target_class)
    if not 0 <= class_index < base_logits.shape[1]:
        raise ValueError("target_class is outside the trained label range")
    maps = []
    for roi_index in range(40):
        activations = {}
        gradients = {}

        def capture(_module, _inputs, output):
            activations["value"] = output
            output.register_hook(lambda grad: gradients.__setitem__("value", grad))

        hook = model.backbone.features.denseblock4.register_forward_hook(capture)
        try:
            model.zero_grad(set_to_none=True)
            fresh = model.backbone(patches[roi_index:roi_index + 1]).unsqueeze(1)
            embeddings = torch.cat((base_embeddings[:, :roi_index], fresh, base_embeddings[:, roi_index + 1:]), dim=1)
            pooled, _ = model.attention(embeddings)
            logits = model.classifier(pooled)
            logits[0, class_index].backward()
            weights = gradients["value"].mean(dim=(2, 3, 4), keepdim=True)
            heatmap = torch.relu((weights * activations["value"]).sum(dim=1, keepdim=True))
            heatmap = functional.interpolate(heatmap, size=patches.shape[-3:], mode="trilinear", align_corners=False)
            maps.append(heatmap[0, 0].detach().numpy())
        finally:
            hook.remove()
    cam = np.stack(maps)
    cam /= np.maximum(cam.max(axis=(1, 2, 3), keepdims=True), 1e-8)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    array_path = prefix.with_suffix(".npz")
    figure_path = prefix.with_suffix(".png")
    np.savez_compressed(array_path, heatmaps=cam, target_class=class_index,
                        patient_id=payload.get("patient_id", "unknown"))
    figure, axes = plt.subplots(5, 8, figsize=(24, 15))
    middle = patches.shape[2] // 2
    for index, axis in enumerate(axes.flat):
        axis.imshow(patches[index, 2, middle].numpy(), cmap="gray")
        axis.imshow(cam[index, middle], cmap="jet", alpha=0.4, vmin=0, vmax=1)
        axis.set_title(f"ROI {index + 1}")
        axis.axis("off")
    figure.suptitle(f"DenseNet121 Grad-CAM, class {class_index}")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return array_path, figure_path
