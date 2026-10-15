from pathlib import Path

import importlib.util

import pytest


torch_available = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not torch_available, reason="PyTorch is not installed")
def test_roi_attention_backbones_output_expected_shapes():
    import torch

    from paper_code_repo.deep_learning.finetuning import available_strategies, apply_fine_tuning_strategy
    from paper_code_repo.deep_learning.models import BACKBONE_SPECS, build_roi_attention_model

    for backbone in ("resnet18", "resnet50", "densenet121"):
        model = build_roi_attention_model(backbone, num_classes=5)
        model.eval()
        x = torch.randn(1, 2, 3, 16, 32, 32)
        with torch.no_grad():
            outputs = model(x, return_features=True)
        assert list(outputs["logits"].shape) == [1, 5]
        assert list(outputs["patient_embedding"].shape) == [1, BACKBONE_SPECS[backbone].embedding_dim]
        assert list(outputs["roi_embeddings"].shape) == [1, 2, BACKBONE_SPECS[backbone].embedding_dim]
        assert list(outputs["attention_weights"].shape) == [1, 2]
        assert torch.allclose(outputs["attention_weights"].sum(dim=1), torch.ones(1), atol=1e-6)
        for strategy in available_strategies(backbone):
            summary = apply_fine_tuning_strategy(model, backbone, strategy)
            assert summary["trainable_parameters"] > 0


@pytest.mark.skipif(not torch_available, reason="PyTorch is not installed")
@pytest.mark.skip(reason="Feature extraction requires an external task-specific pretrained checkpoint")
def test_deep_feature_extraction_manifest_validation(tmp_path: Path):
    import pandas as pd
    import torch

    from paper_code_repo.deep_learning.features import extract_deep_features_from_manifest

    rows = []
    for idx, label in enumerate(["ALD", "MLD"]):
        patch_path = tmp_path / f"sample_{idx}.pt"
        torch.save(
            {
                "patches": torch.randn(2, 3, 16, 32, 32),
                "patient_id": f"P{idx}",
                "label": label,
            },
            patch_path,
        )
        rows.append({"patient_id": f"P{idx}", "label": label, "patch_path": str(patch_path)})
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    outputs = extract_deep_features_from_manifest(
        manifest_csv=manifest,
        output_dir=tmp_path / "features",
        backbone="resnet18",
        batch_size=1,
        num_classes=2,
    )
    frame = pd.read_csv(outputs["features"])
    assert len(frame) == 2
    assert "patient_id" in frame.columns
    assert sum(col.startswith("resnet18_deep_") for col in frame.columns) == 512


def test_selected_feature_fusion_uses_only_selected_columns(tmp_path: Path):
    import pandas as pd

    from paper_code_repo.hybrid.fusion import save_selected_fused_features

    radiomics = pd.DataFrame(
        {
            "patient_id": ["P1", "P2"],
            "label": ["ALD", "MLD"],
            "rad_a": [1.0, 2.0],
            "rad_b": [3.0, 4.0],
            "rad_unused": [5.0, 6.0],
        }
    )
    deep = pd.DataFrame(
        {
            "patient_id": ["P1", "P2"],
            "label": ["ALD", "MLD"],
            "deep_a": [0.1, 0.2],
            "deep_b": [0.3, 0.4],
            "deep_unused": [0.5, 0.6],
        }
    )
    radiomics_csv = tmp_path / "radiomics.csv"
    deep_csv = tmp_path / "deep.csv"
    radiomics_selection = tmp_path / "radiomics_selection.csv"
    deep_selection = tmp_path / "deep_selection.csv"
    radiomics.to_csv(radiomics_csv, index=False)
    deep.to_csv(deep_csv, index=False)
    pd.DataFrame({"rank": [1, 2], "feature_name": ["rad_b", "rad_a"]}).to_csv(radiomics_selection, index=False)
    pd.DataFrame({"rank": [1], "feature_name": ["deep_b"]}).to_csv(deep_selection, index=False)

    output, _ = save_selected_fused_features(
        radiomics_csv=radiomics_csv,
        deep_features_csv=deep_csv,
        radiomics_selection_csv=radiomics_selection,
        deep_selection_csv=deep_selection,
        output_csv=tmp_path / "fused.csv",
    )
    fused = pd.read_csv(output)
    assert list(fused.columns) == [
        "patient_id",
        "label",
        "radiomics__rad_b",
        "radiomics__rad_a",
        "deep__deep_b",
    ]
