from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        return torch, nn, F
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "PyTorch is required for the deep-learning modules. Install the project dependencies first."
        ) from exc


def available_torch() -> bool:
    try:
        _require_torch()
        return True
    except ImportError:
        return False


torch, nn, F = _require_torch()


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    embedding_dim: int
    family: str


BACKBONE_SPECS = {
    "resnet18": BackboneSpec("resnet18", 512, "resnet"),
    "resnet50": BackboneSpec("resnet50", 2048, "resnet"),
    "densenet121": BackboneSpec("densenet121", 1024, "densenet"),
}


def conv3x3x3(in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)


def conv1x1x1(in_planes: int, out_planes: int, stride: int | tuple[int, int, int] = 1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, dilation: int = 1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, dilation: int = 1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride, dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class MedicalNetShortcutA(nn.Module):
    """MedicalNet's average-pooling and zero-padding residual shortcut."""

    def __init__(self, out_channels: int, stride: int):
        super().__init__()
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x):
        out = F.avg_pool3d(x, kernel_size=1, stride=self.stride)
        padding = out.new_zeros(out.shape[0], self.out_channels - out.shape[1], *out.shape[2:])
        return torch.cat((out, padding), dim=1)


class ResNet3DBackbone(nn.Module):
    def __init__(self, block, layers: list[int], in_channels: int = 3, shortcut_type: str = "B"):
        super().__init__()
        self.inplanes = 64
        self.shortcut_type = shortcut_type
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=(2, 2, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], dilation=4)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.embedding_dim = 512 * block.expansion
        self._init_weights()

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1, dilation: int = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.shortcut_type == "A":
                downsample = MedicalNetShortcutA(planes * block.expansion, stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion),
                )
        layers = [block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample)]
        self.inplanes = planes * block.expansion
        layers.extend(block(self.inplanes, planes, dilation=dilation) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class _DenseLayer3D(nn.Module):
    def __init__(self, num_input_features: int, growth_rate: int, bn_size: int = 4, drop_rate: float = 0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm3d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = float(drop_rate)

    def forward(self, prev_features):
        bottleneck = self.conv1(self.relu1(self.norm1(prev_features)))
        new_features = self.conv2(self.relu2(self.norm2(bottleneck)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([prev_features, new_features], 1)


class _DenseBlock3D(nn.Sequential):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float):
        super().__init__()
        for idx in range(num_layers):
            layer = _DenseLayer3D(num_input_features + idx * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module(f"denselayer{idx + 1}", layer)


class _Transition3D(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int):
        super().__init__(
            nn.BatchNorm3d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False),
            nn.AvgPool3d(kernel_size=2, stride=2),
        )


class DenseNet3DBackbone(nn.Module):
    def __init__(
        self,
        growth_rate: int = 32,
        block_config: tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0.0,
        in_channels: int = 3,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, num_init_features, kernel_size=7, stride=(1, 2, 2), padding=3, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        num_features = num_init_features
        for idx, num_layers in enumerate(block_config):
            block = _DenseBlock3D(num_layers, num_features, growth_rate, bn_size, drop_rate)
            self.features.add_module(f"denseblock{idx + 1}", block)
            num_features += num_layers * growth_rate
            if idx != len(block_config) - 1:
                trans = _Transition3D(num_features, num_features // 2)
                self.features.add_module(f"transition{idx + 1}", trans)
                num_features //= 2
        self.features.add_module("norm5", nn.BatchNorm3d(num_features))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.embedding_dim = num_features
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = self.avgpool(out)
        return torch.flatten(out, 1)


class GatedAttentionAggregator(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.attention_v = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, roi_embeddings):
        gates = self.attention_v(roi_embeddings) * self.attention_u(roi_embeddings)
        logits = self.attention_weights(gates).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        patient_embedding = torch.sum(roi_embeddings * weights.unsqueeze(-1), dim=1)
        return patient_embedding, weights


class ROIAttentionClassifier(nn.Module):
    """ROI-based end-to-end classifier and deep feature extractor.

    Input shape: ``(batch, roi, channel, depth, height, width)``.
    """

    def __init__(self, backbone: nn.Module, num_classes: int = 5, attention_hidden_dim: int = 256):
        super().__init__()
        self.backbone = backbone
        self.embedding_dim = int(backbone.embedding_dim)
        self.attention = GatedAttentionAggregator(self.embedding_dim, attention_hidden_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def encode_rois(self, x):
        batch_size, roi_count = x.shape[:2]
        flattened = x.reshape(batch_size * roi_count, *x.shape[2:])
        from torch.utils.checkpoint import checkpoint

        # A logical patient batch contains 40 large 3D patches per patient.
        # Compute one patch at a time to limit volumetric GPU memory use.
        trainable_backbone = any(param.requires_grad for param in self.backbone.parameters())
        parts = []
        for patch in flattened.split(1, dim=0):
            if torch.is_grad_enabled() and trainable_backbone:
                parts.append(checkpoint(self.backbone, patch, use_reentrant=False))
            elif not trainable_backbone:
                with torch.no_grad():
                    parts.append(self.backbone(patch))
            else:
                parts.append(self.backbone(patch))
        embeddings = torch.cat(parts, dim=0)
        return embeddings.reshape(batch_size, roi_count, -1)

    def extract_patient_features(self, x):
        roi_embeddings = self.encode_rois(x)
        patient_embedding, attention_weights = self.attention(roi_embeddings)
        return patient_embedding, roi_embeddings, attention_weights

    def forward(self, x, return_features: bool = False):
        patient_embedding, roi_embeddings, attention_weights = self.extract_patient_features(x)
        logits = self.classifier(self.dropout(patient_embedding))
        if return_features:
            return {
                "logits": logits,
                "patient_embedding": patient_embedding,
                "roi_embeddings": roi_embeddings,
                "attention_weights": attention_weights,
            }
        return logits


def build_backbone(model_name: str, in_channels: int = 3):
    model_name = model_name.lower()
    if model_name == "resnet18":
        return ResNet3DBackbone(BasicBlock3D, [2, 2, 2, 2], in_channels=in_channels, shortcut_type="A")
    if model_name == "resnet50":
        return ResNet3DBackbone(Bottleneck3D, [3, 4, 6, 3], in_channels=in_channels, shortcut_type="B")
    if model_name == "densenet121":
        return DenseNet3DBackbone(in_channels=in_channels)
    raise ValueError(f"Unknown backbone: {model_name}")


def build_roi_attention_model(
    model_name: str,
    in_channels: int = 3,
    num_classes: int = 5,
    attention_hidden_dim: int = 256,
    medicalnet_checkpoint: str | Path | None = None,
    radimagenet_checkpoint: str | Path | None = None,
):
    backbone = build_backbone(model_name, in_channels=in_channels)
    if medicalnet_checkpoint and radimagenet_checkpoint:
        raise ValueError("Provide only the checkpoint for the selected backbone.")
    if model_name.lower() in {"resnet18", "resnet50"}:
        if radimagenet_checkpoint:
            raise ValueError("ResNet18/50 require a MedicalNet checkpoint, not RadImageNet.")
        pretraining = load_medicalnet_checkpoint(backbone, medicalnet_checkpoint, model_name) if medicalnet_checkpoint else None
    else:
        if medicalnet_checkpoint:
            raise ValueError("DenseNet121 requires a RadImageNet checkpoint, not MedicalNet.")
        pretraining = load_radimagenet_checkpoint(backbone, radimagenet_checkpoint) if radimagenet_checkpoint else None
    model = ROIAttentionClassifier(backbone, num_classes=num_classes, attention_hidden_dim=attention_hidden_dim)
    model.pretraining = pretraining
    return model


def _checkpoint_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "net", "model", "backbone"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        if all(isinstance(key, str) for key in checkpoint):
            return checkpoint
    raise ValueError("Checkpoint must contain a PyTorch state_dict.")


def _strip_prefix(key: str) -> str:
    for prefix in ("module.", "model.", "backbone."):
        if key.startswith(prefix):
            return _strip_prefix(key[len(prefix) :])
    return key


def _adapt_first_conv_weight(weight, expected_shape):
    if tuple(weight.shape) == tuple(expected_shape):
        return weight
    if len(weight.shape) == 5 and weight.shape[1] == 1 and expected_shape[1] == 3:
        repeated = weight.repeat(1, 3, 1, 1, 1) / 3.0
        if tuple(repeated.shape) == tuple(expected_shape):
            return repeated
    if len(weight.shape) == 5 and weight.shape[1] == 3 and expected_shape[1] == 1:
        averaged = weight.mean(dim=1, keepdim=True)
        if tuple(averaged.shape) == tuple(expected_shape):
            return averaged
    return weight


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source_state(checkpoint_path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(checkpoint_path).expanduser().resolve(strict=True)
    source = _checkpoint_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    normalized = {_strip_prefix(key): value for key, value in source.items()}
    if len(normalized) != len(source):
        raise ValueError(f"Duplicate keys after checkpoint prefix removal: {path}")
    return path, normalized


def _load_complete_backbone(model: nn.Module, adapted: dict[str, Any], path: Path) -> None:
    required = {key for key in model.state_dict() if not key.endswith("num_batches_tracked")}
    missing = sorted(required - adapted.keys())
    if missing:
        raise ValueError(f"Incomplete backbone checkpoint {path}: {len(missing)} missing keys, e.g. {missing[:5]}")
    model.load_state_dict(adapted, strict=False)


def load_medicalnet_checkpoint(model: nn.Module, checkpoint_path: str | Path, model_name: str) -> dict[str, Any]:
    """Load the official MedicalNet 23-dataset ResNet18/50 backbone."""

    expected_name = {"resnet18": "resnet_18_23dataset.pth", "resnet50": "resnet_50_23dataset.pth"}.get(model_name.lower())
    if expected_name is None:
        raise ValueError("MedicalNet 23-dataset weights are supported for ResNet18 and ResNet50.")
    if Path(checkpoint_path).name.lower() != expected_name:
        raise ValueError(f"{model_name} requires the official 23-dataset checkpoint named {expected_name}.")
    path, source_state = _load_source_state(checkpoint_path)
    model_state = model.state_dict()
    adapted: dict[str, Any] = {}
    for key, value in source_state.items():
        if key not in model_state:
            continue
        if key == "conv1.weight":
            value = _adapt_first_conv_weight(value, model_state[key].shape)
        if tuple(value.shape) != tuple(model_state[key].shape):
            raise ValueError(f"Incompatible MedicalNet tensor {key}: {tuple(value.shape)} != {tuple(model_state[key].shape)}")
        adapted[key] = value
    _load_complete_backbone(model, adapted, path)
    return {
        "source": "MedicalNet 23-dataset",
        "checkpoint_path": str(path),
        "checkpoint_sha256": _checkpoint_sha256(path),
        "loaded_tensors": len(adapted),
        "input_channel_adaptation": "repeat 1-channel kernel across 3 MRI channels and divide by 3",
    }


def _densenet_2d_key(key: str) -> str:
    if key == "features.0.weight":
        return "features.conv0.weight"
    if key.startswith("features.1."):
        return "features.norm0." + key[len("features.1."):]
    for idx in (1, 2, 3):
        prefix = f"features.transition{idx}."
        if key.startswith(prefix):
            suffix = key[len(prefix):]
            part, _, rest = suffix.partition(".")
            return prefix + {"0": "norm", "2": "conv"}.get(part, part) + "." + rest
    return key


def _inflate_2d_kernel(weight, expected_shape, key: str):
    if weight.ndim != 4 or len(expected_shape) != 5:
        raise ValueError(f"Expected a 2D convolution for {key}, got {tuple(weight.shape)}")
    if key == "features.0.weight":
        weight = _adapt_first_conv_weight(weight.unsqueeze(2), (expected_shape[0], expected_shape[1], 1, *expected_shape[3:]))
        weight = weight.squeeze(2)
    if tuple(weight.shape) != (expected_shape[0], expected_shape[1], *expected_shape[3:]):
        raise ValueError(f"Incompatible RadImageNet kernel {key}: {tuple(weight.shape)} != {tuple(expected_shape)}")
    depth = expected_shape[2]
    return weight.unsqueeze(2).repeat(1, 1, depth, 1, 1) / depth


def load_radimagenet_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    """Inflate every RadImageNet DenseNet121 2D convolution into a 3D kernel."""

    path, source_state = _load_source_state(checkpoint_path)
    model_state = model.state_dict()
    adapted: dict[str, Any] = {}
    for key, expected in model_state.items():
        source_key = _densenet_2d_key(key)
        if source_key not in source_state:
            continue
        value = source_state[source_key]
        if expected.ndim == 5:
            value = _inflate_2d_kernel(value, expected.shape, key)
        elif tuple(value.shape) != tuple(expected.shape):
            raise ValueError(f"Incompatible RadImageNet tensor {source_key}: {tuple(value.shape)} != {tuple(expected.shape)}")
        adapted[key] = value
    _load_complete_backbone(model, adapted, path)
    return {
        "source": "RadImageNet DenseNet121 2D",
        "checkpoint_path": str(path),
        "checkpoint_sha256": _checkpoint_sha256(path),
        "loaded_tensors": len(adapted),
        "kernel_inflation": "repeat each 2D convolution kernel along depth and divide by kernel depth",
        "input_channel_adaptation": "repeat and divide by 3 if the source stem has one channel",
    }
