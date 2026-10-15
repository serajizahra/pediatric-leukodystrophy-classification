from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FineTuningStrategy:
    name: str
    family: str
    description: str
    frozen_prefixes: tuple[str, ...]


RESNET_STRATEGIES = {
    "linear_probe": FineTuningStrategy(
        "linear_probe",
        "resnet",
        "Freeze the backbone and attention; train only the classifier layer.",
        ("backbone.", "attention."),
    ),
    "from_layer2": FineTuningStrategy(
        "from_layer2",
        "resnet",
        "Freeze Conv1, BN1, and Layer1; fine-tune Layer2 onward.",
        ("backbone.conv1", "backbone.bn1", "backbone.layer1"),
    ),
    "from_layer3": FineTuningStrategy(
        "from_layer3",
        "resnet",
        "Freeze Conv1, BN1, Layer1, and Layer2; fine-tune Layer3 onward.",
        ("backbone.conv1", "backbone.bn1", "backbone.layer1", "backbone.layer2"),
    ),
    "from_layer4": FineTuningStrategy(
        "from_layer4",
        "resnet",
        "Freeze Conv1, BN1, Layer1, Layer2, and Layer3; fine-tune Layer4 onward.",
        ("backbone.conv1", "backbone.bn1", "backbone.layer1", "backbone.layer2", "backbone.layer3"),
    ),
    "full": FineTuningStrategy("full", "resnet", "Fine-tune the full model.", ()),
}


DENSENET_STRATEGIES = {
    "linear_probe": FineTuningStrategy(
        "linear_probe",
        "densenet",
        "Freeze the backbone and attention; train only the classifier layer.",
        ("backbone.", "attention."),
    ),
    "from_denseblock2": FineTuningStrategy(
        "from_denseblock2",
        "densenet",
        "Freeze the stem and Dense Block 1; fine-tune Dense Block 2 onward.",
        (
            "backbone.features.0",
            "backbone.features.1",
            "backbone.features.2",
            "backbone.features.3",
            "backbone.features.denseblock1",
            "backbone.features.transition1",
        ),
    ),
    "from_denseblock3": FineTuningStrategy(
        "from_denseblock3",
        "densenet",
        "Freeze the stem, Dense Blocks 1-2, and Transition 2; fine-tune Dense Block 3 onward.",
        (
            "backbone.features.0",
            "backbone.features.1",
            "backbone.features.2",
            "backbone.features.3",
            "backbone.features.denseblock1",
            "backbone.features.transition1",
            "backbone.features.denseblock2",
            "backbone.features.transition2",
        ),
    ),
    "from_denseblock4": FineTuningStrategy(
        "from_denseblock4",
        "densenet",
        "Freeze the stem and Dense Blocks 1-3; fine-tune Dense Block 4 onward.",
        (
            "backbone.features.0",
            "backbone.features.1",
            "backbone.features.2",
            "backbone.features.3",
            "backbone.features.denseblock1",
            "backbone.features.transition1",
            "backbone.features.denseblock2",
            "backbone.features.transition2",
            "backbone.features.denseblock3",
            "backbone.features.transition3",
        ),
    ),
    "full": FineTuningStrategy("full", "densenet", "Fine-tune the full model.", ()),
}


def available_strategies(backbone_name: str) -> tuple[str, ...]:
    family = _family(backbone_name)
    strategies = RESNET_STRATEGIES if family == "resnet" else DENSENET_STRATEGIES
    return tuple(strategies.keys())


def _family(backbone_name: str) -> str:
    name = backbone_name.lower()
    if name in {"resnet18", "resnet50"}:
        return "resnet"
    if name == "densenet121":
        return "densenet"
    raise ValueError(f"Unknown backbone: {backbone_name}")


def get_strategy(backbone_name: str, strategy_name: str) -> FineTuningStrategy:
    family = _family(backbone_name)
    strategies = RESNET_STRATEGIES if family == "resnet" else DENSENET_STRATEGIES
    if strategy_name not in strategies:
        allowed = ", ".join(strategies)
        raise ValueError(f"Unknown fine-tuning strategy '{strategy_name}' for {backbone_name}. Allowed: {allowed}")
    return strategies[strategy_name]


def apply_fine_tuning_strategy(model, backbone_name: str, strategy_name: str) -> dict[str, int]:
    strategy = get_strategy(backbone_name, strategy_name)
    for _, param in model.named_parameters():
        param.requires_grad = True
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in strategy.frozen_prefixes):
            param.requires_grad = False
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen = sum(param.numel() for param in model.parameters() if not param.requires_grad)
    return {
        "strategy": strategy.name,
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(frozen),
    }
