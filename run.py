"""Print a supported classifier architecture without starting the inference API.

Usage:
    python run.py --model densenet121
    python run.py --model se_resnext
    python run.py --model se_resnext --no-pretrained

The default uses ImageNet-pretrained weights. This script is intentionally
independent from ``main.py`` and does not load project checkpoints.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import timm


class AppSEResNeXt(nn.Module):
    """Match the SE-ResNeXt feature backbone and five-map head used by the app."""

    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "seresnext50_32x4d",
            pretrained=pretrained,
            features_only=True,
            out_indices=(4,),
        )
        channels = self.backbone.feature_info.channels()[0]
        self.class_conv = nn.Conv2d(channels, 5, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)[0]
        return self.class_conv(features).mean(dim=(2, 3))


def build_model(model_name: str, pretrained: bool) -> tuple[nn.Module, nn.Module, str]:
    """Build one of the classifier architectures used by the inference app."""
    if model_name == "densenet121":
        model = timm.create_model("densenet121", pretrained=pretrained, num_classes=5)
        return model, model.features.norm5, "features.norm5"

    model = AppSEResNeXt(pretrained=pretrained)
    return model, model.backbone.layer4, "backbone.layer4"


def print_architecture(model_name: str, pretrained: bool, input_size: int) -> None:
    """Create the selected model and print its complete structure and key shapes."""
    model, target_layer, target_name = build_model(model_name, pretrained)
    model.eval()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    display_name = (
        "DenseNet-121" if model_name == "densenet121" else "SE-ResNeXt-50 32x4d"
    )
    print(f"Model: {display_name}")
    print(f"Pretrained ImageNet weights: {pretrained}")
    print(f"Total parameters: {parameter_count:,}")
    print(f"Trainable parameters: {trainable_count:,}")
    print("\n===== COMPLETE MODEL =====")
    print(model)

    print("\n===== FEATURE LAYERS =====")
    feature_root = model.features if model_name == "densenet121" else model.backbone
    for name, layer in feature_root.named_children():
        print(f"{('features' if model_name == 'densenet121' else 'backbone')}.{name}: {layer.__class__.__name__}")

    if model_name == "densenet121":
        print("\n===== DENSE BLOCK DETAILS =====")
        for block_name in ("denseblock1", "denseblock2", "denseblock3", "denseblock4"):
            block = getattr(model.features, block_name)
            dense_layers = list(block.named_children())
            print(f"{block_name}: {len(dense_layers)} dense layers")
            for layer_name, layer in dense_layers:
                print(f"  features.{block_name}.{layer_name}: {layer.__class__.__name__}")
    else:
        print("\n===== SE-RESNEXT STAGES =====")
        for stage_name in ("layer1", "layer2", "layer3", "layer4"):
            stage = getattr(model.backbone, stage_name, None)
            if stage is not None:
                print(f"backbone.{stage_name}: {stage.__class__.__name__}")

    print("\n===== CLASSIFIER =====")
    if model_name == "densenet121":
        print(f"classifier: {model.classifier}")
    else:
        print(f"class_conv: {model.class_conv}")

    # A forward hook verifies the exact tensor used by the app's Grad-CAM target.
    captured: dict[str, tuple[int, ...]] = {}

    def capture_norm5(_module, _inputs, output):
        captured["norm5"] = tuple(output.shape)

    handle = target_layer.register_forward_hook(capture_norm5)
    try:
        with torch.no_grad():
            logits = model(torch.zeros(1, 3, input_size, input_size))
    finally:
        handle.remove()

    print("\n===== SHAPE CHECK =====")
    print(f"Grad-CAM target: {target_name}")
    print(f"Input tensor: (1, 3, {input_size}, {input_size})")
    print(f"Target output: {captured['norm5']}")
    print(f"Classifier logits: {tuple(logits.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("densenet121", "se_resnext"),
        default="densenet121",
        help="Classifier architecture to print.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=384,
        help="Square input size for the shape check (default: 384).",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Create the architecture with random weights and avoid downloading weights.",
    )
    args = parser.parse_args()
    print_architecture(
        model_name=args.model,
        pretrained=not args.no_pretrained,
        input_size=args.input_size,
    )


if __name__ == "__main__":
    main()
