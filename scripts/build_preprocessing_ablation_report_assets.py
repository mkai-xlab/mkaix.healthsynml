"""Render selected DenseNet native-CAM examples without importing timm."""

from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "2026-07-25_23-48-22_997435_UTC_preprocessing_quality_ablation"
DATASET = Path("/home/viet/Downloads/kaggle_knee_osteoarthritis")
OUTPUT = ROOT / "docs/report/dense_net_121/assets/2026-07-25_23-48-22_preprocessing_quality_ablation"


class DenseLayer(nn.Module):
    def __init__(self, input_features: int, growth_rate: int = 32, bn_size: int = 4):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(input_features)
        self.conv1 = nn.Conv2d(input_features, bn_size * growth_rate, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, 3, padding=1, bias=False)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        merged = torch.cat(features, dim=1)
        bottleneck = self.conv1(F.relu(self.norm1(merged), inplace=True))
        return self.conv2(F.relu(self.norm2(bottleneck), inplace=True))


class DenseBlock(nn.ModuleDict):
    def __init__(self, layers: int, input_features: int):
        super().__init__()
        for index in range(layers):
            self.add_module(
                f"denselayer{index + 1}",
                DenseLayer(input_features + index * 32),
            )

    def forward(self, initial: torch.Tensor) -> torch.Tensor:
        features = [initial]
        for layer in self.values():
            features.append(layer(features))
        return torch.cat(features, dim=1)


class DenseBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features_conv0 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.features_norm0 = nn.BatchNorm2d(64)
        self.features_denseblock1 = DenseBlock(6, 64)
        self.features_transition1 = self._transition(256, 128)
        self.features_denseblock2 = DenseBlock(12, 128)
        self.features_transition2 = self._transition(512, 256)
        self.features_denseblock3 = DenseBlock(24, 256)
        self.features_transition3 = self._transition(1024, 512)
        self.features_denseblock4 = DenseBlock(16, 512)
        self.features_norm5 = nn.BatchNorm2d(1024)

    @staticmethod
    def _transition(input_features: int, output_features: int) -> nn.Sequential:
        return nn.Sequential(OrderedDict([
            ("norm", nn.BatchNorm2d(input_features)),
            ("conv", nn.Conv2d(input_features, output_features, 1, bias=False)),
            ("pool", nn.AvgPool2d(2, stride=2)),
        ]))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features_conv0(images)
        x = F.relu(self.features_norm0(x), inplace=True)
        x = F.max_pool2d(x, 3, stride=2, padding=1)
        x = self.features_denseblock1(x)
        x = self.features_transition1.norm(x)
        x = self.features_transition1.conv(F.relu(x, inplace=True))
        x = self.features_transition1.pool(x)
        x = self.features_denseblock2(x)
        x = self.features_transition2.norm(x)
        x = self.features_transition2.conv(F.relu(x, inplace=True))
        x = self.features_transition2.pool(x)
        x = self.features_denseblock3(x)
        x = self.features_transition3.norm(x)
        x = self.features_transition3.conv(F.relu(x, inplace=True))
        x = self.features_transition3.pool(x)
        x = self.features_denseblock4(x)
        return F.relu(self.features_norm5(x), inplace=True)


class DenseNetNativeCAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = DenseBackbone()
        self.class_conv = nn.Conv2d(1024, 5, 1)

    def class_maps(self, images: torch.Tensor) -> torch.Tensor:
        return self.class_conv(self.backbone(images))


def preprocess(path: Path) -> tuple[np.ndarray, torch.Tensor]:
    image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=1.25, tileGridSize=(8, 8)).apply(lightness)
    enhanced = cv2.cvtColor(
        cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2RGB
    )
    height, width = enhanced.shape[:2]
    side = max(height, width)
    top = (side - height) // 2
    bottom = side - height - top
    left = (side - width) // 2
    right = side - width - left
    padded = cv2.copyMakeBorder(
        enhanced, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    resized = np.asarray(
        Image.fromarray(padded).resize((384, 384), Image.Resampling.BILINEAR)
    ).copy()
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return resized, ((tensor - mean) / std).unsqueeze(0)


def normalize_cam(class_maps: torch.Tensor, grade: int) -> np.ndarray:
    cam = F.relu(class_maps[:, grade : grade + 1])
    cam = F.interpolate(cam, (384, 384), mode="bilinear", align_corners=False)[0, 0]
    cam = cam / cam.max().clamp_min(1e-8)
    return cam.cpu().numpy()


def overlay(image: np.ndarray, cam: np.ndarray) -> np.ndarray:
    heat = cv2.applyColorMap(np.uint8(np.clip(cam, 0, 1) * 255), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.uint8(np.clip(0.58 * image + 0.42 * heat, 0, 255))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(
        RUN / "clahe1_25_then_pad/best_model.pth",
        map_location="cpu",
        weights_only=False,
    )
    model = DenseNetNativeCAM().eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    rows = {
        Path(row["path"]).name: row
        for row in csv.DictReader(
            (RUN / "clahe1_25_then_pad/native_cam_audit.csv").open()
        )
    }
    cases = [
        ("9421302R.png", "Correct + concentrated", "Strong joint concentration and correct Grade 1"),
        ("9409250R.png", "Wrong + concentrated", "Plausible joint CAM, but Grade 3 predicted as Grade 0"),
        ("9512864L.png", "Correct + border-heavy", "Correct Grade 4; substantial marginal/border energy"),
        ("9360034L.png", "Wrong + lower/border-heavy", "Grade 3 predicted as Grade 2; CAM spills below joint"),
    ]
    figure, axes = plt.subplots(len(cases), 3, figsize=(11, 14), constrained_layout=True)
    for row_index, (filename, category, interpretation) in enumerate(cases):
        audit = rows[filename]
        true_grade = int(audit["true_grade"])
        expected_prediction = int(audit["predicted_grade"])
        image_path = DATASET / "val" / str(true_grade) / filename
        image, tensor = preprocess(image_path)
        with torch.no_grad():
            maps = model.class_maps(tensor)
            prediction = int(maps.mean((2, 3)).argmax(1).item())
        if prediction != expected_prediction:
            raise RuntimeError(
                f"Prediction mismatch for {filename}: rendered={prediction}, audit={expected_prediction}"
            )
        predicted_cam = normalize_cam(maps, prediction)
        true_cam = normalize_cam(maps, true_grade)
        axes[row_index, 0].imshow(image)
        axes[row_index, 1].imshow(overlay(image, predicted_cam))
        axes[row_index, 2].imshow(overlay(image, true_cam))
        axes[row_index, 0].set_title(
            f"{category}\nTrue G{true_grade}, predicted G{prediction}\n{interpretation}", fontsize=10
        )
        axes[row_index, 1].set_title(
            "Predicted-class native CAM\n"
            f"joint={float(audit['predicted_joint_energy']):.3f}, "
            f"border={float(audit['predicted_border_energy']):.3f}", fontsize=10
        )
        axes[row_index, 2].set_title(
            "True-class native CAM\n"
            f"occlusion drop={float(audit['joint_occlusion_probability_drop']):.3f}", fontsize=10
        )
        for axis in axes[row_index]:
            axis.axis("off")
    figure.suptitle(
        "DenseNet-121 CLAHE 1.25: good and bad native-CAM behavior",
        fontsize=15,
    )
    figure.savefig(OUTPUT / "good_vs_bad_native_cam.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT / "good_vs_bad_native_cam.png")


if __name__ == "__main__":
    main()
