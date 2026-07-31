# AI Production Report

## Scope

This document records the current local production configuration for research and capstone demonstration. It is an assistive KL-grade classifier, not a clinical diagnostic device.

## Active Configuration

| Item | Value |
| --- | --- |
| Classifier | timm DenseNet-121, five CE logits |
| Classifier checkpoint | `checkpoints/densenet121/2026-07-30_09-03-29_850983_UTC_paired_view_yolo_roi/best_model.pth` |
| Checkpoint metadata | `timm_densenet121_linear_gradcam`, epoch 4, CE |
| Detector | YOLOv8 joint detector |
| Detector checkpoint | `checkpoints/yolov8/2026-07-26_20-49-25_joint_detection/best.pt` |
| Input | `384 x 384` RGB tensor, ImageNet normalization |
| Laterality | Natural orientation; no right-knee mirroring |
| Explanation | Grad-CAM from DenseNet `features.norm5` |

## Inference Pipeline

1. YOLO detects each tibiofemoral joint.
2. The box is expanded to `1.15 x max(width, height)`, centred, and made square.
3. Black pixels are added only where the expanded square crosses the original radiograph boundary.
4. The ROI receives LAB CLAHE (`clipLimit=1.25`, `8 x 8` tiles), square padding as a safety check, resize to `384 x 384`, tensor conversion, and ImageNet normalization.
5. DenseNet produces five KL logits. The API returns softmax probabilities and Grad-CAM for the predicted grade.

The JSON field remains `gradcam_image` for backward compatibility. The result viewer calls it “Class activation heatmap” because other selectable architectures use native CAM; the active DenseNet path uses Grad-CAM.

## Training Lineage

The active checkpoint came from paired-view adaptation. The base DenseNet was fine-tuned for five CE epochs using a 50/50 mix of published labelled 224 crops and matched YOLO square ROIs. The selected checkpoint reached fixed-YOLO-ROI validation QWK `0.7405`, macro F1 `0.6220`, and macro AP `0.6665` before the subsequent production-crop correction.

The later production-ROI robustness run improved validation QWK to `0.7650` and macro F1 to `0.6451`, but increased average Grad-CAM border energy from `0.0761` to `0.0844` and reduced joint energy from `0.8648` to `0.8505`. It was not promoted.

## Limitations and Use

KL labels supervise a whole-knee grade, not the exact location of joint-space narrowing or osteophytes. Grad-CAM is therefore evidence visualization, not segmentation or proof of causality. The final DenseNet feature grid is coarse, so maps can be broad. Low class confidence or a border-dominant map should be treated as uncertain. Do not use unlabeled API images to claim model accuracy.

## Operations

Use `make up` to start the API and viewer. Open `http://localhost:8005/docs` for the API and `http://localhost:8088` to paste and inspect a prediction response. Run `make experiments` after adding completed report artifacts to refresh `docs/report/all_experiments.xlsx` and its CSV tabs.
