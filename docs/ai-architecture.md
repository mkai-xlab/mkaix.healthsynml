# Knee OA Detection: AI Architecture Diagrams

This folder contains architecture diagrams for the Knee OA detection system.

## Inference Pipeline

![Inference Pipeline](ai_model/inference-pipeline.png)

End-to-end inference pipeline from X-ray input to KL grade prediction. YOLOv8 detects ROI, crops and resizes to 384×384, applies preprocessing (CLAHE, augmentation), runs parallel CNN models (DenseNet-121 & SE-ResNeXt-50), ensembles predictions, and generates GradCAM visualization.

---

## DenseNet-121 Architecture

### Page 1: Overall Architecture

![DenseNet-121 Overall](ai_model/densenet121-01.png)

Complete DenseNet-121 architecture showing: input 384×384×3 → conv 7×7 → maxpool → Dense Block 1 (6 layers, k=32) → Transition 1 (θ=0.5) → Dense Block 2 (12 layers) → Transition 2 → Dense Block 3 (24 layers) → Transition 3 → Dense Block 4 (16 layers) → BN + ReLU → avgpool → FC 5 logits. k=32 is growth rate, θ=0.5 is compression factor.

### Page 2: Dense Block & Transition Layer

![DenseNet-121 Dense Block](ai_model/densenet121-02.png)

Block-level view of Dense Blocks and Transition Layers. Each Dense Block contains multiple layers with channel-wise concatenation [cat]. Transition Layers reduce channels by θ=0.5 and spatial resolution by 2.

### Page 3: Dense Layer (BN+ReLU+Conv)

![DenseNet-121 Dense Layer](ai_model/densenet121-03.png)

Internal structure of a single Dense Layer: Input → BN → ReLU → conv 1×1 (C→4k) → BN → ReLU → conv 3×3 (4k→k) → concat with input. Output channels grow by k=32 each layer.

### Page 4: Transition Layer (Compression)

![DenseNet-121 Transition](ai_model/densenet121-04.png)

Transition layer details: Dense Block output → BN → ReLU → conv 1×1 (reduces channels by θ=0.5) → avgpool 2×2 (halves spatial resolution).

---

## SE-ResNeXt-50 Architecture

### Page 1: Overall Architecture

![SE-ResNeXt-50 Overall](ai_model/resnext50-01.png)

Complete SE-ResNeXt-50 32x4d architecture: input 384×384×3 → conv 7×7 → maxpool → Layer 1 (3 blocks) → Layer 2 (4 blocks, stride 2) → Layer 3 (6 blocks) → Layer 4 (3 blocks) → conv 1×1 → avgpool → 5 logits. g=32 is cardinality (grouped conv), SE provides channel attention.

### Page 2: Stage Detail

![SE-ResNeXt-50 Stage](ai_model/resnext50-02.png)

Stage-level detail showing blocks within a layer. Each block has: 1×1 conv (4C) → 3×3 grouped conv (g=32) → 1×1 conv (4C→C) → SE attention → residual add. Dashed lines show skip connections.

### Page 3: SE-Bottleneck Block

![SE-ResNeXt-50 Bottleneck](ai_model/resnext50-03.png)

Detailed SE-Bottleneck structure: Input → conv 1×1 (C→4C) → conv 3×3 grouped (4C) → conv 1×1 (4C→C_out) → SE module (GAP→FC→σ) → add with shortcut → ReLU → Output.

### Page 4: Squeeze-Excitation Module

![SE-ResNeXt-50 SE Module](ai_model/resnext50-04.png)

Squeeze-Excitation attention module: Input feature map H×W×C → GAP (squeeze to 1×1×C) → FC (C→C/16) → ReLU → FC (C/16→C) → Sigmoid → Scale (channel-wise multiplication). Reduction ratio r=16.
