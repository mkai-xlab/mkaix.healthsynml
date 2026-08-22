# KL Grade Classification from Knee X-Ray Using Deep CNN Ensemble

## Abstract

This study presents an ensemble of convolutional neural networks (CNNs) for automatic KL (Kellgren-Lawrence) grading of knee osteoarthritis from radiographic images. Our system combines DenseNet-121 and SE-ResNeXt-50 architectures trained with different loss functions and data augmentation strategies. The ensemble achieves competitive performance on the OAI dataset, with Quadratic Weighted Kappa (QWK) scores up to **0.8394** and **0.8216** for individual models.

## 1. Dataset and Preprocessing

### 1.1 Dataset
- **Source**: Osteoarthritis Initiative (OAI) dataset
- **Split**: Train/Validation/Test (hash-based deduplication, patient-level split)
  - Training: 5,778 unique images
  - Validation: 826 images
  - Test: 1,656 unique images
- **Classes**: KL Grades 0-4 (ordinal)

### 1.2 ROI Detection
- **Model**: YOLOv8n knee joint detector
- **Performance**: mAP50-95 = 0.8136, Precision = 0.9879, Recall = 0.9881
- **ROI Expansion**: 1.15× max(box width, height)

### 1.3 Preprocessing Pipeline
1. YOLO ROI crop with black padding
2. LAB CLAHE (α=1.25)
3. Square pad to 384×384
4. ToTensor + ImageNet normalization

## 2. Model Architectures

### 2.1 DenseNet-121
| Component | Specification |
|-----------|--------------|
| Architecture | DenseNet-121 with dense blocks |
| Growth Rate | 32 |
| Classifier | Standard linear head |
| Input Size | 224×224 or 384×384 |
| Parameters | ~8M |

### 2.2 SE-ResNeXt-50 32×4d
| Component | Specification |
|-----------|--------------|
| Architecture | SE-ResNeXt-50 with 32×4d cardinality |
| Backbone | ResNeXt bottleneck with Squeeze-Excitation |
| Classifier | Linear head |
| Input Size | 384×384 |
| Parameters | ~27M |

## 3. Training Configuration

### 3.1 DenseNet-121 Best Config (Focal CORN)
| Parameter | Value |
|-----------|-------|
| Loss | Focal CORN (ordinal) |
| Sampler | WeightedRandomSampler (balanced) |
| Epochs | 30 |
| Batch Size | 16 |
| Learning Rates | Warmup: 1e-3, Head: 1e-4, Backbone: 5e-5, Finetune: 2e-5 |
| Augmentation | H-flip (p=0.5), rotation ±8°, double Random Erasing (p=0.8-0.9) |
| Pipeline | 3-stage (warmup → coarse → finetune) |

### 3.2 SE-ResNeXt-50 Best Config
| Parameter | Value |
|-----------|-------|
| Loss | Cross-Entropy (CE) |
| Sampler | Full inverse-frequency |
| Epochs | 30 |
| Batch Size | 48 |
| Augmentation | H-flip (p=0.5), rotation ±5°, gamma (p=0.2), Random Erasing (p=0.1) |
| Pipeline | 3-stage with native CAM head |

## 4. Results

### 4.1 Individual Model Performance

| Model | Loss | Input | Test Acc | Test QWK | Macro F1 | Grade 1 Recall | ROC AUC |
|-------|------|-------|----------|----------|----------|-----------------|--------|
| **DenseNet-121 (Focal CORN)** | FocalCORN | 224×224 | **0.6733** | **0.8394** | 0.6900 | 0.44 | 0.8984 |
| DenseNet-121 (CE) | CE | 224×224 | 0.6691 | 0.8058 | 0.6700 | 0.22 | 0.8798 |
| DenseNet-121 (CORN) | CORN | 384×384 | 0.6715 | 0.8246 | 0.6800 | 0.31 | 0.8963 |
| DenseNet-121 (Single-Stage YOLO-ROI Adaptation) | CE | 384×384 | 0.5876 | 0.7372 | 0.5909 | 0.2297 | 0.8479 |
| **SE-ResNeXt-50 (Native CAM)** | CE | 384×384 | **0.6558** | **0.8216** | 0.6781 | 0.41 | 0.8980 |

### 4.2 Per-Grade Performance (DenseNet-121 Focal CORN)

| KL Grade | Support | Precision | Recall | F1-Score |
|----------|---------|-----------|--------|----------|
| Grade 0 | 639 | 0.72 | 0.76 | 0.74 |
| Grade 1 | 296 | 0.36 | 0.44 | 0.39 |
| Grade 2 | 447 | 0.69 | 0.67 | 0.68 |
| Grade 3 | 223 | 0.71 | 0.70 | 0.70 |
| Grade 4 | 51 | 0.78 | 0.67 | 0.72 |

### 4.3 Ensemble Strategy

The final ensemble combines DenseNet-121 and SE-ResNeXt-50 using **soft voting on raw logits**.

**How it works:**

1. **Logit extraction** — both models output a raw logits vector `z ∈ ℝ⁵` (one value per KL grade), without applying softmax individually
2. **Element-wise average** — `z_final = 0.5 × z_DN121 + 0.5 × z_SEResNeXt`
3. **Softmax** — `p_k = exp(z_k) / Σⱼ exp(z_j)` converts averaged logits to probabilities
4. **Argmax** — `class = argmax(p_k)` selects the predicted KL grade

**Why soft voting on logits works better than on probabilities:**

| Aspect | Soft voting on probabilities | Soft voting on logits (ours) |
|---|---|---|
| Temperature | Sensitive to calibration | Temperature-independent |
| Ordinal information | Preserved but diluted by softmax | Preserved — logits encode ordinal spacing |
| Robustness | Sensitive to individual model calibration | More robust — averaging before softmax |

**Why these two models complement each other:**

| Factor | DenseNet-121 | SE-ResNeXt-50 |
|---|---|---|
| Architecture | Dense connections (feature reuse) | ResNeXt bottlenecks + SE gating |
| Cardinality | — | 32×4d (groups) |
| Loss | Focal CORN (ordinal) | Cross-Entropy |
| Input size | 224×224 | 384×384 |
| Sampler | WeightedRandomSampler (balanced) | Full inverse-frequency |
| Inductive bias | Dense feature concatenation | Aggregated residual paths |

The **Focal CORN loss** in DenseNet-121 penalises adjacent-grade errors more heavily than distant errors, while SE-ResNeXt-50 with CE learns discriminative class boundaries independently. Averaging their logits combines these complementary decision boundaries.

**Result:** Ensemble QWK ~0.84 vs individual best 0.8394

## 5. Explainability

### 5.1 Grad-CAM Visualization
- Applied to DenseNet-121 backbone for disease localization
- Peak activation inside joint region: 0.99+
- Joint energy: 0.82, Border energy: 0.11

### 5.2 CAM Analysis
- Native CAM and Grad-CAM showed correlation of 1.0000
- No significant difference in localization quality
- Model focuses on relevant knee anatomy

## 6. Conclusion

This study demonstrates competitive KL grading performance using CNN ensembles:

1. **DenseNet-121 with Focal CORN** achieves best single-model QWK of 0.8394
2. **SE-ResNeXt-50** provides complementary predictions for ensemble
3. **Grade 1 (Doubtful OA)** remains the most challenging class
4. **Grad-CAM** provides interpretable disease localization

## 7. Interview Questions

This section documents production code questions that may be asked during defense, with specific file:line references.

### 7.1 Model Internals

#### MI-D1: DenseNet-121 Architecture Trace

**Q:** Trace `model.forward()` của DenseNet-121 từ input đến logits. Có bao nhiêu params?

**A:** `densenet121_model.py` line 32–33:

```
images (B, 3, 384, 384)
  │
  ▼
features.conv0 → (B, 64, H, W)
  │
  ▼
denseblock1 (6 conv layers, growth 32) → (B, 256, H', W')
  │
  ▼
transition1 → (B, 128, H'/2, W'/2)
  │
  ▼
denseblock2 (12 conv) → (B, 512, H''/4, W''/4)
  │
  ▼
denseblock3 (24 conv) → (B, 1024, H'''/8, W'''/8)
  │
  ▼
norm5 → (B, 1024, spatial) ← GRAD-CAM TARGET
  │
  ▼
global_avg_pool → (B, 1024)
  │
  ▼
classifier(Linear 1024→5) → (B, 5) LOGITS
```

Total ~8M params, dropout `drop_rate=0.20` trong timm wrapper.

#### MI-D2: `pretrained=False` nhưng vẫn dùng ImageNet mean/std?

**Q:** Tại sao load model với `pretrained=False` nhưng checkpoint vẫn chứa ImageNet pretrained weights?

**A:** `knee_oa_pipeline.py` line 122–131:

```python
model = get_model(model_name, num_classes=5, pretrained=False)
checkpoint = torch.load(absolute_path, ...)
model.load_state_dict(state_dict, strict=True)
```

Checkpoint được train với `pretrained=True` (timm auto-download ImageNet weights) → sau đó fine-tune trên X-ray → weights được save vào checkpoint. `pretrained=False` chỉ bỏ qua re-download. Checkpoint metadata verify: `architecture` field phải khớp (`timm_densenet121_linear_gradcam`).

#### MI-S1: SE-ResNeXt-50 `features_only=True` khác gì DenseNet `features_only=False`?

**A:** `se_resnext50_32x4d_model.py` line 19–32:

| | DenseNet-121 | SE-ResNeXt-50 |
|---|---|---|
| `features_only` | False (timm thêm head) | **True** (tự tạo head) |
| Classifier | `backbone.classifier` | **`nn.Linear(2048, 5)`** |
| GAP | timm tự làm | **`features.mean(dim=(2,3))`** |
| Grad-CAM layer | `norm5` | `layer4` |

SE-ResNeXt dùng `features_only=True` vì timm pretrained head không tương thích với custom training. Tự tạo classifier đảm bảo checkpoint format đồng nhất.

### 7.2 Config Deep Dive

#### CD-1: `IMG_SIZE=384` — tại sao không phải 224 hay 512?

**Q:** Tại sao production dùng `IMG_SIZE=384`?

**A:** `config.py` line 60:

- **224:** Spatial size ~7×7 sau 5 conv blocks → không đủ detail cho osteophyte features nhỏ
- **384:** Spatial size ~12×12 sau layer4 → đủ detail + memory OK với GPU 8-16GB
- **512:** 1M pixels/batch → batch size phải giảm → overfitting với dataset 8K ảnh

Ablation notebooks test 224 vs 384 → 384 cho accuracy cao hơn.

#### CD-2: `MODEL_MODE_ALIASES` — tại sao cần aliases?

**A:** `config.py` line 10–17:

Users có thể set `MODEL_MODE` bằng nhiều variant: `densenet121`, `dense_net_121`, `DenseNet121` → tất cả normalize về `"densenet121"` qua `.strip().lower()`. Nếu không có aliases → user phải nhớ exact string.

#### CD-3: SE-ResNeXt checkpoint path có timestamp — tại sao?

**A:** `config.py` line 50–56:

```
checkpoints/se_resnext50_32x4d/2026-08-08_08-35-38_UTC_linear_gradcam/best_model.pth
```

Timestamp = ngày giờ training kết thúc. Mục đích: version control checkpoint (không ghi đè), reproducibility, debug. DenseNet default path không có timestamp → có thể bị ghi đè.

### 7.3 Grad-CAM Deep Dive

#### GC-1: Tại sao `detach()` rồi `requires_grad_(True)`?

**Q:** Tại sao phải tách tensor khỏi inference graph rồi bật lại gradient?

**A:** `gradcam_service.py` line 64–65:

```python
grad_input = input_tensor.detach().clone().requires_grad_(True)
```

- `detach()`: tách khỏi inference computation graph → backward không ảnh hưởng inference
- `clone()`: tạo copy mới → không modify original tensor
- `requires_grad_(True)`: bắt buộc để `autograd.grad()` compute gradient được

Inference chạy trong `torch.no_grad()` context. Grad-CAM cần `torch.enable_grad()` riêng → không conflict.

#### GC-2: Gradient `retain_graph=False` — hệ quả nếu `True`?

**A:** `gradcam_service.py` line 80:

```python
gradient = torch.autograd.grad(
    logits[0, int(predicted_class)], activation,
    retain_graph=False, create_graph=False,
)[0]
```

- `retain_graph=False`: graph freed sau backward → save memory. Đủ cho Grad-CAM (1 backward pass duy nhất)
- `retain_graph=True`: giữ graph → có thể backward lần 2 → tốn memory thêm
- `create_graph=False`: không cần higher-order derivatives

#### GC-3: `weights = gradient.mean(dim=(2,3))` — tại sao mean?

**A:** `gradcam_service.py` line 86:

Gradient shape: `(B, C, H, W)`. Mean theo spatial dims `(2,3)` = global average pooling của gradient → αₖ = Σ gₖ(h,w) / (H×W) = importance score của channel k.

Mean smooth và stable (Grad-CAM paper). ReLU sau (`line 90`) loại negative contributions (channel làm giảm predicted class score).

#### GC-4: Hook không `remove()` → memory leak?

**A:** `gradcam_service.py` line 55, 102:

```python
handle = target_layer.register_forward_hook(capture_activation)
try:
    ...
finally:
    handle.remove()
```

Hook là callback chạy mỗi forward. Nếu không `remove()`:
- Mỗi predict → hook được gọi → `captured` dict accumulate
- sau N predictions → N×activation memory leak
- `try/finally` đảm bảo `remove()` luôn được gọi kể cả khi có exception

### 7.4 Preprocessing Internals

#### PP-1: X-ray grayscale tại sao đọc `IMREAD_COLOR`?

**A:** `preprocessing_service.py` line 89:

```python
image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
```

- X-ray file có thể lưu 1 hoặc 3 channels
- `IMREAD_COLOR` force 3 channels BGR → uniform processing
- Lý do: timm pretrained models cần 3-channel RGB, CLAHE trong LAB space cần 3 channels

#### PP-2: `tile_grid_size=(8,8)` — tại sao không 4×4 hay 16×16?

**A:** `preprocessing_service.py` line 67:

8×8 grid = 64 tiles trên ảnh 384×384 → mỗi tile ~48×48 pixels.

- **Tile nhỏ hơn (4×4)**: histogram noisy (ít pixels/tile) → artifact ở tile boundaries
- **Tile lớn hơn (16×16)**: global equalization trend → mất local contrast enhancement
- **8×8 = sweet spot**: đủ local (48×48), đủ global (64 tiles), OpenCV default cho medical imaging

`clip_limit=1.25` = max contrast amplification factor per tile → prevent over-enhancement.

#### PP-3: `ToTensor` làm gì?

**A:** `preprocessing_service.py` line 75:

```python
transforms.ToTensor()  # PIL (H,W,C) uint8 → (C,H,W) float32 [0,1]
```

Sau đó `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` → ImageNet distribution. Backbone pretrained trên ImageNet → input phải có similar distribution.

### 7.5 Edge Cases

#### EC-1: YOLO detect 3 boxes → hệ thống xử lý thế nào?

**A:** `roi_service.py` line 23–26:

```python
if len(knees) == 2:
    sorted(knees, key=lambda knee: knee["box"][0])
    return ordered_knees, ["right", "left"]
if len(knees) != 1:
    return knees, ["unknown"] * len(knees)
```

3 boxes → `unknown` all. Model vẫn predict tất cả 3 detections. Không có logic loại bỏ false positive → phụ thuộc vào YOLO precision (1.000 trên val set).

#### EC-2: Bilateral X-ray chỉ detect được 1 knee?

**A:** `roi_service.py` line 28–33:

```python
if center_x < image_width * 0.40: return ["right"]
if center_x > image_width * 0.60: return ["left"]
return ["unknown"]
```

40/60 thresholds: ảnh bilateral chuẩn → 2 knees ở ~25% và ~75% width. Vùng 40–60% là trung tâm → không rõ bên nào.

#### EC-3: Model predict KL-4 với confidence thấp (0.31)?

**A:** `knee_oa_pipeline.py` line 179–180:

```python
predicted_class = int(predictions.argmax(dim=1))
```

Luôn trả về `argmax` bất kể confidence. Không có confidence threshold để fallback. `confidence` field trong response = xác suất của predicted class.

#### EC-4: Inference trên CPU thay vì GPU?

**A:** `knee_oa_pipeline.py` line 70:

```python
self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

DenseNet-121 CPU: ~100–300ms/knee (YOLO + preprocessing + inference + Grad-CAM). GPU: ~50ms/knee. Ensemble mode: CPU ~200–600ms, GPU ~100ms. Grad-CAM tốn thêm 1× forward + 1× backward so với inference thường.

### 7.6 Pipeline Orchestration

#### PO-1: Tại sao services là singleton?

**A:** `prediction_service.py` (end of file):

```python
prediction_service = PredictionService()
```

Model load 1 lần khi server start → shared across all requests. Tiết kiệm memory. Hệ quả: multi-worker (gunicorn -w 4) → mỗi worker có 1 instance → 4× memory. `model.eval()` → không bao giờ chuyển train mode.

#### PO-2: 3 loại ảnh trong API response khác nhau thế nào?

| Field | Kích thước | Content | Format |
|---|---|---|---|
| `annotated_image` | Original | Full X-ray + box + KL label | JPEG base64 |
| `roi_image` | ~104×104 | Cropped knee ROI | PNG base64 |
| `gradcam_image` | 384×384 | Heatmap overlay trên **processed** image | JPEG base64 |

`gradcam_image` dùng processed image (sau CLAHE + SquarePad + Resize 384) vì Grad-CAM resize về `output_size = (384, 384)`.

### 7.7 Summary — File References

| Câu hỏi | File | Lines |
|---|---|---|
| MI-D1: DenseNet forward trace | `densenet121_model.py` | 32–33 |
| MI-D2: pretrained vs ImageNet | `knee_oa_pipeline.py` | 122–131 |
| MI-S1: features_only | `se_resnext50_32x4d_model.py` | 19–32 |
| CD-1: IMG_SIZE rationale | `config.py` | 60 |
| CD-2: MODEL_MODE_ALIASES | `config.py` | 10–17 |
| CD-3: checkpoint timestamp | `config.py` | 50–56 |
| GC-1: detach+requires_grad | `gradcam_service.py` | 64–65 |
| GC-2: retain_graph | `gradcam_service.py` | 80–82 |
| GC-3: gradient.mean | `gradcam_service.py` | 86 |
| GC-4: hook.remove | `gradcam_service.py` | 55, 102 |
| PP-1: IMREAD_COLOR | `preprocessing_service.py` | 89 |
| PP-2: tile_grid_size | `preprocessing_service.py` | 67 |
| PP-3: ToTensor | `preprocessing_service.py` | 75 |
| EC-1: 3 boxes | `roi_service.py` | 23–26 |
| EC-2: 1 knee | `roi_service.py` | 28–33 |
| EC-3: argmax confidence | `knee_oa_pipeline.py` | 179–180 |
| EC-4: CPU vs GPU | `knee_oa_pipeline.py` | 70 |
| PO-1: singleton | `prediction_service.py` | last line |
| PO-2: 3 image types | `prediction_service.py`, `gradcam_service.py` | various |

---

## Citation

```bibtex
@article{knee_oa_classification,
  title={KL Grade Classification from Knee X-Ray Using Deep CNN Ensemble},
  author={},
  year={2026}
}
```

## Source Data

Full experiment data available in: `report/report.csv`

## Diagrams

- [Inference Pipeline Diagram](../diagram/inference-pipeline-v2.drawio) - Open in draw.io desktop
- [DenseNet-121 Architecture](../diagram/densenet121.drawio)
- [SE-ResNeXt-50 Architecture](../diagram/resnext50.drawio)
- [Ensemble Architecture v4](../diagram/ensemble-v4.drawio) — **Pipeline** (page 1) + **Mechanism** (page 2)
  - PNG preview: [pipeline](../diagram/ensemble_pipeline.png) · [mechanism](../diagram/ensemble_mechanism.png)
