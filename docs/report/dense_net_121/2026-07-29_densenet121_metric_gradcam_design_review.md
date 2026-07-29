# DenseNet KL-Grading and Grad-CAM Design Review

**Review date:** 2026-07-29  
**Dataset:** Kaggle Knee Osteoarthritis Dataset with Severity Grading (OAI-derived)  
**Scope:** image size, DenseNet depth, imbalance handling, augmentation, loss, and Grad-CAM localization

## Executive Decision

Do not add another complex head, ensemble, EMA, or unvalidated localization loss. The most defensible baseline is still ImageNet-pretrained **DenseNet-121**, five-class **cross-entropy**, one imbalance mechanism, complete knee ROI preservation, and post-hoc Grad-CAM. The next meaningful change is not DenseNet-201: it is training and validating on crops produced by the exact production YOLO pipeline.

The current `384x384` input is reasonable but is not proven optimal. A 224-pixel source resized to 384 does not gain new anatomical detail and the file on disk is unchanged. It does give DenseNet's final stride-32 tensor roughly `12x12` spatial cells instead of `7x7` at 224, so the interpolated CAM can be less blocky. That is a visualization/sampling advantage, not proof of better pathological localization.

## What the Exact Dataset Represents

The Kaggle release appears to repackage the same OAI-derived release described by Chen et al.: 4,130 bilateral radiographs, 8,260 labeled knees, and a 7:1:2 train/validation/test split. The original project used a customized YOLOv2 detector, expanded detected boxes by `1.3`, resized DenseNet/ResNet/VGG inputs to `224x224`, and compared cross-entropy with an ordinal loss. Their best overall classifier was VGG-19 with ordinal loss (about `69.7%` accuracy and `0.344` MAE). For DenseNet, DenseNet-121 outperformed DenseNet-201 in the reported table:

| Architecture | Manual crop CE | Manual crop ordinal | Automatic crop CE | Automatic crop ordinal |
| --- | ---: | ---: | ---: | ---: |
| DenseNet-121 | 67.3% | **68.2%** | **67.4%** | **67.8%** |
| DenseNet-169 | 66.8% | 67.1% | 66.6% | 65.4% |
| DenseNet-201 | 65.7% | 67.3% | 66.3% | 66.3% |

The authors also presented Grad-CAM examples around joint space, but only qualitative examples. They did not prove that every heatmap was anatomically correct. See the [original paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9531250/), [original code](https://github.com/PingjunChen/KneeAnalysis), and [Mendeley data release](https://data.mendeley.com/datasets/56rmx5bjcr/1).

At least four later indexed papers explicitly mention the named Kaggle dataset. The clearest verified example is Malathi et al., which combines normalization, augmentation, ROI segmentation, feature extraction, and a deep neural angular extreme learning machine. It is not a simple, directly reproducible DenseNet production recipe, and its abstract reports relative gains rather than enough split details for a fair comparison. Treat these papers as related work, not as a trusted leaderboard: [Malathi et al. (2024)](https://doi.org/10.1007/s43538-024-00366-y), [Enhanced EfficientNet-B4 (2025)](https://doi.org/10.1109/ICoICC64033.2025.11052211), [Grad-CAM comparative study (2025)](https://doi.org/10.1109/RCSM67767.2025.11506728), and [automated severity classification (2025)](https://doi.org/10.1109/ICCA66035.2025.11430959).

## Evidence From This Repository

The completed current DenseNet-121 run uses CLAHE `1.25 -> square pad -> resize 384`, natural orientation, CE, full inverse-frequency sampling, mild augmentation, and Grad-CAM at `features.norm5`. Its saved test output is approximately Accuracy `0.6697`, QWK `0.8305`, macro F1 `0.68`, AUC `0.8991`, and AP `0.7316`. Grade-1 recall remains weak (about `0.41`). The all-case Grad-CAM rerun must complete before reporting its final gate rate.

There is no valid controlled 224-vs-384 experiment in the repository. The historical 224 result used a different head/loss/augmentation, while one 384 experiment had a frozen-backbone bug. Therefore, image-size superiority is currently unproven.

The local DenseNet-201 result (QWK `0.7632`, AP `0.6798`, Grade-1 recall `0.06`) is substantially worse, although its setup was not a depth-only comparison. Published OAI results above also do not favor DenseNet-201. Keep DenseNet-121.

Preprocessing is the clearest controlled improvement. `CLAHE 1.25 -> pad -> 384` beat `pad -> CLAHE 2.0` twice on validation and improved both QWK and broad CAM geometry. This still requires confirmation on a locked holdout and production-YOLO crops.

## Recommended Configuration

| Component | Recommendation | Reason |
| --- | --- | --- |
| Backbone | DenseNet-121, ImageNet initialization | Best local trade-off; DenseNet-201 adds cost without evidence of benefit |
| Head | Standard linear five-logit head | Simple, checkpoint-compatible, supports faithful final-layer Grad-CAM |
| Loss | Plain five-class CE | Current ordinal variants collapsed or did not beat CE; revisit only with a clean controlled implementation |
| Input | Keep `384x384` for the current checkpoint | Produces a denser final CAM grid; changing size requires retraining |
| Preprocessing | CLAHE 1.25, then square pad, then resize; ImageNet normalization | Best controlled local preprocessing result; retains marginal osteophytes |
| Cropping | Exact production-YOLO ROI; no center crop | ROI-domain mismatch is the dominant unresolved issue |
| Sampler | Full inverse-frequency sampler only | Best current QWK/Grade-1 sensitivity evidence; do not also use class-weighted CE |
| Augmentation | Flip `0.5`, rotation `+/-5`, brightness/contrast `0.08`, conservative erasing | Strong geometry/photometric augmentation reduced grading metrics locally |
| EMA/TTA | Disabled | No demonstrated benefit; the EMA bundle collapsed |
| Heatmap | Grad-CAM from `features.norm5` | Faithful diagnostic for this head; do not call it a lesion segmentation map |
| Selection | `0.55 QWK + 0.30 macro-F1 + 0.15 macro-AP` | Preserves ordinal agreement and minority performance |

Do not combine a balanced sampler and class-weighted CE: that applies the imbalance correction twice and can overpredict rare grades. If calibration or Grade-0/1 reciprocal confusion is the priority, compare full and square-root sampling under the same checkpoint selection; otherwise retain the current sampler.

Current augmentation is sufficient for a baseline, but not proof against shortcut learning. ROI geometry/acquisition augmentation modestly improved broad CAM geometry while reducing QWK/F1; every arm in that experiment used the same erasing setting, so this was not a Cutout comparison. The dedicated mild-versus-aggressive Cutout ablation has not completed. If shortcut resistance is required, run that controlled validation-only comparison and keep the full ROI; never solve border activation with a center crop that removes marginal osteophytes.

## What Actually Improves Grad-CAM

Grad-CAM quality has three different meanings:

1. **Faithfulness:** the map reflects evidence used by the model. Final-layer Grad-CAM and native CAM were nearly identical locally (`correlation 1.0000`).
2. **Anatomical location:** energy lies near joint space/margins. This needs masks, landmarks, or an explicit quantitative anatomy audit.
3. **Clinical validity:** highlighted pixels correspond to osteophytes, JSN, sclerosis, or deformity. This ultimately needs expert feature annotations or occlusion/perturbation evidence.

Increasing input size affects only the first two indirectly. It cannot force the network to use joint pathology. The strongest next step is production-aligned training: generate training ROIs with the same YOLO checkpoint, confidence threshold, crop policy, padding, and preprocessing used by the API. Then quantify CAMs by grade on the same ROI distribution. A higher-resolution stride-16 explanation head is worth testing only after that; it may sharpen maps but can also learn sharper shortcuts.

## Minimal Next Experiment

Run one controlled experiment, not another broad search:

- Same patient-disjoint train/validation split, seed, CE loss, sampler, schedule, and preprocessing.
- Same production-YOLO-generated source ROIs.
- Compare input sizes `224`, `320`, and `384` over at least three seeds.
- Select using validation QWK, macro F1, and AP; report MAE and per-grade precision/recall.
- Separately report CAM joint energy, border energy, peak-inside rate, and joint occlusion drop on at least 50 cases per grade.
- Do not use the repeatedly inspected test set for selection.

If `320` is statistically equivalent to `384`, use 320 for lower latency. If 384 improves both classification confidence intervals and localization measurements, retain 384. No conclusion should be based on visual smoothness alone.

## Relevant Published Methods

- [Tiulpin et al., Scientific Reports (2018)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5789045/): trained a Siamese CNN on MOST and externally validated on OAI; medial/lateral patches, symmetry, CE, QWK about `0.83`, and attention maps. Stronger evidence than a random Kaggle split because it includes external validation.
- [Norman et al., Journal of Digital Imaging (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6499841/): U-Net localization, `500x500` ROIs, DenseNet ensemble, CE, and targeted augmentation for severe grades. It grouped KL 0/1, so its four-class results are not directly comparable to this five-class task.
- [Chen et al., CMIG (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9531250/): the closest expert method to this exact release; YOLOv2 ROI detection, 224 inputs, ordinal loss comparison, and qualitative Grad-CAM.
- [Zhang et al., ISBI (2020)](https://doi.org/10.1109/ISBI45749.2020.9098456): attention-based CNN for OAI KL grading, relevant if explicit multi-scale attention is tested later.
- [Shahid et al., Frontiers in Medicine (2025)](https://doi.org/10.3389/fmed.2025.1707588): independent 602-knee clinical dataset; DenseNet-121 outperformed DenseNet-201, ResNet50, and MobileNet, but its aggressive augmentation and different population prevent direct transfer of settings.

## Final Position

The project has become more complicated than the evidence supports. Keep the current DenseNet-121/CE/Grad-CAM pipeline as the baseline, but do not call its heatmaps clinically correct yet. Do not switch to DenseNet-201, do not stack imbalance methods, and do not infer that 384 creates detail. Resolve production ROI alignment first, then run the single size comparison above. That experiment can answer a real question; another mixed multi-change run cannot.
