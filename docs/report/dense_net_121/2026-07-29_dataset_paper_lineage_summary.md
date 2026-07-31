# Dataset and Reference-Paper Lineage Summary

**Date:** 2026-07-29  
**Local dataset checked:** `/home/viet/Downloads/kaggle_knee_osteoarthritis`

## Conclusion

The local Kaggle dataset is **almost certainly a repackaging of the same OAI-derived KneeXrayData release used by Chen et al. (2019)**. It is not possible to claim byte-for-byte identity without checksums for the original Kaggle and Mendeley archives. Kaggle may have converted, resized, or reorganized the files.

Confidence that the underlying subjects, labels, and published split come from the same release is high because all of the following match:

| Evidence | Local Kaggle folder | Chen/Mendeley release |
| --- | ---: | ---: |
| Training knees | 5,778 | 70% of 8,260 = 5,782 nominally; published packaged inventory is 5,778 |
| Validation knees | 826 | 10% of 8,260 = 826 |
| Test knees | 1,656 | 20% of 8,260 = 1,652 nominally; published packaged inventory is 1,656 |
| Total labeled knees | **8,260** | **8,260** |
| Bilateral radiographs | Implied by paired `L/R` files | **4,130** |
| Filename convention | OAI patient ID plus `L` or `R` | Same patient/side convention |
| Grades | KL 0–4 | KL 0–4 |
| Source | Kaggle description: OAI-derived | OAI baseline radiographs |

The small differences from purely multiplying the 7:1:2 percentages are expected because splitting is performed at the bilateral-image level and stratified by one knee's grade, not by independently allocating every knee.

## The Paper We Should Cite

The primary dataset/method citation is:

> Chen P, Gao L, Shi X, Allen K, Yang L. **Fully Automatic Knee Osteoarthritis Severity Grading Using Deep Neural Networks with a Novel Ordinal Loss.** Computerized Medical Imaging and Graphics. 2019;75:84–92.

- [Open-access paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9531250/)
- [Original KneeAnalysis repository](https://github.com/PingjunChen/KneeAnalysis)
- [Original Mendeley KneeXrayData release](https://data.mendeley.com/datasets/56rmx5bjcr/1)

## Chen et al. Method

1. Standardized bilateral OAI radiographs to a common physical resolution and retained 4,130 images with both knee labels.
2. Split the bilateral images 7:1:2, producing 8,260 labeled knees.
3. Detected both knee joints using a customized YOLOv2 model.
4. Expanded each detected/manual bounding box by a factor of `1.3`.
5. Resized crops to `224x224` for VGG, ResNet, and DenseNet (`299x299` for InceptionV3).
6. Compared cross-entropy with an ordinal loss that penalized distant-grade errors more strongly.
7. Reported VGG-19 plus ordinal loss as the best overall model, about `69.7%` accuracy and `0.344` MAE.
8. Used Grad-CAM on selected test examples; those images were qualitative evidence, not an all-case localization audit.

Their DenseNet results did not favor the deeper model:

| Model | Manual CE | Manual ordinal | Automatic CE | Automatic ordinal |
| --- | ---: | ---: | ---: | ---: |
| DenseNet-121 | 67.3% | **68.2%** | **67.4%** | **67.8%** |
| DenseNet-169 | 66.8% | 67.1% | 66.6% | 65.4% |
| DenseNet-201 | 65.7% | 67.3% | 66.3% | 66.3% |

## Are We Using Their Exact Configuration?

**No.** We use the same task and almost certainly the same underlying labeled release, but our current configuration is different.

| Component | Chen et al. | Current project |
| --- | --- | --- |
| Detector | Customized YOLOv2 | Production YOLOv8 |
| Crop | Box expanded by `1.3` | YOLOv8 crop policy |
| DenseNet input | `224x224` | `384x384` |
| Preprocessing | Training-set normalization | CLAHE 1.25, square pad, ImageNet normalization |
| Main comparison | CE versus novel ordinal loss | CE selected after local loss ablations |
| Imbalance | Not the current sampler recipe | Full inverse-frequency sampler |
| Augmentation | Paper-specific training setup | Flip, mild rotation/jitter, conservative erasing |
| Heatmap evidence | Selected qualitative Grad-CAMs | Attempted complete quantitative Grad-CAM audit |

Therefore, the project is **based on the same dataset lineage and two-stage detection/classification idea**, not a reproduction of their exact model.

## Why Their Heatmaps Can Look Better

The paper displayed a small number of selected Grad-CAM examples from its own test distribution. It did not publish a failure rate over all 1,656 test knees. Our difficult heatmaps mainly occur on external images cropped by a different YOLO model and are judged with a strict anatomy gate. The visual evidence is therefore not directly comparable.

Our current DenseNet-121 accuracy, around `67%`, is already close to the paper's DenseNet range. The unresolved problem is external ROI/domain alignment and clinically valid localization, not evidence that their classifier was dramatically more accurate.

## Reporting Language

Use this statement in the capstone report:

> We used a Kaggle-packaged version of the OAI-derived KneeXrayData severity-grading release associated with Chen et al. (2019). The local inventory matches the published 4,130 bilateral radiographs and 8,260 knee labels. Because archive-level checksums were unavailable, we describe the Kaggle package as derived from the same release rather than claiming byte-for-byte identity.
