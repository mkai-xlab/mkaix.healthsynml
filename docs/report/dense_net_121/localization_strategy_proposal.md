# KL Grading and Anatomical Localization Strategy

## Decision

The `2026-07-20 14:06:30 UTC` ablation established **CE + full inverse-frequency sampling** as the best predictive baseline under the predeclared validation score. It did not establish that this model is anatomically reliable. Review of all 50 exported predicted-class/true-class CAM pairs found frequent switching between joint compartments, repeated hotspots on image margins, and occasional activation on femoral texture or corners. The next model must therefore pass a localization gate before predictive metrics are used for final selection.

Grad-CAM cannot be made "exact" without an anatomical reference. For this dataset, the most defensible immediate reference is a reproducible central joint-space region because the images are already knee crops. The stronger future reference is a small manually reviewed CSV of joint-space rectangles or landmarks. About 20 images per grade would already make localization comparisons substantially more credible than visual inspection alone.

## Options

| Priority | Strategy | Expected metric effect | Expected localization effect | Suitable now? |
| --- | --- | --- | --- | --- |
| 1 | Canonicalize left/right knees before augmentation | Reduces anatomical variance and may improve Grade 0/1 separation | Keeps medial/lateral evidence on consistent sides | Yes; filenames end in `L` or `R` |
| 2 | Soft joint-ROI emphasis | Usually preserves more information than a hard crop | Penalizes reliance on superior/inferior bone and borders | Yes; central soft mask is available now |
| 3 | Central joint-strip crop | Can improve subtle joint-space-width learning but may lose large osteophytes | Strongest simple spatial constraint | Yes; compare rather than assume |
| 4 | Border randomization during training | May reduce shortcut learning with a small accuracy cost | Directly attacks the observed margin hotspots | Yes; keep the border narrow |
| 5 | Ordinal soft-label loss with ROI emphasis | May improve QWK and adjacent-grade behavior | Same spatial benefit as ROI emphasis | Yes; it was second-best in the first ablation |
| 6 | Dual-view full knee + joint strip | Often preserves global osteophytes while emphasizing joint-space narrowing | Produces a dedicated anatomical branch | Second phase; roughly doubles compute |
| 7 | CAM/gradient penalty outside an annotated ROI | Can trade a small amount of accuracy for stronger localization | Strongest direct supervision when masks are correct | After creating reviewed ROI annotations |
| 8 | Auxiliary JSN/osteophyte prediction | Often improves clinically grounded features | Directly aligns features with KL criteria | Future only; requires labels or reliable pseudo-labels |
| 9 | Bone/joint landmark alignment or segmentation | Reduces acquisition and positioning variation | Gives the best anatomical coordinate system | Future; requires a landmark/segmentation model |
| 10 | Model ensemble | Can improve QWK/F1 | Does not automatically improve or clarify localization | Only after individual models pass the localization gate |

## Controlled Experiment

The immediate experiment compares six arms under the same train/validation split, full sampler, seed, DenseNet-121 architecture, optimizer, and training schedule:

1. `ce_full_flip_aug`: current predictive baseline.
2. `ce_canonical`: deterministic left/right canonicalization and no random horizontal flip.
3. `ce_canonical_soft_roi`: canonicalization plus softly dimmed pixels outside the central joint band.
4. `ce_canonical_joint_strip`: canonicalization plus a central joint-strip crop.
5. `ce_canonical_soft_roi_border_aug`: soft ROI plus randomized narrow borders during training.
6. `softlabel_canonical_soft_roi`: ordinal soft-label loss plus the same soft ROI.

The test split is not used. Each arm reports accuracy, QWK, macro precision, macro recall, macro F1, and Grade 1 recall. It also audits up to 50 validation images per grade using joint-ROI energy enrichment, border-energy enrichment, Grad-CAM/occlusion rank correlation, and joint-versus-border occlusion drops.

Selection is constrained rather than based on one arbitrary mixed score. A candidate must first satisfy localization thresholds and must not be Pareto-dominated across predictive and localization objectives. Among candidates that pass, the predictive score ranks checkpoints. If no candidate passes, the experiment reports that no defensible winner exists.

## Evidence

- Tiulpin and Saarakkala showed the value of predicting individual radiographic OA features, supporting anatomically grounded supervision rather than KL grade alone: [Diagnostics 2020](https://doi.org/10.3390/diagnostics10110932).
- Ordinal modeling has been evaluated directly for knee OA severity classification: [Multimedia Tools and Applications 2021](https://doi.org/10.1007/s11042-021-10557-0).
- Multi-task structural-feature learning has been studied on OAI knee data: [Computer Methods and Programs in Biomedicine 2023](https://doi.org/10.1016/j.cmpb.2023.107807).
- Medical saliency studies warn that plausible maps are not necessarily trustworthy localization: [Radiology: AI 2021](https://doi.org/10.1148/ryai.2021200267) and [Nature Machine Intelligence 2022](https://doi.org/10.1038/s42256-022-00536-x).
- Explanation constraints are motivated by right-for-the-right-reasons training: [Ross et al. 2017](https://arxiv.org/abs/1703.03717).
- Guided attention supervision is represented by GAIN: [CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Li_Tell_Me_Where_CVPR_2018_paper.html).
- Saliency methods must also pass model/randomization sanity checks: [Adebayo et al. 2018](https://arxiv.org/abs/1810.03292).

