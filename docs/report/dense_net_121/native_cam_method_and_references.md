# DenseNet-121 Native CAM Method and References

Prepared at: `2026-07-21T20:34:40+07:00` (`2026-07-21T13:34:40Z`)

Editable architecture diagram: [DenseNet-121 native CAM architecture](assets/densenet121_native_cam_architecture.drawio)

## Selected Method

The selected configuration is `canonical_final_linear_cam`:

- Right-knee images are mirrored before augmentation so left and right knees use one anatomical orientation.
- DenseNet-121 produces only its final semantic feature tensor, `F`.
- A 1x1 convolution produces five spatial maps, one for each KL grade.
- Global average pooling of map `M_k` gives class logit `z_k`.
- The displayed native CAM is the positive part of `M_k`, resized to the processed input image.
- CE, full inverse-frequency sampling, mixed precision, batch size 48, and the 5/15/10 training stages are retained from the controlled comparison.

For feature channel `c`, grade `k`, and spatial position `(x, y)`:

```text
M_k(x, y) = sum_c W[k, c] * F_c(x, y) + b[k]
z_k       = mean_(x,y)(M_k(x, y))
CAM_k     = upsample(ReLU(M_k))
```

This is a native class activation map, not Grad-CAM. The same grade map used in the visualization is spatially averaged to obtain that grade's logit. No target-layer name lookup, backward hook, or gradient-derived channel weight is required.

## Experimental Evidence

The faithful-CAM run directory was created at `2026-07-21T04:58:55.035163Z` (the supplied archive is timestamped `2026-07-21T07:01:10Z`). On its validation split, `canonical_final_linear_cam` produced:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.6550 |
| QWK | 0.8065 |
| Macro precision | 0.6897 |
| Macro recall | 0.6928 |
| Macro F1 | 0.6906 |
| Grade 1 recall | 0.3987 |
| Macro AP | 0.7154 |
| Macro AUC | 0.8862 |
| Joint-region energy enrichment | 2.0458 |
| Border-energy enrichment | 0.4803 |
| CAM/occlusion Spearman correlation | 0.5810 |
| Joint occlusion probability drop | 0.5696 |
| Border occlusion probability drop | 0.4245 |

These results support promotion of the method, but they do not prove that every heatmap is anatomically correct. The joint region is a fixed central-band proxy, not an expert joint-space segmentation. The validation split also contains only 27 Grade 4 images, so a 50-case Grade 4 validation audit is impossible without reusing cases or touching the test split. The report should use “more faithful and better localized,” not “perfect Grad-CAM.”

## Papers to Cite

The three most important citations for explaining this implementation are Zhou et al. for native CAM, Huang et al. for DenseNet, and Tiulpin et al. for radiographic KL grading.

1. B. Zhou et al., **Learning Deep Features for Discriminative Localization**, CVPR 2016. This is the core CAM paper: global average pooling followed by a linear class head makes the class-specific spatial map directly recoverable. [CVF paper](https://openaccess.thecvf.com/content_cvpr_2016/html/Zhou_Learning_Deep_Features_CVPR_2016_paper.html), [DOI](https://doi.org/10.1109/CVPR.2016.319), [arXiv](https://arxiv.org/abs/1512.04150).

2. G. Huang et al., **Densely Connected Convolutional Networks**, CVPR 2017. This defines the DenseNet-121 backbone and dense connectivity used to produce the final semantic tensor. [CVF paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html), [DOI](https://doi.org/10.1109/CVPR.2017.243), [arXiv](https://arxiv.org/abs/1608.06993).

3. A. Tiulpin et al., **Automatic Knee Osteoarthritis Diagnosis from Plain Radiographs: A Deep Learning-Based Approach**, Scientific Reports 2018. This is directly relevant to automatic KL grading from knee radiographs. [Journal paper](https://www.nature.com/articles/s41598-018-20132-7), [DOI](https://doi.org/10.1038/s41598-018-20132-7).

4. J. Antony et al., **Quantifying Radiographic Knee Osteoarthritis Severity Using Deep Convolutional Neural Networks**, ICPR 2016. This is an earlier CNN-based KL-severity study and is useful for the related-work section. [DOI](https://doi.org/10.1109/ICPR.2016.7900254), [arXiv](https://arxiv.org/abs/1609.02469).

5. R. R. Selvaraju et al., **Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization**, ICCV 2017. Cite this when explaining the previous hook-based method and why Grad-CAM is needed for nonlinear heads but not for the new spatially linear head. [CVF paper](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html), [DOI](https://doi.org/10.1109/ICCV.2017.74), [arXiv](https://arxiv.org/abs/1610.02391).

6. R. Draelos and L. Carin, **Use HiResCAM instead of Grad-CAM for faithful explanations of convolutional neural networks**, Nature Machine Intelligence 2021. This supports using activation-gradient contributions when a nonlinear or multiscale reference head cannot expose a native class map. [Journal paper](https://www.nature.com/articles/s42256-020-00262-0), [DOI](https://doi.org/10.1038/s42256-020-00262-0), [arXiv](https://arxiv.org/abs/2011.08891).

For the SE-ResNeXt comparison notebook, also cite the two component architectures:

7. S. Xie et al., **Aggregated Residual Transformations for Deep Neural Networks**, CVPR 2017. [CVF paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.html), [DOI](https://doi.org/10.1109/CVPR.2017.634), [arXiv](https://arxiv.org/abs/1611.05431).

8. J. Hu et al., **Squeeze-and-Excitation Networks**, CVPR 2018. [CVF paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html), [DOI](https://doi.org/10.1109/CVPR.2018.00745), [arXiv](https://arxiv.org/abs/1709.01507).

## Recommended Report Wording

> We modified DenseNet-121 to use a spatially linear five-class head. A 1x1 convolution produces one evidence map for each KL grade, and global average pooling of each map gives its corresponding class logit. Consequently, the displayed native CAM is derived from the classifier's own grade-specific evidence rather than from a post-hoc target-layer gradient approximation. Laterality canonicalization reduces left/right anatomical variance. Validation localization was assessed using joint- and border-energy enrichment and occlusion sensitivity; these measurements support improved localization but do not replace expert anatomical annotations.
