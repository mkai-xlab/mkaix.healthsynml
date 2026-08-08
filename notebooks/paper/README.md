# Paper Reproduction Notebooks

These notebooks are isolated paper-reproduction experiments and do not change the production training or inference notebooks.

## Files

- `densenet121.ipynb` implements the DenseNet-121 KL-grade recipe described in `docs/paper/fmed-12-1707588.md`. It can run on the local `kneeKL224` image dataset. The paper used a different 301-patient clinical dataset, so its published scores are not expected to reproduce.
- `se_resnext50_32x4d.ipynb` is a single-head KL-grade adaptation of the Tiulpin et al. CNN recipe: SE-ResNeXt-50, five subject-grouped folds, one five-class KL head, frozen-convolution warm-up, 20-epoch fine-tuning, paper-inspired augmentation, and the reported Adam schedule. It uses the local `kneeKL224` dataset and does not predict future progression or use clinical data.

Run every cell from top to bottom in a fresh Colab GPU runtime. Check the configuration cells before starting because the dataset paths are Google Drive paths. Outputs are written to timestamped folders under `Models/paper_*`.

The paper notebooks use paper-specific preprocessing and metrics. Do not compare their results directly with production QWK/CAM runs without recording the dataset, target definition, split, and preprocessing differences.
