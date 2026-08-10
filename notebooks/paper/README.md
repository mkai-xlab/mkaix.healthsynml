# Paper Reproduction Notebooks

These notebooks reproduce ideas from papers on the local dataset. They do not change the production model.

- `densenet121.ipynb`: DenseNet-121 KL-grading experiment based on [the referenced paper](../../docs/paper/fmed-12-1707588.md).
- `se_resnext50_32x4d.ipynb`: SE-ResNeXt-50 KL-grading experiment inspired by Tiulpin et al.

Run each notebook from top to bottom in a fresh GPU runtime. Check the dataset paths first. Keep paper results separate from production results because the data, preprocessing, and metrics may differ.
