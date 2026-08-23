# DenseNet-121 Pipeline

Three notebooks that take DenseNet-121 from ImageNet weights to a checkpoint the API can serve.
Run them in order — each one reads the checkpoint the previous one selected.

| Order | Notebook | Trains on | Writes to Drive | Approx. runtime (T4) |
|---|---|---|---|---|
| 1 | `01_train_original.ipynb` | Published crops, 384×384 | `Models/densenet121_original/<timestamp>/` | ~45 min (12 epochs) |
| 2 | `02_train_paired_roi.ipynb` | 50% published / 50% YOLO ROI | `Models/densenet121_paired_roi/<timestamp>/` | ~20 min (5 epochs) |
| 3 | `03_evaluate_roi_test.ipynb` | Nothing — evaluation only | `Models/densenet121_evaluation/<timestamp>/` | ~5 min |

## How the notebooks find each other

Notebook 01 writes `SELECTED_CHECKPOINT.txt` into its run directory. Notebook 02 reads the **newest**
such pointer instead of a hard-coded timestamp, and does the same for Notebook 03. Re-running an
earlier stage therefore feeds the newer checkpoint forward automatically — no path editing.

To pin a specific historical checkpoint, set `CHECKPOINT_OVERRIDE` in Notebook 03.

## Required Drive layout

```
MyDrive/Datasets/KneeXrayData_Mendeley_v1/
├── extracted/KneeXrayData/ClsKLData/kneeKL224/   # published crops
│   ├── train/{0,1,2,3,4}/*.png
│   ├── val/{0,1,2,3,4}/*.png
│   └── test/{0,1,2,3,4}/*.png
└── derived/densenet121_yolo_square_roi_trainvaltest_v2/   # YOLO ROI crops
    ├── train/{0,1,2,3,4}/*.png
    ├── val/{0,1,2,3,4}/*.png
    └── test/{0,1,2,3,4}/*.png
```

The ROI folder is produced by `notebooks/datasets/01_prepare_original_and_yolo_roi_datasets.ipynb`.
Filenames must match between the two trees; Notebook 02 raises `FileNotFoundError` on the first
unpaired image rather than silently dropping it.

## Configuration

Every tunable value lives in one uppercase block near the top of each notebook. The values that
matter most:

| Constant | Notebook | Default | Notes |
|---|---|---|---|
| `INPUT_SIZE` | all | 384 | Must match `IMG_SIZE` in the API `.env` |
| `EPOCHS` | 01 / 02 | 12 / 5 | |
| `LEARNING_RATE` | 01 / 02 | 1e-4 / 1e-5 | Stage 2 is a fine-tune, hence the smaller value |
| `ALTERNATE_VIEW_PROBABILITY` | 02 | 0.50 | P(draw the YOLO ROI instead of the published crop) |
| `CHECKPOINT_OVERRIDE` | 03 | `None` | Set to a path to evaluate a specific checkpoint |

## Test split discipline

The test split is used **only** in Notebook 03, once. Notebooks 01 and 02 select checkpoints on
validation. Do not use Notebook 03's metrics to choose a different epoch or preprocessing setting —
that turns the test split into a validation split and the reported numbers stop being honest.

## After Notebook 03

1. Review the Grad-CAM grids it prints. Activation should sit on the joint space, not on the ROI
   border, laterality markers, or mid-shaft bone.
2. Copy `report_row.json` into `docs/report/report.csv`.
3. Only then point `DENSENET121_CHECKPOINT_PATH` in the environment file at the new checkpoint.
