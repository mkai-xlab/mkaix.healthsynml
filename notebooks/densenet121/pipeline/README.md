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
| `CORN_TASK_WEIGHTS` | 01 / 02 | `[2.0, 1.8, 1.2, 1.0]` | Per-threshold weight in the CORN loss |
| `LEARNING_RATE` | 01 / 02 | 1e-4 / 1e-5 | Stage 2 is a fine-tune, hence the smaller value |
| `ALTERNATE_VIEW_PROBABILITY` | 02 | 0.50 | P(draw the YOLO ROI instead of the published crop) |
| `CHECKPOINT_OVERRIDE` | 03 | `None` | Set to a path to evaluate a specific checkpoint |

## Loss function: Focal CORN (ordinal)

All three notebooks train and decode with **Focal CORN**, not cross-entropy. The head emits
`NUM_ORDINAL_LOGITS = 4` conditional logits `P(y>0) … P(y>3)` instead of 5 class logits, and
`corn_probabilities()` turns them back into a 5-class distribution with a cumulative product, so
every downstream metric (QWK, macro F1, macro AP, Grad-CAM) is computed on a proper distribution.

| Constant | Default | Effect |
| --- | --- | --- |
| `CORN_GAMMA` | 2.0 | Focal exponent — down-weights thresholds the model already gets right |
| `CORN_ALPHA` | 0.25 | Focal scale |
| `CORN_LABEL_SMOOTHING` | 0.1 | Softens the binary threshold targets |
| `CORN_TASK_WEIGHTS` | `[2.0, 1.8, 1.2, 1.0]` | Emphasises the 0\|1 and 1\|2 boundaries, where Grade 1 is lost |

Two consequences worth knowing before you read the results:

- **These checkpoints cannot be served by the current API.** `_load_component` builds a 5-logit head
  and applies softmax; a CORN checkpoint has a 4-logit head. Promoting one means changing the head,
  the decode step, and the Grad-CAM path together.
- **Grad-CAM targets a threshold, not a class.** There is no "grade k" logit, so the CAM for grade k
  backpropagates through threshold `k-1` (and through `-threshold 0` for grade 0). Read a grade-2 CAM
  as "what made this look worse than grade 1".

## Test split discipline

The test split is used **only** in Notebook 03, once. Notebooks 01 and 02 select checkpoints on
validation. Do not use Notebook 03's metrics to choose a different epoch or preprocessing setting —
that turns the test split into a validation split and the reported numbers stop being honest.

## After Notebook 03

1. Review the Grad-CAM grids it prints. Activation should sit on the joint space, not on the ROI
   border, laterality markers, or mid-shaft bone.
2. Copy `report_row.json` into `docs/report/report.csv`.
3. Only then point `DENSENET121_CHECKPOINT_PATH` in the environment file at the new checkpoint.
