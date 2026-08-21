# Knee Osteoarthritis Dataset Report

> **Pipeline:** YOLOv8 (knee detection) → DenseNet-121 / SE-ResNeXt-50 (KL grading)
> **Sources:** [NIH OAI](https://nda.nih.gov/oai) · [Mendeley doi:10.17632/56rmx5bjcr.1](https://data.mendeley.com/datasets/56rmx5bjcr/1) · [Roboflow Knee Xray Yolo](https://app.roboflow.com/nguyens-workspace-tm7at/knee-joint-84k23)
> **Generated:** `ml/notebooks/tools/dataset_analysis.ipynb` (verified counts, 2026-08-17)

---

## 1. Origin

```
NIH / NIAMS + NIA
        │
        ▼
OAI — Osteoarthritis Initiative  (nda.nih.gov/oai)
  Longitudinal cohort, 2004–2014
  ~4,796 participants, 45–79 yrs, US
  Raw: bilateral PA fixed-flexion DICOM
  KL grades: Boston University X-ray Reading Center (3 radiologists, majority vote)
        │
        ▼  Chen et al. (2018), CC BY 4.0
Mendeley KneeXrayData  (doi:10.17632/56rmx5bjcr.1)
  YOLOv2 detection → 224×224 single-knee crops
  ~9,786 crops from MULTIPLE OAI visits
        │
        ├──────────────────────────────┐
        ▼                              ▼
Kaggle re-upload              KneeXrayData_Mendeley_v1 (Drive ZIP)
  9,786 files                  Single-visit subset: 8,260 crops
  (all visits)               4,130 bilateral H5
```

> **Why 8,260 instead of 9,786?** Mendeley uses multiple OAI visits (baseline + 12M + … + 96M). This project uses only one visit per patient → 4,130 bilaterals → 8,260 single-knee crops. Avoids temporal leakage.

---



## 2. Quantity



### Dataset 1 — YOLO Detection (knee joint detection)

Used to train YOLOv8n for localising knee joints.


| Split       | Images    | Note                        |
| ----------- | --------- | --------------------------- |
| **Train**   | 461       | Actual count (YOLO scan log) |
| **Valid**   | 58        | Actual count (YOLO scan log) |
| **Test**    | 0         | No held-out test split       |
| **Total**   | **519**   |                             |


- **Source:** [Roboflow — Knee Xray Yolo](https://app.roboflow.com/nguyens-workspace-tm7at/knee-joint-84k23) (`Knee Xray Yolo.yolov8.zip`)
- **Workspace:** `nguyens-workspace-tm7at` · **Project:** `knee-joint-84k23` · **Version:** `dataset`
- **Classes:** 1 — `joint` (`nc=1`)
- **Format:** YOLOv8 format (`images/` + `labels/` folders, normalised `x_center y_center width height`)
- **Verified counts (from `yolo detect train` log, 2026-08):** `train/labels` → 461 images, `valid/labels.cache` → 58 images — NOT the 946/315 values previously listed here.
- **No `test/` split in YOLO zip:** External/clinical test must be sourced separately (e.g. re-annotation on Mendeley H5 or OAI downloaded images).
- **Random-seed reproducibility:** `seed=42`, `deterministic=True`, `patience=100` (CLI override; original `TrainingConfig.patience=10` was bypassed by CLI args), `close_mosaic=10`, `amp=True` mixed precision.
- **Used by:** `yolov8_knee_detection_cli.ipynb` → produces `best.pt` checkpoint; loaded by `ml/app/services/roi_service.py::KneeJointDetectionService`
- **Reported metrics (validation, 58 images, 117 instances):** `Box(P)=1.000, R=0.991, mAP50=0.995, mAP50-95=0.902`. Single-class detection; no held-out test set; CI not reported.



### Dataset 2 — CNN Classification (KL grading)

Used to train DenseNet-121 / SE-ResNeXt-50 for KL grade prediction.


| Dataset                            | Total images | Bilateral originals | Resolution           | Color     |
| ---------------------------------- | ------------ | ------------------- | -------------------- | --------- |
| **kneeKL224** (Mendeley published) | 8,260        | 4,130               | 224×224 (uniform)    | Grayscale |
| **YOLO Square ROI** (re-processed) | 8,260        | 4,130               | ~70–73 px (variable) | RGB       |




#### CNN dataset split breakdown


| Split     | Bilateral | Single-knee | %     |
| --------- | --------- | ----------- | ----- |
| **Train** | 2,889     | 5,778       | 70.0% |
| **Val**   | 413       | 826         | 10.0% |
| **Test**  | 828       | 1,656       | 20.0% |
| **Total** | **4,130** | **8,260**   | 100%  |


- **Format:** PNG, folder-per-grade (0–4)
- **KL grades:** 0–4 (source: folder name)
- **YOLO Square ROI YOLO ratio:** 1.00× (both knees detected in all 4,130 bilaterals)

---



## 3. Class Distribution (CNN Dataset)

KL grades assigned by 3 board-certified radiologists; **ordinal** (KL-3 > KL-2).


| Grade | Label     | Train     | Val     | Test      | Total     | %     |
| ----- | --------- | --------- | ------- | --------- | --------- | ----- |
| **0** | Healthy   | 2,286     | 328     | 639       | 3,253     | 39.4% |
| **1** | Doubtful  | 1,046     | 153     | 296       | 1,495     | 18.1% |
| **2** | Minimal   | 1,516     | 212     | 447       | 2,175     | 26.3% |
| **3** | Moderate  | 757       | 106     | 223       | 1,086     | 13.1% |
| **4** | Severe    | 173       | 27      | 51        | 251       | 3.0%  |
|       | **Total** | **5,778** | **826** | **1,656** | **8,260** | 100%  |


```
KL-0 Healthy   ████████████████████████████████████  39.4%
KL-1 Doubtful   ████████████                         18.1%
KL-2 Minimal    ███████████████████                  26.3%
KL-3 Moderate   ██████████                            13.1%
KL-4 Severe     ██                                      3.0%
```

**Imbalance:** KL-0 : KL-4 = **13.2 : 1** (train) · KL-0 : KL-4 = **13.0 : 1** (overall)
Class proportions are consistent across train/val/test — confirms a proper stratified split.

---



## 4. Folder Architecture



### YOLO Detection Dataset (Dataset 1)

Actual layout extracted from `yolo detect train` scan log on Colab (Tesla T4, ultralytics 8.4.106):

```
/content/Datasets/                          ← Knee Xray Yolo.yolov8.zip (Roboflow)
│
├── data.yaml                               ← YOLOv8 dataset config
│                                           classes: [joint]   (nc=1)
│                                           train: train/images
│                                           val:   valid/images
│                                           roboflow:
│                                             workspace: nguyens-workspace-tm7at
│                                             project:   knee-joint-84k23
│                                             version:   dataset
│
├── train/
│   ├── images/  (461 PNGs)                 ← Knee joint crops, variable size
│   └── labels/  (461 TXT YOLO txts)         ← x_center y_center width height (normalized)
│
└── valid/
    ├── images/  (58 PNGs)
    └── labels/  (58 TXT YOLO txts)         ← cache file: labels.cache (58 entries)

NOTE: earlier docs reported 946/315, but the YOLO CLI scanned 461/58.
      This is the version actually used for the production best.pt checkpoint.
```



### CNN Classification Dataset (Dataset 2)

```
/content/drive/MyDrive/Datasets/KneeXrayData_Mendeley_v1/
│
├── KneeXrayData.zip                       ← Downloaded ZIP (Mendeley)
│
├── extracted/KneeXrayData/
│   │
│   ├── ClsKLData/kneeKL224/               ← READ-ONLY — source of truth for labels
│   │   │                                   Original Mendeley 224×224 crops
│   │   ├── train/{0,1,2,3,4}/           (2,286 / 1,046 / 1,516 / 757 / 173 PNGs)
│   │   ├── val/{0,1,2,3,4}/              (  328 /   153 /   212 / 106 /  27 PNGs)
│   │   └── test/{0,1,2,3,4}/              (  639 /   296 /   447 / 223 /  51 PNGs)
│   │       Naming: {patient_id}{L|R}.png  e.g. 9059946L.png
│   │       Labels from folder name (0–4)
│   │
│   └── DetKneeData/H5/                     ← READ-ONLY — original bilateral HDF5
│       ├── trainH5/{patient_id}.h5            (2,889 files)
│       ├── valH5/{patient_id}.h5              (  413 files)
│       └── testH5/{patient_id}.h5             (  828 files)
│           Note: no L/R suffix — one file = both knees
│
└── derived/                                ← Generated by 01_prepare_*.ipynb
    │
    ├── full_bilateral_png_v2/              Cell 7 — H5 → bilateral PNG (intermediate)
    │   ├── train/{patient_id}.png             (2,889 files, ~195 MB)
    │   ├── val/{patient_id}.png               (  413 files,  ~29 MB)
    │   └── test/{patient_id}.png              (  828 files,  ~57 MB)
    │       Total: 4,130 files, ~514 MB
    │       Naming: {patient_id}.png  (no L/R — bilateral)
    │
    └── densenet121_yolo_square_roi_trainvaltest_v2/
                                          Cell 9 — YOLOv8n best.pt → square ROI crops
        ├── train/{0,1,2,3,4}/               (same counts as kneeKL224)
        ├── val/{0,1,2,3,4}/
        └── test/{0,1,2,3,4}/
        Total: 8,260 files, ~124 MB
            Naming: {patient_id}{L|R}.png  (L/R from YOLO x1 sort order)
```



### Naming conventions


| Dataset               | Filename           | L/R?    | Role                            |
| --------------------- | ------------------ | ------- | ------------------------------- |
| kneeKL224             | `{patient_id}{L    | R}.png` | Yes                             |
| DetKneeData/H5        | `{patient_id}.h5`  | No      | Bilateral; one file = two knees |
| full_bilateral_png_v2 | `{patient_id}.png` | No      | Bilateral PNG; intermediate     |
| YOLO Square ROI       | `{patient_id}{L    | R}.png` | Yes                             |




### YOLO ROI processing parameters (cell 9)


| Parameter            | Value                                                       |
| -------------------- | ----------------------------------------------------------- |
| Model                | `yolov8n.pt` fine-tuned on custom knee X-ray ROI dataset    |
| Confidence           | 0.45                                                        |
| Image size           | 640                                                         |
| Batch size           | 32                                                          |
| Detections per image | Exactly 2 (raises error if ≠ 2)                             |
| Left/Right           | Sorted by `x1` → smaller = L, larger = R                    |
| Box expansion        | `BOX_EXPANSION = 1.15`                                      |
| Square method        | `side = ceil(max(w, h) × 1.15)`, black pad if out-of-bounds |


---



## 5. Limitations


| #   | Issue                                                            | Impact                                                                |
| --- | ---------------------------------------------------------------- | --------------------------------------------------------------------- |
| 1   | **Class imbalance** — KL-4 = 3.0%, 13:1 vs KL-0                  | Mitigated by WeightedRandomSampler, focal loss, minority oversampling |
| 2   | **Single-visit subset** — only one OAI timepoint                 | Avoids temporal leakage; limits dataset size vs full Mendeley         |
| 3   | **Bilateral → two labels** — both knees share the same KL grade  | Two crops from one image are not fully independent                    |
| 4   | **OAI-only population** — US participants, fixed-flexion PA view | May not generalise to other views (lateral, standing) or populations  |
| 5   | **No external validation** — test set is in-distribution (OAI)   | External validation on MOST or other cohorts would strengthen results |
| 6   | **YOLO detection dataset is tiny and has no test split** — 461 train + 58 valid, no `test/` | Detector mAP50-95=0.902 is measured on a 58-image validation set only; CI is unreported and there is no held-out test to confirm generalisation. Production inputs from non-OAI equipment (DICOM, paediatric, lateral view, post-surgical implants) have not been evaluated. |


---



## References


| Citation                                                                                                  | URL                                                                                   | License            |
| --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------ |
| Chen P (2018). Knee Osteoarthritis Severity Grading Dataset. Mendeley Data, V1. doi:10.17632/56rmx5bjcr.1 | [mendeley.com](https://data.mendeley.com/datasets/56rmx5bjcr/1)                       | CC BY 4.0          |
| NIH. Osteoarthritis Initiative (OAI) Study. NDA.                                                          | [nda.nih.gov/oai](https://nda.nih.gov/oai)                                            | OAI Data Use Terms |
| Knee Xray Yolo. Roboflow.                                                                                 | [app.roboflow.com](https://app.roboflow.com/nguyens-workspace-tm7at/knee-joint-84k23) | Roboflow Terms     |
| Chen et al. (2019), *Comput. Med. Imag. Graphics*                                                         | doi:10.1016/j.compmedimag.2019.06.002                                                 | —                  |
| Tiulpin et al. (2019), *Sci. Rep.*                                                                        | doi:10.1038/s41598-019-56527-3                                                        | —                  |


