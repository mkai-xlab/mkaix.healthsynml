# AI Agent Coding Guidelines & Safeguards

This document defines standard constraints and expectations for all AI coding agents working on this Knee Osteoarthritis Deep Learning repository. Subsequent agents **MUST** read and adhere to these guidelines.

---

## 1. Automated Run Reporting & Logging
Every time a training pipeline is successfully run or its execution outputs are modified:
1. **Generate Markdown Log:** Parse the notebook's inline outputs and append the training run metrics to `/home/viet/Capstone/ml/docs/report/{model_name}/report.md`.
2. **Metadata Table:** Include a structured Markdown configuration table listing:
   * Model selection
   * Image size
   * Pipeline structure (e.g. 1-stage standard vs 3-stage)
   * Epoch count
   * Imbalance parameters (`use_balanced_sampler`, `use_minority_aug`)
   * Loss function
3. **Execution History:** Parse the complete `TRAINING HISTORY LOG SUMMARY` printed in the cell output and format it into a clean, readable Markdown table. Do not truncate the epochs.
4. **Failure Analysis:** Include the SOTA test metrics confidence intervals and full classification report.
5. **Plot Extraction:** Decode and save all generated figures (Loss curves, Confusion Matrices, Grad-CAM maps) as timestamped PNGs under `assets/` and link them in the report.
6. **Notebook Archival:** Copy the executed `.ipynb` file to the report folder, renamed using format: `{timestamp}_{model_name}_{pipeline_type}.ipynb`.

> [!TIP]
> Run the utility script `/home/viet/Capstone/ml/generate_report.py` to automate this workflow.

---

## 2. Knee OA Clinical Safeguards & Training Constraints

### A. Bounding Box & ROI Extraction (DO NOT CenterCrop)
* **The Osteophyte Constraint:** Standard center-cropping (e.g. `CenterCrop(224)`) to remove corner text shortcuts is **strictly prohibited**. 
* **The Reason:** Severe Kellgren-Lawrence grades are defined by the growth of large **osteophytes (bone spurs)** on the extreme lateral and medial margins of the joint line. Slicing off the edges of the X-ray destroys these features, dropping validation QWK by ~0.05.
* **The Defense:** To prevent shortcut learning (such as surgical pins and text markers) without removing osteophytes, rely exclusively on **aggressive Cutout / Random Erasing** (ideally double cutout at `p=0.8` or `p=0.9` during training).

### B. Standard Pipeline Checkpoint Bug
* When running standard fine-tuning (`training_pipeline = "standard"`), ensure that checkpoint saving triggers properly for `"Standard"` mode:
  ```python
  if stage_name in ["Stage 3", "Standard"]:
      # Save best_model.pth
  ```
  Failing to do so will run standard training successfully but skip saving the best checkpoint, crashing final evaluations.

---

## 3. Backend Prediction Endpoint Specs
The production FastAPI backend must orchestrate YOLOv8 knee joint ROI detection and classification together:
* **JSON Output:** Return a list of predictions (`"predictions": [...]`) containing coordinates, predicted grades, and confidence.
* **Image Output:** Return an annotated base64 image (`"annotated_image"`) with drawn green bounding boxes showing the predicted `Grade X (Confidence %)` overlay.
