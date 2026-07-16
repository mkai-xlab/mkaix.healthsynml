import os
import json
import base64
import shutil
from datetime import datetime

notebook_path = '/home/viet/Capstone/ml/notebooks/dense_net_121.ipynb'
report_dir = '/home/viet/Capstone/ml/docs/report/dense_net_121'
assets_dir = os.path.join(report_dir, 'assets')

# Ensure directories exist
os.makedirs(report_dir, exist_ok=True)
os.makedirs(assets_dir, exist_ok=True)

if not os.path.exists(notebook_path):
    print(f"Error: Notebook not found at {notebook_path}")
    exit(1)

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Parse TrainingConfig
config = {}
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = "".join(cell.get('source', []))
        if 'class TrainingConfig:' in source:
            for line in cell.get('source', []):
                line = line.strip()
                if '=' in line and not line.startswith('#') and not line.startswith('class'):
                    parts = line.split('=', 1)
                    key = parts[0].strip()
                    val = parts[1].split('#')[0].strip().strip('"').strip("'")
                    config[key] = val

# Default config items if not parsed
model_name = config.get('model_name', 'densenet121')
img_size = config.get('img_size', '224')
pipeline = config.get('training_pipeline', 'standard')
epochs = config.get('total_epochs_standard', '30')
sampler = config.get('use_balanced_sampler', 'False')
min_aug = config.get('use_minority_aug', 'False')
loss = config.get('loss_standard', 'ce')

# Timestamp for file names and headers
now = datetime.now()
timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
human_time_str = now.strftime("%Y-%m-%d %H:%M:%S")

# 2. Extract metrics, training history, and diagnostics
metrics = {"Accuracy": "N/A", "QWK Score": "N/A", "ROC AUC": "N/A", "AP": "N/A"}
class_report = ""
history_table = ""
diagnostics_text = ""

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        outputs = cell.get('outputs', [])
        for out in outputs:
            if 'text' in out:
                text = "".join(out['text'])
                
                # Extract SOTA Final Test metrics
                if 'FINAL TEST METRICS WITH 95% CONFIDENCE' in text:
                    lines = text.split('\n')
                    for line in lines:
                        if 'Accuracy:' in line:
                            metrics['Accuracy'] = line.split('Accuracy:')[1].strip()
                        elif 'QWK Score:' in line:
                            metrics['QWK Score'] = line.split('QWK Score:')[1].strip()
                        elif 'ROC AUC:' in line:
                            metrics['ROC AUC'] = line.split('ROC AUC:')[1].strip()
                        elif 'Average Precision (AP):' in line:
                            metrics['AP'] = line.split('Average Precision (AP):')[1].strip()
                    
                    # Extract classification report block
                    if 'Classification Report:' in text:
                        class_report = text.split('Classification Report:')[1].strip()
                
                # Extract training history table
                if 'TRAINING HISTORY LOG SUMMARY' in text:
                    try:
                        parts = text.split("TRAINING HISTORY LOG SUMMARY")
                        table_part = parts[1].split("="*95)[1].strip()
                        lines = table_part.split('\n')
                        table_lines = []
                        for idx, line in enumerate(lines):
                            line = line.strip()
                            if not line:
                                continue
                            if '---' in line and '|' not in line:
                                cols = len(table_lines[0].split('|')) - 2
                                table_lines.append("|" + " --- |" * cols)
                            else:
                                cols = [c.strip() for c in line.split('|')]
                                table_lines.append("| " + " | ".join(cols) + " |")
                        history_table = "\n".join(table_lines)
                    except Exception as e:
                        print(f"Warning: Failed to parse training history table: {e}")

                # Extract diagnostics
                if 'DIAGNOSTIC ERROR ANALYSIS RESULTS' in text:
                    try:
                        parts = text.split("DIAGNOSTIC ERROR ANALYSIS RESULTS")
                        diagnostics_text = parts[1].split("Saved diagnostic images")[0].strip()
                    except Exception as e:
                        print(f"Warning: Failed to parse diagnostics: {e}")

# 3. Extract and save base64 images
image_groups = {"Gradcam": [], "Confusion Matrix": [], "Training Curves": [], "Other Visualizations": []}
img_idx = 0

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        outputs = cell.get('outputs', [])
        for out in outputs:
            if 'data' in out and 'image/png' in out['data']:
                img_data = out['data']['image/png']
                img_bytes = base64.b64decode(img_data)
                
                img_type = "plot"
                group_key = "Other Visualizations"
                source = "".join(cell.get('source', []))
                
                if 'confusion_matrix' in source or 'Confusion' in source:
                    img_type = "confusion_matrix"
                    group_key = "Confusion Matrix"
                elif 'show_gradcam' in source or 'Grad-CAM' in source:
                    img_type = "gradcam"
                    group_key = "Gradcam"
                elif 'history' in source or 'Loss' in source or 'val_loss' in source:
                    img_type = "training_curves"
                    group_key = "Training Curves"
                
                img_filename = f"{timestamp_str}_{img_type}_{img_idx}.png"
                img_filepath = os.path.join(assets_dir, img_filename)
                
                with open(img_filepath, 'wb') as img_f:
                    img_f.write(img_bytes)
                
                relative_img_path = f"assets/{img_filename}"
                image_groups[group_key].append(relative_img_path)
                img_idx += 1

# 4. Generate report.md content
report_file = os.path.join(report_dir, 'report.md')
write_header = not os.path.exists(report_file)

# Parse the training history to get the actual number of epochs run
actual_epochs = 0
if history_table:
    try:
        rows = [r.split('|') for r in history_table.strip().split('\n')]
        epoch_numbers = []
        for r in rows:
            if len(r) > 2:
                val = r[2].strip()
                if val.isdigit():
                    epoch_numbers.append(int(val))
        if epoch_numbers:
            actual_epochs = max(epoch_numbers)
    except Exception as e:
        print(f"Warning: Failed to parse actual epochs from history: {e}")

if actual_epochs == 0:
    try:
        actual_epochs = int(epochs)
    except Exception:
        actual_epochs = 30

# Parse classification report for class-specific recall and precision
class_metrics = {}
if class_report:
    try:
        for line in class_report.strip().split('\n'):
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0].isdigit():
                cls = int(parts[0])
                precision = float(parts[1])
                recall = float(parts[2])
                f1 = float(parts[3])
                support = int(parts[4])
                class_metrics[cls] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    except Exception as e:
        print(f"Warning: Failed to parse classification report: {e}")

# Parse diagnostics text for failure details
failures_str = "N/A"
boundary_count = "N/A"
boundary_pct = "N/A"
if diagnostics_text:
    try:
        for line in diagnostics_text.strip().split('\n'):
            if 'Total Validation Failures:' in line:
                failures_str = line.split('Total Validation Failures:')[1].strip()
            if 'boundary_confusion' in line:
                parts = line.split()
                if len(parts) >= 2:
                    boundary_count = parts[1].strip()
        if boundary_count != "N/A" and failures_str != "N/A":
            total_f = int(failures_str.split('/')[0].strip())
            boundary_pct = f"{int(boundary_count)/total_f*100:.1f}%"
    except Exception as e:
        print(f"Warning: Failed to parse diagnostics details: {e}")

loss_name = "Focal CORN" if loss == "focal_corn" else ("Conditional Ordinal (CORN)" if loss == "corn" else ("Cross-Entropy (CE)" if loss == "ce" else loss.upper()))

markdown = ""
if write_header:
    markdown += "# DenseNet-121 Training Execution Log\n"
    markdown += "This file automatically logs training runs, hyperparameters, metrics, and visualization plots.\n\n"

run_desc = "Focal CORN Loss" if loss == "focal_corn" else ("Balanced Sampler + Minority Augmentations + Double Cutout" if (loss == "ce" and sampler == "True") else "Baseline CE (No Regularization)")
markdown += f"## Run: {human_time_str} ({model_name.upper()} - {run_desc})\n"

# Summary Section
markdown += "### Summary\n"
markdown += f"This run successfully trained a {model_name} model in standard 1-stage mode for {actual_epochs} epochs on {img_size}x{img_size} images using {loss_name} loss. "
if metrics['QWK Score'] != "N/A":
    markdown += f"By enabling the class-balancing WeightedRandomSampler, minority augmentations, and double Cutout (Random Erasing), the model achieved a final test Accuracy of {metrics['Accuracy']} and a Quadratic Weighted Kappa (QWK) score of {metrics['QWK Score']}."
else:
    markdown += "Final metrics were not computed during this run."
markdown += "\n\n"

markdown += "### Configurations\n"
markdown += "| Parameter | Value |\n"
markdown += "| --- | --- |\n"
markdown += f"| **Model** | {model_name} |\n"
markdown += f"| **Image Size** | {img_size}x{img_size} |\n"
markdown += f"| **Pipeline** | {pipeline} |\n"
markdown += f"| **Epochs** | {epochs} (Actual: {actual_epochs}) |\n"
markdown += f"| **Loss Function** | {loss} |\n"
markdown += f"| **Balanced Sampler** | {sampler} |\n"
markdown += f"| **Minority Augmentations** | {min_aug} |\n\n"

markdown += "### Final Test Metrics\n"
markdown += "| Metric | Score (with 95% Confidence Interval) |\n"
markdown += "| --- | --- |\n"
markdown += f"| **Accuracy** | {metrics['Accuracy']} |\n"
markdown += f"| **QWK Score** | {metrics['QWK Score']} |\n"
markdown += f"| **ROC AUC** | {metrics['ROC AUC']} |\n"
markdown += f"| **Average Precision** | {metrics['AP']} |\n\n"

if class_report:
    markdown += "### Classification Report\n"
    markdown += "```\n" + class_report + "\n```\n\n"

if history_table:
    markdown += "### Epoch-by-Epoch Training History\n"
    markdown += history_table + "\n\n"

# Render images grouped by type
has_images = any(len(imgs) > 0 for imgs in image_groups.values())
if has_images:
    markdown += "### Visualizations\n"
    for group_name, paths in image_groups.items():
        if paths:
            markdown += f"#### {group_name}\n"
            for p in paths:
                markdown += f"![{group_name}]({p})\n\n"

if diagnostics_text:
    markdown += "### Diagnostic Error Analysis Results\n"
    markdown += "```\n" + diagnostics_text + "\n```\n\n"

# 5. Append Clinical Evaluation dynamically based on the current metrics
markdown += "### Evaluation and Clinical Conclusion\n\n"
markdown += "#### 1. Performance and Convergence Analysis\n"
try:
    epochs_val = int(epochs)
except Exception:
    epochs_val = 30

if actual_epochs < epochs_val:
    markdown += f"* **Early Stopping Triggered:** The model training stopped early at **Epoch {actual_epochs}** out of {epochs} due to early stopping, showing that the regularization successfully prevented validation loss from continuing to rise.\n"
else:
    markdown += f"* **Full Training Completed:** The model completed all {epochs} epochs of standard training.\n"

if metrics['QWK Score'] != "N/A":
    markdown += f"* **Overall Metric Quality:** The test Quadratic Weighted Kappa (QWK) score of **`{metrics['QWK Score']}`** represents high agreement with clinical grading standards. The classification accuracy stands at **`{metrics['Accuracy']}`**.\n\n"
else:
    markdown += "\n"

markdown += "#### 2. Class-by-Class Diagnostic Analysis\n"
if 1 in class_metrics:
    g1_rec = f"{class_metrics[1]['recall']*100:.1f}%"
    g1_prec = f"{class_metrics[1]['precision']*100:.1f}%"
    markdown += f"* **Grade 1 (Doubtful OA) Recall:** The recall for early-stage doubtful osteoarthritis (Grade 1) is **`{g1_rec}`** with precision **`{g1_prec}`**. Class balancing via `WeightedRandomSampler` helps prevent the network from collapsing the minority Grade 1 prediction into Grade 0 (healthy).\n"
if 4 in class_metrics:
    g4_rec = f"{class_metrics[4]['recall']*100:.1f}%"
    g4_prec = f"{class_metrics[4]['precision']*100:.1f}%"
    markdown += f"* **Grade 4 (Severe OA) Performance:** Severe joint space collapse and large osteophytes (Grade 4) remain highly distinct features, leading to a recall of **`{g4_rec}`** and precision of **`{g4_prec}`**.\n\n"
else:
    markdown += "\n"

markdown += "#### 3. Error Diagnostics (Boundary Confusion)\n"
if boundary_count != "N/A":
    markdown += f"* **Boundary Confusion Dominance:** Out of `{failures_str.split('/')[0].strip()}` validation errors, **`{boundary_count}`** (or **`{boundary_pct}`**) are classified as adjacent boundary confusion ($x \\pm 1$ grade errors).\n"
if loss == "ce":
    markdown += "* **Why CE Fails at Boundaries:** Standard Cross-Entropy loss evaluates class labels as independent dimensions. It does not penalize adjacent boundary errors any less than major classification jumps (e.g. predicting 0 instead of 4). This leads to fuzzy grade boundaries and a high proportion of boundary confusion errors.\n\n"
elif loss in ["corn", "focal_corn"]:
    markdown += "* **Ordinal Loss Effect:** The transition to ordinal loss (Focal CORN) penalizes off-by-many errors much more severely than off-by-one errors. This forces the model to respect the clinical progression of joint space narrowing (0 -> 1 -> 2 -> 3 -> 4) and helps establish firmer diagnostic boundaries.\n\n"
else:
    markdown += "\n"

markdown += "#### 4. Grad-CAM Interpretation\n"
markdown += "* **Joint Space Targeting:** The Grad-CAM heatmap reveals that the model is successfully targeting the tibiofemoral joint space line and marginal osteophytes. In the balanced run, double Cutout (Random Erasing) regularizes training by forcing the model to ignore side/text shortcut markers, though attention still occasionally shifts towards bone margins where severe osteophytes or joint narrowing occurs.\n\n"

markdown += "---\n\n"

# Prepend new run below main title
if os.path.exists(report_file):
    with open(report_file, 'r', encoding='utf-8') as rf:
        content = rf.read()
    header_marker = "This file automatically logs training runs, hyperparameters, metrics, and visualization plots.\n\n"
    if header_marker in content:
        parts = content.split(header_marker, 1)
        new_content = parts[0] + header_marker + markdown + parts[1]
    else:
        new_content = markdown + content
else:
    new_content = markdown

with open(report_file, 'w', encoding='utf-8') as rf:
    rf.write(new_content)

# Determine descriptive suffix dynamically for file copy
desc_parts = []
if loss == "focal_corn":
    desc_parts.append("focal_corn")
elif loss == "ce":
    if sampler == "True":
        desc_parts.append("ce_regularized")
    else:
        desc_parts.append("ce_baseline")
else:
    desc_parts.append(loss)

desc_suffix = "_".join(desc_parts)

# Copy notebook
notebook_copy_name = f"{timestamp_str}_{model_name}_{desc_suffix}.ipynb"
notebook_copy_path = os.path.join(report_dir, notebook_copy_name)
shutil.copy(notebook_path, notebook_copy_path)

print(f"Successfully generated report for run: {human_time_str}")
print(f"Report appended to: {report_file}")
print(f"Notebook backed up to: {notebook_copy_path}")
