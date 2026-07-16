import json

notebook_path = "notebooks/dense_net_201.ipynb"

# Load notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Modify cells
modified_config = False
modified_loss = 0

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_lines = cell['source']
        source_str = "".join(source_lines)
        
        # 1. Modify TrainingConfig
        if "class TrainingConfig:" in source_str:
            new_lines = []
            for line in source_lines:
                if "use_balanced_sampler = True" in line:
                    line = line.replace("use_balanced_sampler = True", "use_balanced_sampler = False")
                    print("Changed use_balanced_sampler to False in config.")
                if "loss_stage3 = \"focal_corn\"" in line:
                    line = line.replace("loss_stage3 = \"focal_corn\"", "loss_stage3 = \"corn\"")
                    print("Changed loss_stage3 to 'corn' in config.")
                new_lines.append(line)
            cell['source'] = new_lines
            modified_config = True
            
        # 2. Modify corn_loss definitions
        if "def corn_loss(logits, y_train, num_classes=5):" in source_str:
            # We want to replace this function with a weighted version
            new_lines = []
            skip = False
            for line in source_lines:
                if "def corn_loss(logits, y_train, num_classes=5):" in line:
                    new_lines.append("def corn_loss(logits, y_train, num_classes=5, task_weights=[1.0, 1.2, 2.0, 3.5]):\n")
                    continue
                if "loss += F.binary_cross_entropy_with_logits(logits_k, targets_k)" in line:
                    new_lines.append("        # Apply task-specific weight to balance gradients for minority classes\n")
                    new_lines.append("        w_k = task_weights[k] if k < len(task_weights) else 1.0\n")
                    new_lines.append("        loss += w_k * F.binary_cross_entropy_with_logits(logits_k, targets_k)\n")
                    continue
                new_lines.append(line)
            cell['source'] = new_lines
            modified_loss += 1
            print(f"Modified corn_loss definition #{modified_loss}")

# Write back
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook modification complete. Config modified: {modified_config}, Loss modified count: {modified_loss}")
