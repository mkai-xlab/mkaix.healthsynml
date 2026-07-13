import json

notebook_path = "notebooks/dense_net_201.ipynb"

# Load notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Track modifications
modified_scheduler = 0
modified_grad_clip = 0

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_lines = cell['source']
        source_str = "".join(source_lines)
        
        # 1. Modify scheduler_stage2 to "cosine"
        if "scheduler_stage2 = \"none\"" in source_str:
            new_lines = []
            for line in source_lines:
                if "scheduler_stage2 = \"none\"" in line:
                    line = line.replace("scheduler_stage2 = \"none\"", "scheduler_stage2 = \"cosine\"")
                    modified_scheduler += 1
                new_lines.append(line)
            cell['source'] = new_lines
            
        # 2. Modify train_epoch to add gradient clipping
        if "loss.backward()" in source_str and "optimizer.step()" in source_str:
            # Check if gradient clipping is already added
            if "clip_grad_norm_" not in source_str:
                new_lines = []
                for line in source_lines:
                    new_lines.append(line)
                    if "loss.backward()" in line:
                        # Find the indentation
                        indent = line[:len(line) - len(line.lstrip())]
                        new_lines.append(f"{indent}torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)\n")
                        modified_grad_clip += 1
                cell['source'] = new_lines

# Write back
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Applied fixes. Schedulers modified: {modified_scheduler}, Gradient clips added: {modified_grad_clip}")
