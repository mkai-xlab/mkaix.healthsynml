import json
import os

def update_losses_in_notebook(notebook_path):
    print(f"Modifying {notebook_path}...")
    if not os.path.exists(notebook_path):
        print(f"Error: {notebook_path} does not exist.")
        return False
        
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    modified_corn = False
    modified_focal = False
    modified_calls = False
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_str = "".join(cell['source'])
            
            # 1. Update corn_loss definition and default weights
            if "def corn_loss" in source_str and "task_weights=" in source_str:
                new_source = []
                for line in cell['source']:
                    if "def corn_loss" in line and "task_weights=" in line:
                        line = "def corn_loss(logits, y_train, num_classes=5, task_weights=[2.0, 1.8, 1.2, 1.0]):\n"
                    new_source.append(line)
                cell['source'] = new_source
                modified_corn = True
                
            # 2. Update focal_corn_loss definition to support and use task_weights
            if "def focal_corn_loss(logits, y_train" in source_str:
                new_source = []
                for line in cell['source']:
                    if "def focal_corn_loss(logits, y_train" in line:
                        line = "def focal_corn_loss(logits, y_train, num_classes=5, gamma=2.0, alpha=0.25, task_weights=[2.0, 1.8, 1.2, 1.0]):\n"
                    elif "loss += (focal_weight * bce).mean()" in line:
                        new_source.append("        w_k = task_weights[k] if k < len(task_weights) else 1.0\n")
                        line = "        loss += w_k * (focal_weight * bce).mean()\n"
                    new_source.append(line)
                cell['source'] = new_source
                modified_focal = True
                
            # 3. Update lambda calls to pass num_classes and default task_weights/arguments if needed
            if "lambda logits, targets: focal_corn_loss(logits, targets, num_classes)" in source_str:
                new_source = []
                for line in cell['source']:
                    if "lambda logits, targets: focal_corn_loss(logits, targets, num_classes)" in line:
                        line = line.replace(
                            "lambda logits, targets: focal_corn_loss(logits, targets, num_classes)",
                            "lambda logits, targets: focal_corn_loss(logits, targets, num_classes, task_weights=[2.0, 1.8, 1.2, 1.0])"
                        )
                    new_source.append(line)
                cell['source'] = new_source
                modified_calls = True
                
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        
    print(f"Finished {notebook_path}. corn_loss: {modified_corn}, focal_corn_loss: {modified_focal}, lambda calls: {modified_calls}\n")
    return True

if __name__ == "__main__":
    update_losses_in_notebook("notebooks/dense_net_121.ipynb")
    update_losses_in_notebook("notebooks/dense_net_201.ipynb")
