import json
import os

def restore_gradcam(notebook_path):
    print(f"Processing {notebook_path}...")
    if not os.path.exists(notebook_path):
        print(f"Error: {notebook_path} does not exist.")
        return False
        
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    target_cell_idx = -1
    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source_str = "".join(cell['source'])
            if "def get_target_layer(model):" in source_str:
                target_cell_idx = idx
                break
                
    if target_cell_idx == -1:
        print("Could not find the target layer helper cell.")
        return False
        
    print(f"Found target layer cell at index {target_cell_idx}.")
    
    # Define the new cell containing class GradCAM and show_gradcam
    gradcam_source = [
        "class GradCAM:\n",
        "    def __init__(self, model, target_layer):\n",
        "        self.model = model\n",
        "        self.target_layer = target_layer\n",
        "        self.gradients = None\n",
        "        self.features = None\n",
        "        \n",
        "        # Register hooks\n",
        "        self.hook_forward = self.target_layer.register_forward_hook(self.save_features)\n",
        "        if hasattr(self.target_layer, \"register_full_backward_hook\"):\n",
        "            self.hook_backward = self.target_layer.register_full_backward_hook(self.save_gradients)\n",
        "        else:\n",
        "            self.hook_backward = self.target_layer.register_backward_hook(self.save_gradients)\n",
        "        \n",
        "    def save_features(self, module, input, output):\n",
        "        self.features = output.clone()\n",
        "        \n",
        "    def save_gradients(self, module, grad_input, grad_output):\n",
        "        self.gradients = grad_output[0].clone()\n",
        "        \n",
        "    def __call__(self, x, class_idx=None):\n",
        "        self.model.eval()\n",
        "        \n",
        "        # Ensure gradients are enabled for the backward pass\n",
        "        with torch.enable_grad():\n",
        "            output = self.model(x)\n",
        "            \n",
        "            loss_type = TrainingConfig.loss_stage3 if TrainingConfig.training_pipeline == \"3-stage\" else (TrainingConfig.loss_stage2 if TrainingConfig.training_pipeline == \"2-stage\" else TrainingConfig.loss_standard)\n",
        "            predict_fn = get_prediction_helper(loss_type)\n",
        "            \n",
        "            if class_idx is None:\n",
        "                class_idx = predict_fn(output).item()\n",
        "                \n",
        "            self.model.zero_grad()\n",
        "            \n",
        "            if loss_type == \"ce\":\n",
        "                loss = output[0, class_idx]\n",
        "            else:\n",
        "                logits = output[0]  # Shape: (4,)\n",
        "                probs = torch.sigmoid(logits)\n",
        "                \n",
        "                # SOTA Ordinal Log-Probability Backpropagation\n",
        "                if loss_type in [\"corn\", \"focal_corn\"]:\n",
        "                    if class_idx == 0:\n",
        "                        loss = torch.log(1.0 - probs[0] + 1e-10)\n",
        "                    elif class_idx == 4:\n",
        "                        loss = torch.sum(torch.log(probs + 1e-10))\n",
        "                    else:\n",
        "                        log_probs = torch.log(probs + 1e-10)\n",
        "                        loss = torch.sum(log_probs[:class_idx]) + torch.log(1.0 - probs[class_idx] + 1e-10)\n",
        "                elif loss_type in [\"threshold\", \"coral\"]:\n",
        "                    if class_idx == 0:\n",
        "                        loss = torch.log(1.0 - probs[0] + 1e-10)\n",
        "                    elif class_idx == 4:\n",
        "                        loss = torch.log(probs[3] + 1e-10)\n",
        "                    else:\n",
        "                        loss = torch.log(probs[class_idx - 1] - probs[class_idx] + 1e-10)\n",
        "                else:\n",
        "                    loss = output[0, class_idx]\n",
        "                \n",
        "            loss.backward()\n",
        "            \n",
        "        # Pool gradients\n",
        "        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)\n",
        "        # Apply weights to features\n",
        "        cam = torch.sum(weights * self.features, dim=1).squeeze(0)\n",
        "        \n",
        "        # Apply ReLU to retain positive influence features\n",
        "        cam = F.relu(cam)\n",
        "        cam = cam.cpu().detach().numpy()\n",
        "        \n",
        "        if cam.max() > 0:\n",
        "            cam = cam / cam.max()\n",
        "            \n",
        "        # Resize to input tensor width and height (correctly shape-swapped for OpenCV)\n",
        "        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))\n",
        "        return cam, class_idx\n",
        "        \n",
        "    def remove_hooks(self):\n",
        "        self.hook_forward.remove()\n",
        "        self.hook_backward.remove()\n",
        "\n",
        "def show_gradcam(image_path, model, target_layer, transform):\n",
        "    if not os.path.exists(image_path):\n",
        "        print(f\"Error: Image not found at {image_path}\")\n",
        "        return\n",
        "        \n",
        "    # Read and preprocess image\n",
        "    img = cv2.imread(image_path)\n",
        "    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
        "    \n",
        "    # Process original image dimensions for display\n",
        "    pad = SquarePadOpenCV()\n",
        "    clahe = OpenCVCLAHE()\n",
        "    img_processed = clahe(pad(img_rgb))\n",
        "    img_resized = cv2.resize(img_processed, (IMG_SIZE, IMG_SIZE))\n",
        "    \n",
        "    # Tensor transform\n",
        "    tensor = transform(img_rgb).unsqueeze(0).to(device)\n",
        "    \n",
        "    # Run Grad-CAM\n",
        "    gradcam = GradCAM(model, target_layer)\n",
        "    cam, class_idx = gradcam(tensor)\n",
        "    gradcam.remove_hooks()\n",
        "    \n",
        "    # Create colormap overlay\n",
        "    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)\n",
        "    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)\n",
        "    \n",
        "    alpha = 0.4\n",
        "    overlay = cv2.addWeighted(img_resized, 1 - alpha, heatmap, alpha, 0)\n",
        "    \n",
        "    # Plot side-by-side\n",
        "    plt.figure(figsize=(12, 6))\n",
        "    plt.subplot(1, 2, 1)\n",
        "    plt.title(\"Original Knee X-ray (Processed)\")\n",
        "    plt.imshow(img_resized)\n",
        "    plt.axis('off')\n",
        "    \n",
        "    plt.subplot(1, 2, 2)\n",
        "    plt.title(f\"Grad-CAM Heatmap (Predicted Grade: {class_idx})\")\n",
        "    plt.imshow(overlay)\n",
        "    plt.axis('off')\n",
        "    \n",
        "    plt.show()\n"
    ]
    
    gradcam_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": gradcam_source
    }
    
    # Define the cell to load the model and call show_gradcam
    call_source = [
        "# Instantiate model instance and load best checkpoint weights\n",
        "loss_type_current = TrainingConfig.loss_stage3 if TrainingConfig.training_pipeline == \"3-stage\" else (TrainingConfig.loss_stage2 if TrainingConfig.training_pipeline == \"2-stage\" else TrainingConfig.loss_standard)\n",
        "best_model = DenseNet201Model(num_classes=5, pretrained=False, loss_type=loss_type_current)\n",
        "\n",
        "best_weight_path = os.path.join(TrainingConfig.checkpoint_dir, \"best_model_stage2.pth\")\n",
        "if not os.path.exists(best_weight_path):\n",
        "    best_weight_path = os.path.join(TrainingConfig.checkpoint_dir, \"best_model.pth\")\n",
        "\n",
        "print(f\"Loading best model weights from: {best_weight_path}\")\n",
        "try:\n",
        "    checkpoint = torch.load(best_weight_path, map_location=device, weights_only=False)\n",
        "except TypeError:\n",
        "    checkpoint = torch.load(best_weight_path, map_location=device)\n",
        "\n",
        "if 'model_state_dict' in checkpoint:\n",
        "    best_model.load_state_dict(checkpoint['model_state_dict'])\n",
        "elif 'model' in checkpoint:\n",
        "    best_model.load_state_dict(checkpoint['model'])\n",
        "else:\n",
        "    best_model.load_state_dict(checkpoint)\n",
        "\n",
        "best_model = best_model.to(device)\n",
        "target_layer = get_target_layer(best_model)\n",
        "\n",
        "# Select dynamically from test/val dataset to visualize\n",
        "dataset_to_use = val_dataset if 'val_dataset' in globals() else (test_dataset if 'test_dataset' in globals() else None)\n",
        "if dataset_to_use is not None:\n",
        "    # Get one example image per class grade (0-4)\n",
        "    class_indices = {i: None for i in range(5)}\n",
        "    for idx, label in enumerate(dataset_to_use.labels):\n",
        "        if class_indices[label] is None:\n",
        "            class_indices[label] = dataset_to_use.image_paths[idx]\n",
        "    for grade, path in sorted(class_indices.items()):\n",
        "        if path is not None and os.path.exists(path):\n",
        "            print(f\"\\n--- Visualizing Grade {grade} ---\")\n",
        "            show_gradcam(path, best_model, target_layer, val_transform)\n",
        "else:\n",
        "    print(\"Dataset loader not active in environment. Run previous cells to load datasets.\")\n"
    ]
    
    call_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": call_source
    }
    
    # We will insert the new cells after target_cell_idx
    # But let's first check if GradCAM class is already present in cells immediately following it, to prevent duplicate inserts
    next_cell_source = "".join(nb['cells'][target_cell_idx + 1]['source']) if target_cell_idx + 1 < len(nb['cells']) else ""
    if "class GradCAM:" in next_cell_source:
        print("GradCAM cell already present. Updating its content.")
        nb['cells'][target_cell_idx + 1]['source'] = gradcam_source
        if target_cell_idx + 2 < len(nb['cells']) and "Instantiate model instance" in "".join(nb['cells'][target_cell_idx + 2]['source']):
            nb['cells'][target_cell_idx + 2]['source'] = call_source
    else:
        # Insert them
        nb['cells'].insert(target_cell_idx + 1, call_cell)
        nb['cells'].insert(target_cell_idx + 1, gradcam_cell)
        print("Inserted GradCAM definition and invocation cells.")
        
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Successfully restored Grad-CAM cells in {notebook_path}.\n")
    return True

if __name__ == "__main__":
    restore_gradcam("notebooks/dense_net_121.ipynb")
    restore_gradcam("notebooks/dense_net_201.ipynb")
