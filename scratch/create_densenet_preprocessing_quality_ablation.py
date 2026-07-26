"""Build the DenseNet-121 preprocessing and image-quality ablation notebook."""

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks/experiments/dense_net_121_final_noncanonical_loss_ablation.ipynb"
OUTPUT = ROOT / "notebooks/experiments/dense_net_121_preprocessing_quality_ablation.ipynb"


def lines(text):
    return dedent(text).lstrip("\n").splitlines(keepends=True)


def new_cell(cell_type, source):
    cell = {"cell_type": cell_type, "metadata": {}, "source": lines(source)}
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


source_notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
imports = "".join(source_notebook["cells"][2]["source"])
imports = imports.replace(
    "import torchvision.transforms as transforms\n",
    "import torchvision.transforms as transforms\nimport torchvision.transforms.functional as TF\n",
)
imports = imports.replace(
    'f"{RUN_TIMESTAMP}_final_noncanonical_loss_ablation"',
    'f"{RUN_TIMESTAMP}_preprocessing_quality_ablation"',
)

model_code = "".join(source_notebook["cells"][7]["source"])
shared_code = "".join(source_notebook["cells"][9]["source"])

start = shared_code.index("def cam_geometry(cam):")
end = shared_code.index("def audit_native_cam", start)
geometry_code = dedent(
    '''
    def normalized_edge_sharpness(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        image = (tensor.cpu() * std + mean).clamp(0, 1)
        gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        low = torch.quantile(gray, 0.01)
        high = torch.quantile(gray, 0.99)
        gray = ((gray - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
        gradient_x = gray[:, 1:] - gray[:, :-1]
        gradient_y = gray[1:, :] - gray[:-1, :]
        return float(
            10000.0
            * (gradient_x.square().mean() + gradient_y.square().mean())
        )


    def cam_geometry(cam):
        height, width = cam.shape
        joint = np.zeros((height, width), dtype=bool)
        joint[
            int(0.28 * height):int(0.72 * height),
            int(0.06 * width):int(0.94 * width),
        ] = True
        border = np.ones((height, width), dtype=bool)
        border[
            int(0.08 * height):int(0.92 * height),
            int(0.08 * width):int(0.92 * width),
        ] = False
        lower_tibia = np.zeros((height, width), dtype=bool)
        lower_tibia[
            int(0.72 * height):int(0.96 * height),
            int(0.06 * width):int(0.94 * width),
        ] = True
        total = float(cam.sum()) + 1e-8
        peak = np.unravel_index(np.argmax(cam), cam.shape)
        joint_energy = float(cam[joint].sum()) / total
        border_energy = float(cam[border].sum()) / total
        lower_tibia_energy = float(cam[lower_tibia].sum()) / total
        peak_inside_joint = bool(joint[peak] and total > 1e-7)
        anatomy_score = (
            joint_energy
            * (1.0 - border_energy)
            * (1.0 - lower_tibia_energy)
        )
        gate_pass = (
            joint_energy >= 0.55
            and border_energy <= 0.25
            and lower_tibia_energy <= 0.25
            and peak_inside_joint
        )
        return {
            "joint_energy": joint_energy,
            "border_energy": border_energy,
            "lower_tibia_energy": lower_tibia_energy,
            "peak_inside_joint": int(peak_inside_joint),
            "anatomy_score": anatomy_score,
            "gate_pass": int(gate_pass),
        }


    '''
)
shared_code = shared_code[:start] + geometry_code + shared_code[end:]
shared_code = shared_code.replace(
    "        tensor, true_grade, path = dataset[index]\n",
    "        tensor, true_grade, path = dataset[index]\n"
    "        input_sharpness = normalized_edge_sharpness(tensor)\n",
)
shared_code = shared_code.replace(
    '            "predicted_grade": predicted_grade,\n',
    '            "predicted_grade": predicted_grade,\n'
    '            "input_sharpness": input_sharpness,\n',
)
shared_code = shared_code.replace(
    '        "peak_inside_joint_rate": float(np.mean([\n'
    '            row["predicted_peak_inside_joint"] for row in rows\n'
    '        ])),\n',
    '        "peak_inside_joint_rate": float(np.mean([\n'
    '            row["predicted_peak_inside_joint"] for row in rows\n'
    '        ])),\n'
    '        "gate_pass_rate": float(np.mean([\n'
    '            row["predicted_gate_pass"] for row in rows\n'
    '        ])),\n'
    '        "anatomy_score": float(np.mean([\n'
    '            row["predicted_anatomy_score"] for row in rows\n'
    '        ])),\n'
    '        "input_sharpness": float(np.mean([\n'
    '            row["input_sharpness"] for row in rows\n'
    '        ])),\n'
    '        "sharpness_anatomy_correlation": float(np.corrcoef(\n'
    '            [row["input_sharpness"] for row in rows],\n'
    '            [row["predicted_anatomy_score"] for row in rows],\n'
    '        )[0, 1]),\n',
)

cells = [
    new_cell(
        "markdown",
        """
        # DenseNet-121 Preprocessing and Image-Quality Ablation

        This experiment tests whether fixed CLAHE, its position relative to black
        square padding, or acquisition-quality variation contributes to off-joint
        native CAMs. Architecture, natural orientation, CE loss, split, sampler,
        training schedule, seed, and CAM masks remain fixed.

        Deterministic controls:

        - `raw_then_pad`: no contrast enhancement.
        - `current_pad_then_clahe2`: exact deployed order and CLAHE strength.
        - `clahe2_then_pad`: same CLAHE, applied before artificial black padding.
        - `clahe1_25_then_pad`: milder pre-padding CLAHE.
        - `percentile_1_99_then_pad`: robust global intensity windowing without
          local histogram amplification.

        After those controls finish, the best validation-eligible deterministic arm
        receives one follow-up with conservative random gamma, blur/sharpness, and
        Gaussian noise during training only. The repeatedly inspected test split is
        not opened in this notebook.
        """,
    ),
    new_cell(
        "markdown",
        """
        ## Execution

        Run every cell from top to bottom in one A100 Colab runtime. The notebook
        performs six complete 5/15/10-stage trainings and saves every checkpoint in
        a unique timestamped directory. Expect this to take substantially longer
        than one production training. Do not skip the preprocessing preview or CAM
        comparison cells.
        """,
    ),
    new_cell("code", imports),
    new_cell(
        "markdown",
        """
        ## Fixed Data and Controlled Preprocessing

        Enhancement is performed on the detected knee ROI. Pre-padding arms avoid
        including synthetic black padding in histogram estimation. Validation is
        deterministic and always uses the same preprocessing assigned to its arm.
        """,
    ),
    new_cell(
        "code",
        r'''
        class SquarePad:
            def __call__(self, image):
                height, width = image.shape[:2]
                side = max(height, width)
                top = (side - height) // 2
                bottom = side - height - top
                left = (side - width) // 2
                right = side - width - left
                return cv2.copyMakeBorder(
                    image, top, bottom, left, right,
                    cv2.BORDER_CONSTANT, value=[0, 0, 0],
                )


        class IdentityEnhancement:
            def __call__(self, image):
                return image


        class CLAHE:
            def __init__(self, clip_limit):
                self.clip_limit = float(clip_limit)

            def __call__(self, image):
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                lightness, channel_a, channel_b = cv2.split(lab)
                lightness = cv2.createCLAHE(
                    clipLimit=self.clip_limit,
                    tileGridSize=(8, 8),
                ).apply(lightness)
                return cv2.cvtColor(
                    cv2.merge((lightness, channel_a, channel_b)),
                    cv2.COLOR_LAB2RGB,
                )


        class PercentileWindow:
            def __init__(self, low=1.0, high=99.0):
                self.low = float(low)
                self.high = float(high)

            def __call__(self, image):
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                low, high = np.percentile(gray, [self.low, self.high])
                if high <= low + 1.0:
                    return image
                scaled = (image.astype(np.float32) - low) * (255.0 / (high - low))
                return np.clip(scaled, 0, 255).astype(np.uint8)


        class RandomGammaCorrection:
            def __init__(self, gamma_range=(0.85, 1.15)):
                self.gamma_range = tuple(gamma_range)

            def __call__(self, image):
                gamma = random.uniform(*self.gamma_range)
                return TF.adjust_gamma(image, gamma=gamma, gain=1.0)


        class RandomBlurOrSharpness:
            def __call__(self, image):
                draw = random.random()
                if draw < 0.15:
                    sigma = random.uniform(0.1, 0.8)
                    return TF.gaussian_blur(image, kernel_size=[3, 3], sigma=[sigma, sigma])
                if draw < 0.30:
                    factor = random.uniform(0.70, 1.40)
                    return TF.adjust_sharpness(image, factor)
                return image


        class RandomGaussianNoise:
            def __init__(self, probability=0.20, std_range=(0.005, 0.020)):
                self.probability = float(probability)
                self.std_range = tuple(std_range)

            def __call__(self, tensor):
                if random.random() >= self.probability:
                    return tensor
                std = random.uniform(*self.std_range)
                return (tensor + torch.randn_like(tensor) * std).clamp(0, 1)


        DETERMINISTIC_SPECS = [
            {
                "name": "raw_then_pad",
                "enhancement": "none",
                "clip_limit": None,
                "enhance_before_padding": True,
                "acquisition_augmentation": False,
                "loss_type": "ce",
                "ordinal_weight": 0.0,
            },
            {
                "name": "current_pad_then_clahe2",
                "enhancement": "clahe",
                "clip_limit": 2.0,
                "enhance_before_padding": False,
                "acquisition_augmentation": False,
                "loss_type": "ce",
                "ordinal_weight": 0.0,
            },
            {
                "name": "clahe2_then_pad",
                "enhancement": "clahe",
                "clip_limit": 2.0,
                "enhance_before_padding": True,
                "acquisition_augmentation": False,
                "loss_type": "ce",
                "ordinal_weight": 0.0,
            },
            {
                "name": "clahe1_25_then_pad",
                "enhancement": "clahe",
                "clip_limit": 1.25,
                "enhance_before_padding": True,
                "acquisition_augmentation": False,
                "loss_type": "ce",
                "ordinal_weight": 0.0,
            },
            {
                "name": "percentile_1_99_then_pad",
                "enhancement": "percentile_1_99",
                "clip_limit": None,
                "enhance_before_padding": True,
                "acquisition_augmentation": False,
                "loss_type": "ce",
                "ordinal_weight": 0.0,
            },
        ]


        def build_enhancement(spec):
            if spec["enhancement"] == "none":
                return IdentityEnhancement()
            if spec["enhancement"] == "clahe":
                return CLAHE(spec["clip_limit"])
            if spec["enhancement"] == "percentile_1_99":
                return PercentileWindow(1.0, 99.0)
            raise ValueError(f"Unknown enhancement: {spec['enhancement']}")


        def deterministic_operations(spec):
            enhancement = build_enhancement(spec)
            if spec["enhance_before_padding"]:
                return [enhancement, SquarePad()]
            return [SquarePad(), enhancement]


        def build_train_transform(spec):
            operations = deterministic_operations(spec)
            operations.extend([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.50),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.08, contrast=0.08),
            ])
            if spec["acquisition_augmentation"]:
                operations.extend([
                    transforms.RandomApply([RandomGammaCorrection()], p=0.30),
                    RandomBlurOrSharpness(),
                ])
            operations.extend([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
            ])
            if spec["acquisition_augmentation"]:
                operations.append(RandomGaussianNoise())
            operations.extend([
                transforms.RandomErasing(
                    p=0.10,
                    scale=(0.02, 0.05),
                    ratio=(0.5, 2.0),
                    value=0,
                ),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            return transforms.Compose(operations)


        def build_evaluation_transform(spec):
            return transforms.Compose([
                *deterministic_operations(spec),
                transforms.ToPILImage(),
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])


        def file_digest(path):
            digest = hashlib.sha256()
            with open(path, "rb") as image_file:
                for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()


        def index_split(split, excluded_hashes=None):
            split_dir = DATASET_ROOT / split
            if not split_dir.is_dir():
                raise FileNotFoundError(f"Dataset split is missing: {split_dir}")
            excluded_hashes = excluded_hashes or set()
            retained_hashes, paths, labels = set(), [], []
            for grade in range(5):
                grade_dir = split_dir / str(grade)
                for path in sorted(grade_dir.iterdir()):
                    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                        continue
                    digest = file_digest(path)
                    if digest in excluded_hashes or digest in retained_hashes:
                        continue
                    retained_hashes.add(digest)
                    paths.append(str(path))
                    labels.append(grade)
            print(f"{split}: {len(paths)} unique images")
            return paths, labels, retained_hashes


        class KneeDataset(Dataset):
            def __init__(self, paths, labels, transform):
                self.paths = list(paths)
                self.labels = list(labels)
                self.transform = transform

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, index):
                path = self.paths[index]
                image = cv2.imread(path)
                if image is None:
                    raise IOError(f"Could not read image: {path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return self.transform(image), self.labels[index], path
        ''',
    ),
    new_cell(
        "code",
        r'''
        train_paths, train_labels, train_hashes = index_split("train")
        validation_paths, validation_labels, validation_hashes = index_split(
            "val", train_hashes
        )
        print("Train class counts:", dict(sorted(Counter(train_labels).items())))
        print("Deterministic preprocessing arms:")
        for spec in DETERMINISTIC_SPECS:
            print(json.dumps(spec, indent=2))

        preview_source = cv2.cvtColor(
            cv2.imread(train_paths[len(train_paths) // 2]),
            cv2.COLOR_BGR2RGB,
        )
        figure, axes = plt.subplots(1, len(DETERMINISTIC_SPECS), figsize=(20, 4))
        for axis, spec in zip(axes, DETERMINISTIC_SPECS):
            preview = preview_source.copy()
            for operation in deterministic_operations(spec):
                preview = operation(preview)
            preview = cv2.resize(preview, (384, 384), interpolation=cv2.INTER_AREA)
            axis.imshow(preview)
            axis.set_title(spec["name"], fontsize=9)
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(RUN_DIR / "preprocessing_preview.png", dpi=180)
        plt.show()
        ''',
    ),
    new_cell(
        "markdown",
        """
        ## Fixed Five-Class Native-CAM DenseNet-121

        The architecture is unchanged. A bias-free 1x1 class convolution followed by
        global average pooling produces five KL logits and the corresponding native
        class maps.
        """,
    ),
    new_cell("code", model_code),
    new_cell(
        "markdown",
        """
        ## Shared CE Training and Frozen CAM Audit

        Each arm resets seed 42 and trains for 5/15/10 epochs. Epoch checkpoints are
        selected using `0.55*QWK + 0.30*macro-F1 + 0.15*macro-AP`. After training,
        up to 50 validation cases per grade receive predicted- and true-class CAMs,
        production gate metrics, joint occlusion, flip consistency, and an input
        sharpness measurement.
        """,
    ),
    new_cell("code", shared_code),
    new_cell(
        "code",
        r'''
        def localization_score(cam):
            return float(
                0.40 * cam["gate_pass_rate"]
                + 0.30 * cam["joint_energy"]
                + 0.15 * (1.0 - cam["border_energy"])
                + 0.15 * (1.0 - cam["lower_tibia_energy"])
            )


        def choose_candidate(results):
            best_classification = max(
                result["validation_metrics"]["selection_score"]
                for result in results.values()
            )
            eligible = {
                name: result
                for name, result in results.items()
                if result["validation_metrics"]["selection_score"]
                >= best_classification - 0.01
            }
            return max(
                eligible,
                key=lambda name: (
                    eligible[name]["localization_score"],
                    eligible[name]["validation_metrics"]["selection_score"],
                ),
            )


        def train_arm(spec):
            seed_everything(42)
            arm_dir = RUN_DIR / spec["name"]
            arm_dir.mkdir(parents=True, exist_ok=True)
            train_dataset = KneeDataset(
                train_paths, train_labels, build_train_transform(spec)
            )
            validation_dataset = KneeDataset(
                validation_paths,
                validation_labels,
                build_evaluation_transform(spec),
            )
            class_counts = Counter(train_dataset.labels)
            sample_weights = [
                1.0 / class_counts[label] for label in train_dataset.labels
            ]
            sampler = WeightedRandomSampler(
                sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
                generator=torch.Generator().manual_seed(42),
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=48,
                sampler=sampler,
                num_workers=4,
                pin_memory=device.type == "cuda",
                persistent_workers=True,
                generator=torch.Generator().manual_seed(42),
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=48,
                shuffle=False,
                num_workers=4,
                pin_memory=device.type == "cuda",
                persistent_workers=True,
            )

            model = DenseNet121NativeCAM(pretrained=True).to(device)
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
            stage2_path = arm_dir / "stage2_best_model.pth"
            best_path = arm_dir / "best_model.pth"
            last_path = arm_dir / "last_model.pth"
            history = []
            best_stage2_score = -float("inf")
            best_final_score = -float("inf")
            global_epoch = 0

            for stage_name, stage_epochs in (
                ("warmup", 5), ("coarse", 15), ("finetune", 10)
            ):
                if stage_name == "warmup":
                    model.freeze_backbone()
                    optimizer = optim.AdamW(
                        model.class_conv.parameters(), lr=3e-4, weight_decay=1e-4
                    )
                    scheduler = None
                elif stage_name == "coarse":
                    model.unfreeze_final_stages()
                    optimizer = optim.AdamW(
                        [
                            {
                                "params": [
                                    parameter
                                    for parameter in model.backbone.parameters()
                                    if parameter.requires_grad
                                ],
                                "lr": 3e-5,
                            },
                            {"params": model.class_conv.parameters(), "lr": 3e-4},
                        ],
                        weight_decay=1e-4,
                    )
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=15, eta_min=1e-7
                    )
                else:
                    if not stage2_path.exists():
                        raise FileNotFoundError(
                            f"No Stage 2 checkpoint for {spec['name']}"
                        )
                    model.load_state_dict(
                        load_checkpoint(stage2_path)["model_state_dict"]
                    )
                    model.unfreeze_all()
                    optimizer = optim.AdamW(
                        model.parameters(), lr=1e-5, weight_decay=1e-3
                    )
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=10, eta_min=1e-7
                    )

                for stage_epoch in range(stage_epochs):
                    global_epoch += 1
                    train_loss = train_one_epoch(
                        model,
                        train_loader,
                        optimizer,
                        scaler,
                        spec,
                        f"{spec['name']} | {stage_name} {stage_epoch + 1}/{stage_epochs}",
                    )
                    validation_metrics, _, _, _, _ = evaluate(
                        model,
                        validation_loader,
                        spec,
                        f"{spec['name']} | VALIDATE",
                    )
                    history.append({
                        "epoch": global_epoch,
                        "stage": stage_name,
                        "train_loss": train_loss,
                        **validation_metrics,
                    })
                    print(
                        f"{spec['name']} | epoch {global_epoch:02d} | {stage_name} | "
                        f"QWK={validation_metrics['qwk']:.4f} | "
                        f"F1={validation_metrics['macro_f1']:.4f} | "
                        f"G1R={validation_metrics['grade1_recall']:.4f} | "
                        f"AP={validation_metrics['macro_ap']:.4f} | "
                        f"selection={validation_metrics['selection_score']:.4f}"
                    )
                    if scheduler is not None:
                        scheduler.step()
                    payload = {
                        "model_state_dict": model.state_dict(),
                        "epoch": global_epoch,
                        "stage": stage_name,
                        "architecture": "final_linear_native_cam",
                        "model_name": "densenet121",
                        "loss_type": "ce",
                        "validation_metrics": validation_metrics,
                        "history": history,
                        "run_timestamp": RUN_TIMESTAMP,
                        "arm": spec,
                        "fixed_training_config": {
                            "orientation": "natural_no_canonicalization",
                            "input_resize": 384,
                            "input_crop": None,
                            "batch_size": 48,
                            "sampler": "full_inverse_frequency",
                            "stage_epochs": [5, 15, 10],
                            "learning_rates": [3e-4, 3e-5, 1e-5],
                            "native_cam": True,
                        },
                    }
                    if (
                        stage_name == "coarse"
                        and validation_metrics["selection_score"] > best_stage2_score
                    ):
                        best_stage2_score = validation_metrics["selection_score"]
                        torch.save(payload, stage2_path)
                    if (
                        stage_name == "finetune"
                        and validation_metrics["selection_score"] > best_final_score
                    ):
                        best_final_score = validation_metrics["selection_score"]
                        torch.save(payload, best_path)
                    torch.save(payload, last_path)

            if not best_path.exists():
                raise RuntimeError(f"{spec['name']} produced no final checkpoint")
            with open(arm_dir / "epoch_metrics.csv", "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=history[0].keys())
                writer.writeheader()
                writer.writerows(history)

            best_checkpoint = load_checkpoint(best_path)
            model.load_state_dict(best_checkpoint["model_state_dict"])
            cam_summary = audit_native_cam(model, validation_dataset, arm_dir)
            result = {
                "arm": spec,
                "checkpoint": str(best_path),
                "best_epoch": int(best_checkpoint["epoch"]),
                "validation_metrics": best_checkpoint["validation_metrics"],
                "native_cam_summary": cam_summary,
                "localization_score": localization_score(cam_summary),
            }
            with open(arm_dir / "arm_manifest.json", "w") as handle:
                json.dump(result, handle, indent=2)
            del model, optimizer, scaler, train_loader, validation_loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return result


        arm_results = {}
        for arm_spec in DETERMINISTIC_SPECS:
            print(f"\n===== TRAINING {arm_spec['name']} =====")
            arm_results[arm_spec["name"]] = train_arm(arm_spec)

        deterministic_winner = choose_candidate(arm_results)
        robust_spec = dict(arm_results[deterministic_winner]["arm"])
        robust_spec["name"] = f"{deterministic_winner}_acquisition_robust"
        robust_spec["acquisition_augmentation"] = True
        print(f"\n===== ROBUST FOLLOW-UP FROM {deterministic_winner} =====")
        arm_results[robust_spec["name"]] = train_arm(robust_spec)

        selected_arm_name = choose_candidate(arm_results)
        selected_result = arm_results[selected_arm_name]
        selected_path = Path(selected_result["checkpoint"])
        print(f"Deterministic winner: {deterministic_winner}")
        print(f"Final validation-selected preprocessing: {selected_arm_name}")
        print(f"Selected checkpoint: {selected_path}")
        ''',
    ),
    new_cell(
        "markdown",
        """
        ## Validation Comparison and Production Candidate

        The test directory is deliberately not loaded. An arm is eligible only when
        its classification selection score is within `0.01` of the best arm; among
        eligible arms, the higher localization score wins. This prevents a visually
        attractive CAM from hiding a material KL-grading regression.
        """,
    ),
    new_cell(
        "code",
        r'''
        comparison_rows = []
        for name, result in arm_results.items():
            metrics = result["validation_metrics"]
            cam = result["native_cam_summary"]
            comparison_rows.append({
                "arm": name,
                "enhancement": result["arm"]["enhancement"],
                "clip_limit": result["arm"]["clip_limit"],
                "enhance_before_padding": result["arm"]["enhance_before_padding"],
                "acquisition_augmentation": result["arm"]["acquisition_augmentation"],
                "best_epoch": result["best_epoch"],
                "qwk": metrics["qwk"],
                "macro_f1": metrics["macro_f1"],
                "macro_recall": metrics["macro_recall"],
                "grade1_recall": metrics["grade1_recall"],
                "macro_ap": metrics["macro_ap"],
                "macro_auc": metrics["macro_auc"],
                "classification_selection": metrics["selection_score"],
                "cam_gate_pass_rate": cam["gate_pass_rate"],
                "cam_joint_energy": cam["joint_energy"],
                "cam_border_energy": cam["border_energy"],
                "cam_lower_tibia_energy": cam["lower_tibia_energy"],
                "cam_peak_inside_joint_rate": cam["peak_inside_joint_rate"],
                "cam_anatomy_score": cam["anatomy_score"],
                "input_sharpness": cam["input_sharpness"],
                "sharpness_anatomy_correlation": cam["sharpness_anatomy_correlation"],
                "joint_occlusion_probability_drop": cam["joint_occlusion_probability_drop"],
                "localization_score": result["localization_score"],
                "checkpoint": result["checkpoint"],
            })

        with open(RUN_DIR / "preprocessing_comparison.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=comparison_rows[0].keys())
            writer.writeheader()
            writer.writerows(comparison_rows)
        with open(RUN_DIR / "ablation_results.json", "w") as handle:
            json.dump(arm_results, handle, indent=2)

        names = [row["arm"] for row in comparison_rows]
        x = np.arange(len(names))
        figure, axes = plt.subplots(1, 3, figsize=(22, 6))
        width = 0.25
        for offset, key, label in (
            (-width, "qwk", "QWK"),
            (0, "macro_f1", "Macro F1"),
            (width, "classification_selection", "Class selection"),
        ):
            axes[0].bar(x + offset, [row[key] for row in comparison_rows], width, label=label)
        axes[0].set_title("Validation classification")
        axes[0].legend()

        for offset, key, label in (
            (-width, "cam_gate_pass_rate", "Gate pass"),
            (0, "cam_joint_energy", "Joint energy"),
            (width, "cam_anatomy_score", "Anatomy score"),
        ):
            axes[1].bar(x + offset, [row[key] for row in comparison_rows], width, label=label)
        axes[1].set_title("Validation native-CAM geometry")
        axes[1].legend()

        axes[2].bar(x - width / 2, [row["input_sharpness"] for row in comparison_rows], width, label="Input sharpness")
        axes[2].bar(x + width / 2, [row["localization_score"] for row in comparison_rows], width, label="Localization score")
        axes[2].set_title("Image quality and localization")
        axes[2].legend()
        for axis in axes:
            axis.set_xticks(x, names, rotation=25, ha="right")
        figure.tight_layout()
        figure.savefig(RUN_DIR / "preprocessing_comparison.png", dpi=180)
        plt.show()

        for row in comparison_rows:
            print(json.dumps(row, indent=2))
        ''',
    ),
    new_cell(
        "markdown",
        """
        ## Reproducible Report

        The report records every deterministic method, the dynamically selected
        robustness follow-up, predictive metrics, CAM geometry, sharpness association,
        and the checkpoint eligible for a future locked-holdout evaluation.
        """,
    ),
    new_cell(
        "code",
        r'''
        manifest = {
            "run_timestamp": RUN_TIMESTAMP,
            "run_directory": str(RUN_DIR),
            "experiment": "densenet121_preprocessing_quality_ablation",
            "shared_config": {
                "architecture": "densenet121_final_linear_native_cam",
                "orientation": "natural_no_canonicalization",
                "loss": "cross_entropy",
                "input_resize": 384,
                "input_crop": None,
                "batch_size": 48,
                "sampler": "full_inverse_frequency",
                "seed": 42,
                "stage_epochs": [5, 15, 10],
                "learning_rates": [3e-4, 3e-5, 1e-5],
                "epoch_selection": "0.55*QWK + 0.30*macro_F1 + 0.15*macro_AP",
                "candidate_rule": "within 0.01 classification score, then highest localization score",
                "test_split_opened": False,
            },
            "deterministic_winner": deterministic_winner,
            "selected_preprocessing": selected_arm_name,
            "selected_checkpoint": str(selected_path),
            "arms": arm_results,
        }
        with open(RUN_DIR / "run_manifest.json", "w") as handle:
            json.dump(manifest, handle, indent=2)
        (RUN_DIR / "SELECTED_CHECKPOINT.txt").write_text(str(selected_path) + "\n")

        table_rows = []
        for row in comparison_rows:
            table_rows.append(
                f"| `{row['arm']}` | {row['qwk']:.4f} | {row['macro_f1']:.4f} | "
                f"{row['grade1_recall']:.4f} | {row['macro_ap']:.4f} | "
                f"{row['classification_selection']:.4f} | {row['cam_gate_pass_rate']:.4f} | "
                f"{row['cam_joint_energy']:.4f} | {row['cam_border_energy']:.4f} | "
                f"{row['localization_score']:.4f} |"
            )
        result_table = "\n".join(table_rows)
        report = f"""# DenseNet-121 Preprocessing and Image-Quality Ablation

        | Field | Value |
        | --- | --- |
        | Run timestamp | {RUN_TIMESTAMP} |
        | Architecture | `densenet121_final_linear_native_cam` |
        | Loss | Cross-entropy |
        | Orientation | Natural; no canonicalization |
        | Deterministic winner | `{deterministic_winner}` |
        | Final selected preprocessing | `{selected_arm_name}` |
        | Selected checkpoint | `{selected_path}` |
        | Test split opened | No |

        ## Validation Classification and Native-CAM Comparison

        | Arm | QWK | Macro F1 | Grade 1 recall | AP | Class selection | Gate pass | Joint energy | Border energy | Localization |
        | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
        {result_table}

        ## Decision Rule

        Only arms within 0.01 of the best validation classification score are eligible.
        Among them, choose the highest frozen localization score. Inspect each arm's
        `native_cam_worst_cases.png`; do not promote an arm based only on a scalar gate.
        The selected checkpoint still requires evaluation on a newly locked labeled
        holdout and on production YOLO crops before deployment.

        ## Method Sources

        - Pizer et al., *Adaptive Histogram Equalization and Its Variations* (1987):
          foundational local histogram enhancement and contrast limiting.
        - Tiulpin et al., *Automatic Knee Osteoarthritis Diagnosis from Plain
          Radiographs* (Scientific Reports, 2018): primary deep-learning KOA grading.
        - Hendrycks et al., *AugMix* (ICLR, 2020): stochastic augmentation for
          corruption robustness; this notebook uses conservative radiograph-safe
          acquisition perturbations rather than its unrestricted transform chain.
        - Local `docs/paper/fmed-12-1707588.md`: knee isolation, sharpening,
          normalization, and geometric augmentation for KL grading.

        No preprocessing method is assumed superior before this controlled result.
        """
        (RUN_DIR / "report.md").write_text(report, encoding="utf-8")
        print(f"Report: {RUN_DIR / 'report.md'}")
        print(f"All artifacts: {RUN_DIR}")
        ''',
    ),
    new_cell(
        "markdown",
        """
        ## References

        - Pizer SM et al. Adaptive Histogram Equalization and Its Variations. 1987.
          https://doi.org/10.1016/S0734-189X(87)80186-X
        - Tiulpin A et al. Automatic Knee Osteoarthritis Diagnosis from Plain
          Radiographs: A Deep Learning-Based Approach. 2018.
          https://doi.org/10.1038/s41598-018-31315-7
        - Hendrycks D et al. AugMix: A Simple Data Processing Method to Improve
          Robustness and Uncertainty. ICLR 2020.
          https://openreview.net/forum?id=S1gmrxHFvB
        - Local KOA pipeline paper: `docs/paper/fmed-12-1707588.md`.

        External search note: Semantic Scholar returned rate limiting and PubMed's API
        gateway returned an invalid response during notebook preparation. The cited
        methods are primary sources; the experiment is explicitly empirical because
        none establishes an optimal preprocessing recipe for this exact dataset and
        YOLO crop distribution.
        """,
    ),
    new_cell(
        "code",
        r'''
        try:
            from google.colab import runtime
            print("All artifacts saved. Releasing the Colab runtime.")
            runtime.unassign()
        except ImportError:
            print("Not running in Colab; runtime release skipped.")
        ''',
    ),
]

notebook = {
    "cells": cells,
    "metadata": source_notebook["metadata"],
    "nbformat": 4,
    "nbformat_minor": 5,
}
notebook["metadata"]["experiment"] = {
    "name": "densenet121_preprocessing_quality_ablation",
    "created": "2026-07-26",
    "test_split_opened": False,
}
OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT)
