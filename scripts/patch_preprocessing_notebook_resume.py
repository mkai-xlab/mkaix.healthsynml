"""Patch the preprocessing ablation notebook with interruption-safe resume logic."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/experiments/dense_net_121_preprocessing_quality_ablation.ipynb"


NEW_TRAIN_ARM = r'''def train_arm(spec):
    seed_everything(42)
    arm_dir = RUN_DIR / spec["name"]
    arm_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = KneeDataset(train_paths, train_labels, build_train_transform(spec))
    validation_dataset = KneeDataset(validation_paths, validation_labels, build_evaluation_transform(spec))
    class_counts = Counter(train_dataset.labels)
    sample_weights = [1.0 / class_counts[label] for label in train_dataset.labels]
    sampler = WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True,
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=48, sampler=sampler, num_workers=4,
        pin_memory=device.type == "cuda", persistent_workers=True,
        generator=torch.Generator().manual_seed(42),
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=48, shuffle=False, num_workers=4,
        pin_memory=device.type == "cuda", persistent_workers=True,
    )

    model = DenseNet121NativeCAM(pretrained=True).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    stage2_path = arm_dir / "stage2_best_model.pth"
    best_path = arm_dir / "best_model.pth"
    last_path = arm_dir / "last_model.pth"
    resume = load_checkpoint(last_path) if last_path.exists() else None
    resume_epoch = int(resume.get("epoch", 0)) if resume else 0
    history = list(resume.get("history", [])) if resume else []
    best_stage2_score = max(
        (item["validation_metrics"]["selection_score"] for item in history if item["stage"] == "coarse"),
        default=-float("inf"),
    )
    best_final_score = max(
        (item["validation_metrics"]["selection_score"] for item in history if item["stage"] == "finetune"),
        default=-float("inf"),
    )
    if resume_epoch:
        print(f"{spec['name']} | resuming after epoch {resume_epoch} from {last_path}")

    stage_specs = (("warmup", 5, 0), ("coarse", 15, 5), ("finetune", 10, 20))
    optimizer = None
    scheduler = None
    for stage_name, stage_epochs, stage_offset in stage_specs:
        stage_end = stage_offset + stage_epochs
        if resume_epoch >= stage_end:
            continue
        if stage_name == "warmup":
            model.freeze_backbone()
            optimizer = optim.AdamW(model.class_conv.parameters(), lr=3e-4, weight_decay=1e-4)
            scheduler = None
        elif stage_name == "coarse":
            model.unfreeze_final_stages()
            optimizer = optim.AdamW(
                [
                    {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": 3e-5},
                    {"params": model.class_conv.parameters(), "lr": 3e-4},
                ], weight_decay=1e-4,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-7)
        else:
            model.unfreeze_all()
            optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-3)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7)

        if resume and resume_epoch >= stage_offset:
            model.load_state_dict(resume["model_state_dict"])
            same_stage_checkpoint = resume.get("stage") == stage_name
            if same_stage_checkpoint and resume.get("optimizer_state_dict"):
                optimizer.load_state_dict(resume["optimizer_state_dict"])
            if same_stage_checkpoint and scheduler is not None and resume.get("scheduler_state_dict"):
                scheduler.load_state_dict(resume["scheduler_state_dict"])
            if same_stage_checkpoint and resume.get("scaler_state_dict"):
                scaler.load_state_dict(resume["scaler_state_dict"])
        elif stage_name == "finetune":
            if not stage2_path.exists():
                raise FileNotFoundError(f"No Stage 2 checkpoint for {spec['name']}")
            model.load_state_dict(load_checkpoint(stage2_path)["model_state_dict"])

        first_stage_epoch = max(0, resume_epoch - stage_offset) if resume else 0
        for stage_epoch in range(first_stage_epoch, stage_epochs):
            global_epoch = stage_offset + stage_epoch + 1
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scaler, spec,
                f"{spec['name']} | {stage_name} {stage_epoch + 1}/{stage_epochs}",
            )
            validation_metrics, _, _, _, _ = evaluate(
                model, validation_loader, spec, f"{spec['name']} | VALIDATE"
            )
            history.append({"epoch": global_epoch, "stage": stage_name, "train_loss": train_loss, **validation_metrics})
            print(
                f"{spec['name']} | epoch {global_epoch:02d} | {stage_name} | "
                f"QWK={validation_metrics['qwk']:.4f} | F1={validation_metrics['macro_f1']:.4f} | "
                f"G1R={validation_metrics['grade1_recall']:.4f} | AP={validation_metrics['macro_ap']:.4f} | "
                f"selection={validation_metrics['selection_score']:.4f}"
            )
            if scheduler is not None:
                scheduler.step()
            payload = {
                "model_state_dict": model.state_dict(), "epoch": global_epoch,
                "stage": stage_name, "architecture": "final_linear_native_cam",
                "model_name": "densenet121", "loss_type": "ce",
                "validation_metrics": validation_metrics, "history": history,
                "run_timestamp": RUN_TIMESTAMP, "arm": spec,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state_dict": scaler.state_dict(),
                "fixed_training_config": {
                    "orientation": "natural_no_canonicalization", "input_resize": 384,
                    "input_crop": None, "batch_size": 48,
                    "sampler": "full_inverse_frequency", "stage_epochs": [5, 15, 10],
                    "learning_rates": [3e-4, 3e-5, 1e-5], "native_cam": True,
                },
            }
            if stage_name == "coarse" and validation_metrics["selection_score"] > best_stage2_score:
                best_stage2_score = validation_metrics["selection_score"]
                torch.save(payload, stage2_path)
            if stage_name == "finetune" and validation_metrics["selection_score"] > best_final_score:
                best_final_score = validation_metrics["selection_score"]
                torch.save(payload, best_path)
            torch.save(payload, last_path)
        resume = load_checkpoint(last_path) if last_path.exists() else resume
        resume_epoch = int(resume.get("epoch", global_epoch)) if resume else global_epoch

    if not best_path.exists():
        raise RuntimeError(f"{spec['name']} produced no final checkpoint")
    best_checkpoint = load_checkpoint(best_path)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    cam_summary = audit_native_cam(model, validation_dataset, arm_dir)
    result = {
        "arm": spec, "checkpoint": str(best_path),
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
'''


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    target = None
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if "def train_arm(spec):" in source:
            target = cell
            break
    if target is None:
        raise RuntimeError("Could not find train_arm cell")
    source = "".join(target["source"])
    if "optimizer_state_dict" not in source:
        source = re.sub(
            r"def train_arm\(spec\):.*?(?=\narm_results = \{)",
            NEW_TRAIN_ARM.rstrip() + "\n\n",
            source,
            flags=re.S,
        )
        if source == "".join(target["source"]):
            raise RuntimeError("train_arm replacement did not change notebook")
    source = source.replace(
        'item["validation_metrics"]["selection_score"] for item in history',
        'item["selection_score"] for item in history',
    )
    source = source.replace(
        '            if resume.get("optimizer_state_dict"):\n'
        '                optimizer.load_state_dict(resume["optimizer_state_dict"])\n'
        '            if scheduler is not None and resume.get("scheduler_state_dict"):\n'
        '                scheduler.load_state_dict(resume["scheduler_state_dict"])\n'
        '            if resume.get("scaler_state_dict"):\n'
        '                scaler.load_state_dict(resume["scaler_state_dict"])',
        '            same_stage_checkpoint = resume.get("stage") == stage_name\n'
        '            if same_stage_checkpoint and resume.get("optimizer_state_dict"):\n'
        '                optimizer.load_state_dict(resume["optimizer_state_dict"])\n'
        '            if same_stage_checkpoint and scheduler is not None and resume.get("scheduler_state_dict"):\n'
        '                scheduler.load_state_dict(resume["scheduler_state_dict"])\n'
        '            if same_stage_checkpoint and resume.get("scaler_state_dict"):\n'
        '                scaler.load_state_dict(resume["scaler_state_dict"])',
    )
    source = source.replace(
        "for arm_spec in DETERMINISTIC_SPECS:\n    print(f\"\\n===== TRAINING {arm_spec['name']} =====\")\n    arm_results[arm_spec[\"name\"]] = train_arm(arm_spec)",
        "for arm_spec in DETERMINISTIC_SPECS:\n    manifest_path = RUN_DIR / arm_spec[\"name\"] / \"arm_manifest.json\"\n    if manifest_path.exists():\n        print(f\"\\n===== SKIPPING COMPLETED {arm_spec['name']} =====\")\n        with open(manifest_path) as handle:\n            arm_results[arm_spec[\"name\"]] = json.load(handle)\n    else:\n        print(f\"\\n===== TRAINING {arm_spec['name']} =====\")\n        arm_results[arm_spec[\"name\"]] = train_arm(arm_spec)",
    )
    source = source.replace(
        'print(f"\\n===== ROBUST FOLLOW-UP FROM {deterministic_winner} =====")\narm_results[robust_spec["name"]] = train_arm(robust_spec)',
        'robust_manifest_path = RUN_DIR / robust_spec["name"] / "arm_manifest.json"\n'
        'if robust_manifest_path.exists():\n'
        '    print(f"\\n===== SKIPPING COMPLETED {robust_spec[\'name\']} =====")\n'
        '    with open(robust_manifest_path) as handle:\n'
        '        arm_results[robust_spec["name"]] = json.load(handle)\n'
        'else:\n'
        '    print(f"\\n===== ROBUST FOLLOW-UP FROM {deterministic_winner} =====")\n'
        '    arm_results[robust_spec["name"]] = train_arm(robust_spec)',
    )
    target["source"] = source.splitlines(keepends=True)
    for cell in notebook["cells"]:
        cell_source = "".join(cell.get("source", []))
        old_run_block = '''RUN_TIMESTAMP = datetime.now(timezone.utc).strftime(
    "%Y-%m-%d_%H-%M-%S_%f_UTC"
)
RUN_DIR = Path("/content/drive/MyDrive/Models/densenet121_checkpoints") / f"{RUN_TIMESTAMP}_preprocessing_quality_ablation"
RUN_DIR.mkdir(parents=True, exist_ok=False)
print(f"Run directory: {RUN_DIR}")'''
        new_run_block = '''RUN_PARENT = Path("/content/drive/MyDrive/Models/densenet121_checkpoints")
requested_resume_dir = os.environ.get("PREPROCESSING_RESUME_DIR")
incomplete_runs = sorted(
    path for path in RUN_PARENT.glob("*_preprocessing_quality_ablation")
    if path.is_dir() and not (path / "run_manifest.json").exists()
)
if requested_resume_dir:
    RUN_DIR = Path(requested_resume_dir)
elif incomplete_runs:
    RUN_DIR = incomplete_runs[-1]
else:
    RUN_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%f_UTC")
    RUN_DIR = RUN_PARENT / f"{RUN_TIMESTAMP}_preprocessing_quality_ablation"
RUN_TIMESTAMP = RUN_DIR.name.split("_preprocessing_quality_ablation")[0]
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"Run directory: {RUN_DIR}")'''
        if old_run_block in cell_source:
            cell["source"] = cell_source.replace(old_run_block, new_run_block).splitlines(keepends=True)
            break
        if (
            "requested_resume_dir = os.environ.get(\"PREPROCESSING_RESUME_DIR\")" in cell_source
            and "PINNED_RESUME_RUN_DIR" not in cell_source
        ):
            pinned_block = '''PINNED_RESUME_RUN_DIR = RUN_PARENT / "2026-07-25_23-48-22_997435_UTC_preprocessing_quality_ablation"
requested_resume_dir = os.environ.get("PREPROCESSING_RESUME_DIR")
if not requested_resume_dir and PINNED_RESUME_RUN_DIR.exists():
    requested_resume_dir = str(PINNED_RESUME_RUN_DIR)'''
            cell["source"] = cell_source.replace(
                'requested_resume_dir = os.environ.get("PREPROCESSING_RESUME_DIR")',
                pinned_block,
            ).splitlines(keepends=True)
            break
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("patched", NOTEBOOK)


if __name__ == "__main__":
    main()
