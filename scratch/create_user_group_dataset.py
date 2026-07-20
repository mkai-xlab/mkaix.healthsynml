#!/usr/bin/env python3
"""Create a patient/group-disjoint copy of the Kaggle knee OA dataset.

The supervised folders are rebuilt from train/val/test using the numeric
identifier in filenames. Regular files are named <group><L|R>.png. The
auto_test folder uses <group>_<view>.png and is copied as an auxiliary split;
it is not included in train/val/test because it overlaps the original test
groups.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path



REGULAR_NAME = re.compile(r"^(?P<group>\d+)(?P<side>[LR])$", re.IGNORECASE)
AUTO_TEST_NAME = re.compile(r"^(?P<group>\d+)_\d+$")
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SUPERVISED_SPLITS = ("train", "val", "test")


def parse_group_id(path: Path, split: str) -> str:
    match = (AUTO_TEST_NAME if split == "auto_test" else REGULAR_NAME).fullmatch(path.stem)
    if match is None:
        expected = "<numeric_id>L/R" if split != "auto_test" else "<numeric_id>_<view>"
        raise ValueError(f"Unexpected filename for {split}: {path.name}; expected {expected}")
    return match.group("group")


def collect_files(root: Path, split: str) -> list[dict]:
    records = []
    split_root = root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_root}")

    for grade_dir in sorted(split_root.iterdir()):
        if not grade_dir.is_dir() or not grade_dir.name.isdigit():
            continue
        grade = int(grade_dir.name)
        if grade not in range(5):
            raise ValueError(f"Expected grade folders 0-4, found {grade_dir}")
        for path in sorted(grade_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                records.append({
                    "source_split": split,
                    "source_path": path,
                    "filename": path.name,
                    "grade": grade,
                    "group_id": parse_group_id(path, split),
                })
    if not records:
        raise ValueError(f"No image files found under {split_root}")
    return records


def validate_groups(records: list[dict], split_name: str) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for record in records:
        groups[record["group_id"]].append(record)

    # A patient can have different grades in the left and right knee. This is
    # expected; the group, rather than the grade, must stay in one split.
    return dict(groups)


def assign_groups(groups: dict[str, list[dict]], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    group_counts = {
        group: Counter(record["grade"] for record in records)
        for group, records in groups.items()
    }
    total_counts = Counter()
    for counts in group_counts.values():
        total_counts.update(counts)

    def select_groups(pool: set[str], fraction: float, random_seed: int) -> set[str]:
        target = {grade: count * fraction for grade, count in total_counts.items()}
        target_images = round(sum(total_counts.values()) * fraction)
        selected = set()
        selected_counts = Counter()
        local_rng = random.Random(random_seed)
        candidates = list(pool)
        local_rng.shuffle(candidates)

        while candidates and sum(selected_counts.values()) < target_images:
            def score(group: str) -> tuple[float, int]:
                after = selected_counts + group_counts[group]
                grade_error = sum(
                    abs(after[grade] - target.get(grade, 0.0))
                    / max(1.0, target.get(grade, 0.0))
                    for grade in total_counts
                )
                size_overshoot = max(0, sum(after.values()) - target_images)
                return grade_error + 0.05 * size_overshoot, sum(group_counts[group].values())

            best = min(candidates, key=score)
            candidates.remove(best)
            selected.add(best)
            selected_counts.update(group_counts[best])
        return selected

    all_group_ids = set(groups)
    test_groups = select_groups(all_group_ids, 0.20, seed)
    remaining_groups = all_group_ids - test_groups
    val_groups = select_groups(remaining_groups, 0.10, seed + 1)
    train_groups = remaining_groups - val_groups

    assignments = {group: "test" for group in test_groups}
    assignments.update({group: "val" for group in val_groups})
    assignments.update({group: "train" for group in train_groups})
    if set(assignments) != all_group_ids:
        raise RuntimeError("Stratified group split did not assign every group exactly once")
    return assignments


def copy_record(record: dict, output_root: Path, destination_split: str) -> Path:
    destination = output_root / destination_split / str(record["grade"]) / record["filename"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(record["source_path"], destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/viet/Capstone/ml/dataset/kaggle_knee_osteoarthritis"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    output = (args.output or source.parent / f"{source.name}-user-group").resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}; use --overwrite explicitly")
        shutil.rmtree(output)

    supervised_records = []
    source_groups = {}
    for split in SUPERVISED_SPLITS:
        records = collect_files(source, split)
        groups = validate_groups(records, split)
        source_groups[split] = groups
        supervised_records.extend(records)
        print(f"{split}: {len(records):,} images, {len(groups):,} groups")

    # Rebuild from all supervised records, not from the original split labels.
    all_groups = validate_groups(supervised_records, "combined supervised data")
    assignments = assign_groups(all_groups, args.seed)

    rows = []
    for group, records in sorted(all_groups.items()):
        destination_split = assignments[group]
        for record in records:
            destination = copy_record(record, output, destination_split)
            rows.append({
                **record,
                "new_split": destination_split,
                "new_path": str(destination),
                "role": "supervised",
            })

    # Preserve auto_test, but never use it as train/validation/test data.
    auto_records = collect_files(source, "auto_test")
    auto_groups = validate_groups(auto_records, "auto_test")
    for record in auto_records:
        destination = copy_record(record, output, "auto_test")
        rows.append({
            **record,
            "new_split": "auto_test",
            "new_path": str(destination),
            "role": "auxiliary_only",
        })

    manifest_path = output / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_split", "new_split", "role", "group_id", "grade",
        "filename", "source_path", "new_path",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["new_split"], item["grade"], item["group_id"], item["filename"])):
            writer.writerow({field: row[field] for field in fields})

    supervised_groups = defaultdict(set)
    for group, split in assignments.items():
        supervised_groups[split].add(group)
    overlaps = {
        f"{left}-{right}": len(supervised_groups[left] & supervised_groups[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    auto_overlap = len(set(auto_groups) & set(source_groups["test"]))

    print(f"\nCreated: {output}")
    print(f"Manifest: {manifest_path}")
    print("New supervised group counts:", {split: len(groups) for split, groups in supervised_groups.items()})
    print("Supervised group overlaps:", overlaps)
    print(f"auto_test groups overlapping original test groups: {auto_overlap:,} (auxiliary only)")
    print("Supervised image counts:")
    for split in ("train", "val", "test"):
        count = sum(1 for row in rows if row["new_split"] == split)
        grades = Counter(row["grade"] for row in rows if row["new_split"] == split)
        print(f"  {split}: {count:,} {dict(sorted(grades.items()))}")


if __name__ == "__main__":
    main()
