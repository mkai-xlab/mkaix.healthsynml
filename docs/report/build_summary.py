#!/usr/bin/env python3
"""Regenerate summary.md from report.csv.

report.csv is the full audit record: one row per configuration, each naming exactly one
notebook. Ablation notebooks contribute one row per arm. The summary is the handful of
columns needed to compare runs at a glance, grouped so the deployed models come first.

report.csv itself is the spreadsheet artifact; there is no separate summary.csv, which would
only be a strict subset of it. Run this after editing report.csv so the two never drift:

    python docs/report/build_summary.py
"""
import csv
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).parent
BLANK = {"", "not reported", "not retained", "—", "n/a"}

# The four artifacts the running application actually uses. Kept as an explicit list so
# "DEPLOYED-LINE ANCESTOR" rows do not leak into this section.
DEPLOYED = {
    "DN-RUN-09",   # training run that produced the deployed DenseNet-121 checkpoint
    "DN-RUN-10",   # its locked test-split evaluation
    "ORPHAN-01",   # deployed SE-ResNeXt-50 checkpoint - no notebook, no test metrics
    "YOLO-02",     # detector checkpoint referenced by the ROI pipeline
}

# Ordered report sections: (heading, predicate on the row)
SECTIONS = OrderedDict([
    ("Artifacts in production",
     lambda r: r["row_id"] in DEPLOYED),
    ("Locked test-split evaluations",
     lambda r: r["record_type"] == "evaluation_only"),
    ("Training runs",
     lambda r: r["record_type"] == "training_run"),
    ("Ablation arms",
     lambda r: r["record_type"] == "ablation_arm"),
    ("Detector training",
     lambda r: r["record_type"] == "detector_training"),
    ("Checkpoints with no notebook",
     lambda r: r["record_type"] == "orphan_checkpoint"),
    ("Incomplete, stale, or never executed",
     lambda r: r["record_type"] in {"aborted_run", "incomplete_run", "stale_outputs",
                                    "template_not_executed"}),
    ("Utility notebooks",
     lambda r: r["record_type"] == "utility"),
])


def cell(value):
    value = (value or "").strip()
    return "—" if value in BLANK else value


def numeric(value):
    """Leading float of a cell like '0.7702' or '0.79 (published) / 0.74 (ROI)'."""
    try:
        return float(str(value).split()[0])
    except (ValueError, IndexError):
        return None


def sort_key(row):
    q = numeric(row.get("qwk"))
    return (q is None, -(q or 0.0), row["row_id"])


def main():
    rows = list(csv.DictReader(open(HERE / "report.csv")))

    assigned, buckets = set(), OrderedDict((name, []) for name in SECTIONS)
    for row in rows:
        for name, predicate in SECTIONS.items():
            if row["row_id"] not in assigned and predicate(row):
                buckets[name].append(row)
                assigned.add(row["row_id"])
                break
    leftover = [r for r in rows if r["row_id"] not in assigned]
    if leftover:
        buckets["Other"] = leftover

    total_columns = len(rows[0])
    lines = [
        "# Run Summary",
        "",
        f"All {len(rows)} recorded configurations across "
        f"{len({r['notebook'] for r in rows if r['notebook'].startswith('notebooks/')})} notebooks. "
        f"Every row names exactly one notebook; ablation notebooks contribute one row per arm.",
        f"Full detail ({total_columns} columns) is in [`report.csv`](report.csv).",
        "",
        "Regenerate with `python docs/report/build_summary.py` after editing `report.csv`.",
        "",
        "## How to read this",
        "",
        "**Row IDs** encode where a row came from: `DN-` DenseNet, `SE-` SE-ResNeXt, `YOLO-` detector,",
        "`ORPHAN-` a checkpoint with no notebook. `-EXP-` is one arm of an ablation, `-RUN-` a standalone",
        "run, `-TPL-` an unexecuted template.",
        "",
        "**Split** is the single most important column. `test` means the locked hold-out split and is the",
        "only number that can be quoted as a result. `validation` was used to choose epochs and settings,",
        "so it is optimistic by construction and is not comparable to a test number. Never compare a",
        "validation row against a test row.",
        "",
        "**Two QWK values in one cell** (`0.79 (published) / 0.74 (ROI)`) mean the run was scored on both",
        "the published crops and the YOLO ROI view. The ROI value is the one the deployed service sees.",
        "",
        "**`—` means the notebook produced no such number.** It is never an estimate or a placeholder for",
        "a value that exists elsewhere. Metrics are copied verbatim from executed output cells.",
        "",
        "**Start with `Artifacts in production`.** Everything else is the evidence trail explaining why",
        "those settings were chosen. `report.csv` carries the full configuration for every row;",
        "[`audit_findings.md`](audit_findings.md) lists the discrepancies this audit turned up.",
        "",
    ]
    for name, group in buckets.items():
        if not group:
            continue
        lines += [
            f"## {name} ({len(group)})",
            "",
            "| ID | Model | Configuration | Input | Loss | Split | Acc | QWK | Macro F1 | G1 recall | Notebook |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in sorted(group, key=sort_key):
            nb = row["notebook"]
            link = f"[{Path(nb).name}](../../{nb})" if nb.startswith("notebooks/") else "—"
            lines.append(
                f"| {cell(row['row_id'])} | {cell(row['model_family'])} "
                f"| {cell(row['arm_or_config'])} | {cell(row['input_size'])} "
                f"| {cell(row['loss_function'])} | {cell(row['split_evaluated'])} "
                f"| {cell(row['accuracy'])} | {cell(row['qwk'])} | {cell(row['macro_f1'])} "
                f"| {cell(row['grade1_recall'])} | {link} |"
            )
        lines.append("")

    (HERE / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"summary.md regenerated from {len(rows)} rows")


if __name__ == "__main__":
    main()
