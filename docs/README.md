# Documentation Index

Knee osteoarthritis KL-grading service — YOLOv8n ROI detection → DenseNet-121 + SE-ResNeXt-50
ensemble → Grad-CAM explanation.

## Start here

| If you want to know | Read | Jump to |
| --- | --- | --- |
| What the deployed models score | [report/paper.md](report/paper.md) | [§4 Results](report/paper.md#4-results) |
| How a request is served end to end | [architecture.md](architecture.md) | — |
| What the networks look like | [ai-architecture.md](ai-architecture.md) | — |
| Where the data came from | [dataset.md](dataset.md) | — |
| How to run it locally | [runbook.md](runbook.md) | — |

## Defence quick links

The three questions a committee asks first, and the exact section that answers each:

| Question | Answer |
| --- | --- |
| "What accuracy did you achieve?" | [Abstract — production results table](report/paper.md#abstract) |
| "Why is Grade 1 so much worse?" | [§4.3 The Grade 1 bottleneck](report/paper.md#43-the-grade-1-bottleneck) |
| "How do you know the model looks at the joint?" | [§5 Explainability](report/paper.md#5-explainability) |
| "What would you improve next?" | [§7 Future work](report/paper.md#7-future-work-low-risk-accuracy-improvements) |
| "Show me every run you did" | [report/summary.md](report/summary.md) — all 78 notebook-backed configurations, grouped |
| "How do you know the numbers are real?" | [report/audit_findings.md](report/audit_findings.md) — notebook-to-record audit |

## All documents

### Results and research

| File | Contents |
| --- | --- |
| [report/paper.md](report/paper.md) | Full write-up: dataset, architectures, training config, results, explainability, future work, implementation Q&A |
| [report/summary.md](report/summary.md) | **Start here** — all 78 notebook-backed configurations, grouped by kind, production artifacts first |
| [report/report.csv](report/report.csv) | Full record — 78 rows x 40 columns, one row per configuration, each backed by exactly one executed notebook; regenerate the summary with `build_summary.py` |
| [report/audit_findings.md](report/audit_findings.md) | Every notebook read file by file, and the ten discrepancies that came out of it |
| [report/literature_review_cnn_kl_grading.md](report/literature_review_cnn_kl_grading.md) | Prior work on CNN-based KL grading |
| [paper/](paper/) | Reference papers (PDF) — see [paper/README.md](paper/README.md) |

### System documentation

| File | Contents |
| --- | --- |
| [architecture.md](architecture.md) | Inference architecture: request flow, services, ensemble weighting |
| [ai-architecture.md](ai-architecture.md) | Rendered architecture diagrams for both backbones and the inference pipeline |
| [dataset.md](dataset.md) | Dataset provenance, splits, folder layout, naming conventions |
| [runbook.md](runbook.md) | Local setup and how to run the service |
| [ci_cd.md](ci_cd.md) | CI pipeline and deployment |
| [testing.md](testing.md) | Unit test layout and how to run them |

### Diagrams

| File | Contents |
| --- | --- |
| [diagram/README.md](diagram/README.md) | Index of every `.drawio` source and its exported preview |

Diagram sources are draw.io XML. Open them at [app.diagrams.net](https://app.diagrams.net) or in the
desktop app; the `.png` next to each one is a preview that renders without any tooling.

### Presentation

| File | Contents |
| --- | --- |
| `AI Report.docx` | Formal report document |

> The `.docx` is maintained separately from `report/paper.md`. Where the two disagree, `paper.md` and
> `report.csv` are the record of what was actually measured — they are cross-checked against the
> executed notebook outputs. Known gaps in the `.docx` are listed in
> [§8 Implementation Q&A](report/paper.md#8-implementation-qa).

## Where the numbers come from

Every metric in `report.csv` traces back to a cell that actually executed in a notebook under
[`../notebooks/`](../notebooks/). The `notebook` column names that file for each row, and every one
of the 57 notebooks in the repository is mapped — ablation notebooks contribute one row per
compared configuration. Where a notebook produced no number, the cell reads `—`; nothing is
estimated or projected.

Checkpoints that exist on disk but that **no** notebook trains or evaluates — including the
deployed SE-ResNeXt-50 checkpoint — do **not** appear as rows here, since a row with no notebook
would contradict the rule above. They are documented in prose instead, in
[audit_findings.md](report/audit_findings.md). Rows marked `template_not_executed`, `aborted_run`,
`incomplete_run`, or `stale_outputs` deliberately carry no metrics.

`paper.md` has **not** yet been reconciled against this audit — see
[audit_findings.md](report/audit_findings.md) for the specific corrections it still needs.
