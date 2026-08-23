# Diagrams

Editable draw.io sources and their exported previews. Open a `.drawio` at
[app.diagrams.net](https://app.diagrams.net) or in the desktop app; the `.png` beside it renders
anywhere with no tooling.

## Architecture

| Source | Preview | Used in |
| --- | --- | --- |
| [`inference-pipeline.drawio`](inference-pipeline.drawio) | [`inference-pipeline.png`](inference-pipeline.png) | — (superseded by v2) |
| [`inference-pipeline-v2.drawio`](inference-pipeline-v2.drawio) | [`inference-pipeline-v2.drawio.png`](inference-pipeline-v2.drawio.png) | [`architecture.md`](../architecture.md) |
| [`densenet121.drawio`](densenet121.drawio) | `ai_model/densenet121-0{1..4}.png` | [`ai-architecture.md`](../ai-architecture.md) |
| [`resnext50.drawio`](resnext50.drawio) | `ai_model/resnext50-0{1..4}.png` | [`ai-architecture.md`](../ai-architecture.md) |
| [`ensemble-v3.drawio`](ensemble-v3.drawio) | — | [`paper.md`](../report/paper.md#diagrams) |

`densenet121.drawio` and `resnext50.drawio` are four-page files; each page exports to one numbered
PNG in [`ai_model/`](ai_model/) — overall, stage/block, bottleneck/layer, and transition/SE module.

## Dataset

| Source | Preview | Used in |
| --- | --- | --- |
| [`dataset_origin.drawio`](dataset_origin.drawio) | [`dataset_origin.png`](dataset_origin.png) | [`dataset.md` §1 Origin](../dataset.md#1-origin) |
| [`dataset_folder_architecture.drawio`](dataset_folder_architecture.drawio) | [`dataset_origin.drawio.png`](dataset_origin.drawio.png) | [`dataset.md` §4 Folder Architecture](../dataset.md#4-folder-architecture) |

The two `_url.txt` files hold shareable viewer links for the diagram of the same name.

## Training

| Source | Tabs |
| --- | --- |
| [`training/densenet121_training.drawio`](training/densenet121_training.drawio) | Original Training · YOLO Paired-View Adaptation · Evaluation · Overall |
| [`training/seresnext50_training.drawio`](training/seresnext50_training.drawio) | Original Training · YOLO Paired-View Adaptation · Evaluation · Overall |

## Editing

draw.io writes a `.$name.drawio.bkp` backup beside each file while you edit. Those are ignored by
git — do not commit them.
