# Notebook Configuration Audit

This ledger records one completed configuration per row, even when several configurations were run in one notebook. Each row links back to the same source notebook and the model report that contains the broader experiment narrative.

The initial audit found 11 missing per-configuration rows from three completed SE-ResNeXt ablation notebooks. They are recorded in `notebook_config_audit.csv` and included in the consolidated workbook at [`../all_experiments.xlsx`](../all_experiments.xlsx).

Notebooks with no executed outputs, data preparation only, paper reproduction, or post-hoc comparison of existing checkpoints are not recorded as new training configurations.
