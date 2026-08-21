# PyMicroglia Wiki

This wiki is the long-form PyMicroglia reference. It explains how to inspect,
run, review, and reproduce microglial bioluminescence time-lapse analyses.

## Start Here

| Page | Purpose |
|---|---|
| [Installation](getting-started/installation.md) | Install the core package or the optional analysis features. |
| [First analysis](getting-started/first-analysis.md) | Check the environment, inspect a pipeline, and start a recorded run. |
| [Check a run](getting-started/check-a-run.md) | Find outputs, read the run record, and recover the equivalent script. |
| [Action index](actions/README.md) | Find one of the 26 runnable operations. |
| [Single-cell dLuc pipeline](pipelines/dluc-single-cell.md) | Run the complete segmentation, tracing, controls, and rhythm workflow. |
| [Artefact store](concepts/artefact-store.md) | Understand what is kept, rebuilt, or reused. |
| [Recording and Batch](results/recording-and-batch.md) | Read results without opening the source pixels. |
| [Troubleshooting](troubleshooting/README.md) | Diagnose installation, action, store, and Fiji problems. |

## Choose the Right Entry Point

| Need | Use |
|---|---|
| A complete analysis of one recording | A [pipeline](pipelines/README.md). |
| One processing, measurement, figure, or video step | A named [action](actions/README.md). |
| Results that already exist on disk | `Recording` or `Batch` from the [results interface](results/README.md). |
| A quick health check | `pymicroglia doctor`. |

## Main Reference

| Page | Purpose |
|---|---|
| [Analysis flow](concepts/analysis-flow.md) | The order from raw pixels to a reviewable result. |
| [API reference](api-reference.md) | Public Python objects and functions grouped by task. |
| [Using actions](actions/using-actions.md) | Describe, validate, run, and record an action. |
| [Pipelines](pipelines/README.md) | End-to-end workflows and their scientific boundaries. |
| [Results](results/README.md) | Stored outputs, lazy access, and provenance. |
| [Workflows](workflows/README.md) | Task-based guides. |
| [Glossary](glossary/README.md) | Short definitions of package terms. |
| [Developer docs](developer/README.md) | Maintainer guidance and documentation upkeep. |
