<!-- aiify app map: app=microglia sources=279019c0066c638f -->
# App map: PyMicroglia

## Overview
PyMicroglia uses its existing scientific or workflow actions, the shared assistant panel and browser bridge. Local guidance, this map and the supplied verified shortcut are enabled by default. Extra web-route actions are optional. The assistant can use installed skills and files through its chosen local agent; it is not restricted to clicking the window.

## Screens

### Workspaces
Choose the microglial workspace or the connected Auto-Organotypic workflow.

### Exploration
Open a saved run and recording; inspect cells, traces and recorded morphology.

### Comparison
Combine saved recordings into a comparison workspace with explicit groups and units.

### Analysis and jobs
Validate declared pipeline/action parameters, supply a scientific claim and submit recorded work.

### Results
Inspect completed jobs and their recorded outputs; compare the same saved recordings.

## Tasks

### How to open a saved run
1. Use Exploration to open the saved run and selected recording.

### How to build a comparison
1. Use Comparison and retain the explicit recording selection, group labels and replication units.

### How to find an analysis
1. Describe the exact registered action. Read its current parameter defaults, choices and pending dependencies.

### How to validate a request
1. Use the existing action or pipeline validation before submitting work.

### How to run an analysis
1. Use action.run with declared parameters and the required claim; it submits to the window's existing job manager.

### How to check progress
1. Read workspace.overview or the native job list; inspect the existing job before resubmitting.

### How to inspect cell traces
1. Use the Exploration controls for the selected recording and cells.

### How to run rhythm analysis
1. Describe rhythm_analysis first and preserve its supported timing and preprocessing options.

### How to use optional web functions
1. Enable Web-route actions for /api/*; read calls run normally and other methods require confirmation.

### How to export recorded results
1. Use the existing result/export controls and retain the original recorded run and provenance.

## Terms

### Controls
The shared panel offers helper switches and the existing provider, model, effort and permission controls. Start New chat after changing helpers.

### Verified shortcut
workspace.overview reads this app's live or pinned native state in one call. It never starts an analysis or silently changes a task.
