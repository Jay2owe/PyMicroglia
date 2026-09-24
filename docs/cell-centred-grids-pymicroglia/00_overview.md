# Tracked-cell image and video grids in PyMicroglia

Prerequisite: Auto-Organotypic's `docs/cell-centred-grids-auto/` plan must be
complete before stage 01 here. The canonical cross-package design remains in
Auto-Organotypic's `docs/consolidation/cell-centred-grid-visualisations.md`.

## End goal

Each recording gains a still grid with all tracked cells across each cell's chosen circadian cycle and a video grid with all tracked cells through the full recording. Every frame uses a cell-centred crop of original photons; the default crop is based on the largest tracked footprint, with own-cell and explicit pixel-size options. Auto-Organotypic's shared grids provide appearance, labels, timing and saving.

## Why

Existing tracked-cell exports outline the whole field or follow individual cells, while the shared grids show fixed recording positions. The reusable tile-source seam built in Auto-Organotypic lets PyMicroglia supply moving cell views to those existing grids without a second rendering implementation.

## Architecture

Original photon TIFF + destination-specific eligible label stack -> PyMicroglia lazy CellTile sources -> auto_organotypic.grid.stack_to_grid / auto_organotypic.video_grid.stack_to_video_grid -> existing visual/images and visual/videos output records. The canonical design is in ../Auto-Organotypic/docs/consolidation/cell-centred-grid-visualisations.md; complete the Auto-Organotypic tile-source plan before executing PyMicroglia stage 01.

## Stage map

| NN | Name | Goal | Rough size | Depends on |
|---|---|---|---|---|
| 01 | moving-cell-tile-sources | Give the shared grid renderers one lazy, original-photon tile source per tracked identity with a fixed crop that follows its position. | 1–2 days | none |
| 02 | cell-cycle-image-grid | Render all tracked cells as rows across each cell's best cycle using Auto-Organotypic's still grid, with shared recording or event time available. | 1–2 days | 01 |
| 03 | cell-video-grid | Render every tracked cell through the full source movie via Auto-Organotypic's existing video grid. | 1–2 days | 01, 02 |
| 04 | pipeline-and-discoverability | Make the new grids part of the automated microglia run and expose their settings and saved outputs to users. | 1–2 days | 02, 03 |

## House rules

Read AGENTS.md and the canonical consolidation design. PyMicroglia owns only tracking-aware crops, trace inputs and pipeline wiring; Auto-Organotypic owns rendering. Source photons and tracked labels remain unchanged. Preserve all tracked identities in the selected images/videos eligibility views and show missing states. Use the same lookup tables, ranges, captions, grid layout and save conventions by passing through upstream options. Stage only this work amid the dirty tree. Run focused checks; no Graphify refresh unless requested or an explicit release check requires it.

## How to run

Use /do-step docs\cell-centred-grids-pymicroglia.
