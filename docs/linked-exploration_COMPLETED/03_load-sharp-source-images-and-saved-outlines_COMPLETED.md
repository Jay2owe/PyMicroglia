# Stage 03: Load sharp source images and saved outlines

<!-- plan-records: {"depends_on": ["02"], "exit_gate": ["python -m pytest tests/test_ui_exploration_images.py tests/test_cell_review_data.py -q passes.", "Native contrast mapping is exact and no blur is applied.", "A missing frame returns no invented selected-cell geometry.", "Raw and label input bytes are unchanged after frame and history reads."], "files_touched": [{"change": "NEW", "path": "src/pymicroglia/ui/exploration_images.py", "reason": "Lazy frame and lossless mask geometry adapter"}, {"change": "MODIFY", "path": "src/pymicroglia/ui/server.py", "reason": "Frame and history endpoints"}, {"change": "NEW", "path": "tests/test_ui_exploration_images.py", "reason": "Native pixels, offset and identity preservation"}], "read_first": ["00_overview.md", "src/pymicroglia/measure/mask_geometry.py", "src/pymicroglia/figure_tables/cell_review_data.py", "src/pymicroglia/measure/inputs.py", "src/pymicroglia/ui/server.py", "tests/test_cell_review_data.py"]} -->

## Why this stage exists

The image and rotating outline history must describe the same saved identity and time as the measurements, without changing the tracking result.

## Prerequisites

02

## Read first

- 00_overview.md
- src/pymicroglia/measure/mask_geometry.py
- src/pymicroglia/figure_tables/cell_review_data.py
- src/pymicroglia/measure/inputs.py
- src/pymicroglia/ui/server.py
- tests/test_cell_review_data.py

## Scope

- Resolve only image and label sources declared by the selected recording.
- Read requested TIFF planes lazily, honour source offsets and channel indexing, and use a fixed recording contrast.
- Provide exact existing contours, mask runs, centres and missing states; provide bounded selected-cell outline history.
- Keep trails a display-only option and compare available base/refined masks only when their provenance identifies both sources.

## Out of scope

- No segmentation, denoising, mask refinement, identity reassociation or full-stack export.
- Browser rendering is stage 04.

## Files touched

| Path | Change | Reason |
|---|---|---|
| src/pymicroglia/ui/exploration_images.py | NEW | Lazy frame and lossless mask geometry adapter |
| src/pymicroglia/ui/server.py | MODIFY | Frame and history endpoints |
| tests/test_ui_exploration_images.py | NEW | Native pixels, offset and identity preservation |

## Implementation sketch

GET /api/exploration/frame?run=...&stem=...&frame=... returns a PNG data URL, native dimensions and observation geometry for that saved frame. GET /api/exploration/history adds one selected identity's observed contours with exact frame indices. Reuse measure.mask_geometry.observation; keep holes and disconnected components. Decode a requested TIFF page, with bounded fallback only when the format cannot supply a lazy plane. Do not allocate or download an entire large compressed stack for a slider movement.

## Exit gate

1. python -m pytest tests/test_ui_exploration_images.py tests/test_cell_review_data.py -q passes.
2. Native contrast mapping is exact and no blur is applied.
3. A missing frame returns no invented selected-cell geometry.
4. Raw and label input bytes are unchanged after frame and history reads.

## Known risks

A raw stack may contain extra axes or unavailable source files. Use explicit declared channel/frame alignment, report unavailable views, and preserve usable trace exploration. Bound geometry and frame caches by source signatures.
