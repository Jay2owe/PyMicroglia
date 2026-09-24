# Stage 01: moving-cell-tile-sources

<!-- plan-records: {"depends_on": [], "exit_gate": ["python -m pytest tests/test_cell_tiles.py tests/test_follow.py -q", "Two cells, a long thin shape, an edge crossing and an internal gap remain correctly centred with no clipped observed footprint.", "The raw and label TIFF hashes are unchanged; source_frame_offset maps every label frame to its correct photon frame."], "files_touched": [{"change": "NEW", "path": "src/pymicroglia/visualisation/cell_tiles.py", "reason": "Tracking-aware generic tile sources and crop policy"}, {"change": "NEW", "path": "tests/test_cell_tiles.py", "reason": "Moving centres, footprint sizing, offset, padding, missing frames and trace checks"}], "read_first": ["00_overview.md", "AGENTS.md", "CLAUDE.md", "src/pymicroglia/figure_tables/follow.py", "src/pymicroglia/eligibility.py", "src/pymicroglia/tracking/engine.py", "src/pymicroglia/visualisation/tracked_outlines.py", "tests/test_follow.py"]} -->

## Why this stage exists

Give the shared grid renderers one lazy, original-photon tile source per tracked identity with a fixed crop that follows its position.

## Prerequisites

All stages in Auto-Organotypic's `docs/cell-centred-grids-auto/` plan are
`_COMPLETED`. No earlier PyMicroglia stage.

## Read first

- 00_overview.md
- AGENTS.md
- CLAUDE.md
- src/pymicroglia/figure_tables/follow.py
- src/pymicroglia/eligibility.py
- src/pymicroglia/tracking/engine.py
- src/pymicroglia/visualisation/tracked_outlines.py
- tests/test_follow.py

## Scope

- Build one source per positive identity in the destination's eligible labels, using the recorded source-frame offset.
- Derive per-frame centroids and full-life required crop size from labels; never assume the morphology module ran.
- Implement crop_basis='largest_cell' default, 'own_cell', and exact crop_size_px=(width,height); automatic crops use Auto-Organotypic final-outline tight/standard/wide factors.
- Return original numeric photon crops lazily; pad at image edges without shifting centre or biasing auto contrast.
- Mark missing identities, hold their last known centre for location only, and calculate a display-only cell brightness trace from observed labelled pixels.

## Out of scope

- Still-grid action and cycle selection belong to stage 02.
- Video-grid action belongs to stage 03.
- Pipeline wiring belongs to stage 04.

## Files touched

| Path | Change | Reason |
|---|---|---|
| src/pymicroglia/visualisation/cell_tiles.py | NEW | Tracking-aware generic tile sources and crop policy |
| tests/test_cell_tiles.py | NEW | Moving centres, footprint sizing, offset, padding, missing frames and trace checks |

## Implementation sketch

Proposed factory: cell_tiles(raw, labels, *, source_frame_offset=0, frame_interval_h=None, crop_basis='largest_cell', crop='tight', crop_size_px=None, identity_prefix='Cell') -> list[auto_organotypic.render.tile_source.TileSource]. For each observed identity and frame, compute centroid of exactly its labelled pixels and maximum x/y distance from centroid to its own pixels. The default square side covers the maximum reach of every included identity over all frames, then multiply by the accepted final-outline crop factor (tight 1.05, standard 1.25, wide 1.50); import the factor from Auto-Organotypic rather than duplicating it. Own-cell uses one side per identity. Exact (width,height) bypasses the preset and refuses clipping with a named minimum. Before first observation use the first centre; during/after gaps use the last centre, but mark every absent frame. Crop only original photons. Keep a shared raw reader and pass a serialisable source/label fingerprint, identity and crop recipe to the upstream TileSource. Trace points come from raw photons within observed identity masks and are display-only.

## Exit gate

1. python -m pytest tests/test_cell_tiles.py tests/test_follow.py -q
2. Two cells, a long thin shape, an edge crossing and an internal gap remain correctly centred with no clipped observed footprint.
3. The raw and label TIFF hashes are unchanged; source_frame_offset maps every label frame to its correct photon frame.

## Known risks

A large cell must be found by maximum centred extent, not just area. Own-cell crops must retain native pixel scale with symmetric grid padding. Existing default analysis eligibility may exclude cells retained for images, so do not require tracked measurement tables.
