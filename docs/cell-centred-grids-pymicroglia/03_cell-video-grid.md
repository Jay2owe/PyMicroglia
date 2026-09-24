# Stage 03: cell-video-grid

<!-- plan-records: {"depends_on": ["01", "02"], "exit_gate": ["python -m pytest tests/test_cell_video_grid.py tests/test_video_parity.py -q", "Output frame count equals the complete requested source window and every tile depicts the same source frame at each output step.", "No identity in the videos eligibility view is omitted, including one excluded from analysis."], "files_touched": [{"change": "NEW", "path": "src/pymicroglia/visualisation/cell_video_grid.py", "reason": "Video-grid action forwarding shared options"}, {"change": "NEW", "path": "tests/test_cell_video_grid.py", "reason": "Full-duration synchronized frames, missing labels and video options"}], "read_first": ["00_overview.md", "AGENTS.md", "CLAUDE.md", "src/pymicroglia/video/__init__.py", "tests/test_video_parity.py"]} -->

## Why this stage exists

Render every tracked cell through the full source movie via Auto-Organotypic's existing video grid.

## Prerequisites

01, 02

## Read first

- 00_overview.md
- AGENTS.md
- CLAUDE.md
- src/pymicroglia/video/__init__.py
- tests/test_video_parity.py

## Scope

- Provide a public PyMicroglia video-grid action built from the videos eligibility view and moving cell tiles.
- Default to every source frame in order on one recording clock; keep all existing video-grid layout, timestamp, speed, contrast and encoder controls.
- Expose explicit phase/window settings where the upstream grid supports them; record any truncation or resampling.
- Keep missing-cell frames as current raw photon crops with visible missing status; never duplicate an old image.

## Out of scope

- Cell crop policy belongs to stage 01.
- Still-grid selection belongs to stage 02.
- Pipeline wiring belongs to stage 04.

## Files touched

| Path | Change | Reason |
|---|---|---|
| src/pymicroglia/visualisation/cell_video_grid.py | NEW | Video-grid action forwarding shared options |
| tests/test_cell_video_grid.py | NEW | Full-duration synchronized frames, missing labels and video options |

## Implementation sketch

Proposed call: cell_video_grid(raw, labels, *, output_dir=None, output_name=None, source_frame_offset=0, crop_basis='largest_cell', crop='tight', crop_size_px=None, **video_options). Construct TileSources via cell_tiles and call auto_organotypic.video_grid.stack_to_video_grid(tiles, align='start', **video_options). Tile source returns the current source-frame numeric crop while reporting whether that identity was observed; downstream renderer handles colour, range, text, layout and encoding. Use upstream AI_Exports video-grid directory rules for direct calls. Reject output dimensions exceeding encoder limits rather than scaling cells.

## Exit gate

1. python -m pytest tests/test_cell_video_grid.py tests/test_video_parity.py -q
2. Output frame count equals the complete requested source window and every tile depicts the same source frame at each output step.
3. No identity in the videos eligibility view is omitted, including one excluded from analysis.

## Known risks

A missing cell still has raw photons at the held location; mark missing so this is not mistaken for observation. Encoder limits can be hit with many cells; report dimensions and a user-adjustable setting rather than silently shrinking tiles.
