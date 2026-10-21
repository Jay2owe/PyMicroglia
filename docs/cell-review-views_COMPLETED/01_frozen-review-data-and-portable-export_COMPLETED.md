# Stage 01: Frozen review data and portable export

<!-- plan-records: {"depends_on": [], "exit_gate": ["python -m pytest -q tests/test_cell_review_data.py passes.", "A synthetic exported HTML embeds its frames and data, loads without external URLs, and its receipt states no display filtering and no motion trails.", "python -m pytest -q tests/test_visualisation_limits.py passes."], "files_touched": [{"change": "NEW", "path": "src/pymicroglia/figure_tables/cell_review_data.py", "reason": "Prepare immutable native-frame observations and trace tables."}, {"change": "NEW", "path": "src/pymicroglia/figure_tables/cell_review_geometry.py", "reason": "Encode exact masks and all observed contours."}, {"change": "NEW", "path": "src/pymicroglia/visualisation/cell_review.py", "reason": "Public unattended portable browser-export function."}, {"change": "NEW", "path": "src/pymicroglia/visualisation/review_assets", "reason": "Small separate browser renderer assets."}, {"change": "MODIFY", "path": "pyproject.toml", "reason": "Include browser assets in installed distributions."}, {"change": "NEW", "path": "tests/test_cell_review_data.py", "reason": "Check offsets, sums, missing observations, source immutability and portable payloads."}], "read_first": ["00_overview.md", "AGENTS.md", "src/pymicroglia/measure/signal.py", "src/pymicroglia/visualisation/cell_tiles.py", "pyproject.toml"]} -->

## Why this stage exists

All four displays need one consistent source of cell identity, mask geometry, time and original signal. Establish that source before any view so selecting a cell can never change its tracking history.

## Prerequisites

None.

## Read first

- 00_overview.md
- AGENTS.md
- src/pymicroglia/measure/signal.py
- src/pymicroglia/visualisation/cell_tiles.py
- pyproject.toml

## Scope

- Validate single-channel raw and non-negative integer label stacks, source-frame offset and cadence.
- Prepare lossless mask row runs, all component contours, observed centres, native times, mask area, total raw intensity and the shared consecutive three-frame intensity mean.
- Encode unfiltered source frames as PNG display images with one recording-wide contrast range.
- Write an offline HTML review, display table and provenance receipt without modifying source files.
- Ship the HTML, CSS and JavaScript assets in the package.

## Out of scope

- Shape-change colouring belongs to stage 02.
- Rotating time sculpture belongs to stage 03.
- Cards and barcode belong to stages 04 and 05.
- Any segmentation, tracking or period fitting is outside this plan.

## Files touched

| Path | Change | Reason |
|---|---|---|
| src/pymicroglia/figure_tables/cell_review_data.py | NEW | Prepare immutable native-frame observations and trace tables. |
| src/pymicroglia/figure_tables/cell_review_geometry.py | NEW | Encode exact masks and all observed contours. |
| src/pymicroglia/visualisation/cell_review.py | NEW | Public unattended portable browser-export function. |
| src/pymicroglia/visualisation/review_assets | NEW | Small separate browser renderer assets. |
| pyproject.toml | MODIFY | Include browser assets in installed distributions. |
| tests/test_cell_review_data.py | NEW | Check offsets, sums, missing observations, source immutability and portable payloads. |

## Implementation sketch

Public function: cell_review(raw, labels, *, output, title=None, source_frame_offset=0, frame_interval_h=0.5, display_lut='red', display_range=None). Preparation returns a JSON-safe payload with ids, times_h, source shape, PNG frames and per-cell observations. An observation stores anchor, centre, bounds, exact row runs and component contours. Every source frame exists; frames without a labelled observation have null area and signal. Reuse measure.signal.centered_frame_mean(window=3), which refuses to bridge missing frames. Write sibling <stem>_measurements.csv and <stem>_review.json receipts. The renderer loads a finished payload and assets; it does not filter or segment. Test a small translated cell with a gap and a source offset against hand-calculated sums and the original input hashes.

## Exit gate

1. python -m pytest -q tests/test_cell_review_data.py passes.
2. A synthetic exported HTML embeds its frames and data, loads without external URLs, and its receipt states no display filtering and no motion trails.
3. python -m pytest -q tests/test_visualisation_limits.py passes.

## Known risks

Contours must include holes and disconnected pieces; exact row runs remain the authority for occupied pixels. A source offset must shift labels to the native frame rather than shortening the movie. Reject incompatible stack geometry rather than guessing. Large exports must check free disk space and preserve failure receipts.
