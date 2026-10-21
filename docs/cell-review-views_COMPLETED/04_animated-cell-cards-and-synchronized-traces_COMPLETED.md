# Stage 04: Animated cell cards and synchronized traces

<!-- plan-records: {"depends_on": ["01", "02"], "exit_gate": ["Browser changes the shared movie frame when either signal or area trace is clicked, and every card cursor agrees.", "Pinning and changing selected cell preserve existing IDs and introduce no duplicate cards.", "A missing mask leaves signal/area gaps and a clear absent-mask label; canvas image smoothing is disabled."], "files_touched": [{"change": "NEW", "path": "src/pymicroglia/visualisation/review_assets/cards.js", "reason": "Animated native-pixel cards and saved-data trace panels."}, {"change": "NEW", "path": "src/pymicroglia/visualisation/review_assets/review.html", "reason": "Card pinning and synchronized trace UI."}, {"change": "NEW", "path": "src/pymicroglia/visualisation/review_assets/review.css", "reason": "Readable movie and trace card layout."}, {"change": "NEW", "path": "tests/test_cell_review_browser.py", "reason": "Verify click-to-time, cursor alignment, pinning and blank observations."}], "read_first": ["00_overview.md", "AGENTS.md", "src/pymicroglia/visualisation/cell_tiles.py", "src/pymicroglia/measure/signal.py"]} -->

## Why this stage exists

Linking sharp cell movies to signal and mask-area traces makes an interesting trace point immediately inspectable. Several pinned cards support comparisons without altering the underlying measurements.

## Prerequisites

01, 02

## Read first

- 00_overview.md
- AGENTS.md
- src/pymicroglia/visualisation/cell_tiles.py
- src/pymicroglia/measure/signal.py

## Scope

- Render the selected cell and a small set of pinned comparison cells as animated cards.
- Draw total raw intensity with the consecutive three-frame mean and mask area from prepared tables.
- Move one time cursor across all traces and movies; allow trace clicks to choose a native frame.
- Show absent masks explicitly, preserve source time and avoid interpolating trace gaps.

## Out of scope

- New rhythm fitting or classifications.
- Smoothing source images or filling missing masks.
- Recording barcode and action discovery belong to stage 05.

## Files touched

| Path | Change | Reason |
|---|---|---|
| src/pymicroglia/visualisation/review_assets/cards.js | NEW | Animated native-pixel cards and saved-data trace panels. |
| src/pymicroglia/visualisation/review_assets/review.html | NEW | Card pinning and synchronized trace UI. |
| src/pymicroglia/visualisation/review_assets/review.css | NEW | Readable movie and trace card layout. |
| tests/test_cell_review_browser.py | NEW | Verify click-to-time, cursor alignment, pinning and blank observations. |

## Implementation sketch

Cards read the shared frame, payload PNG and exact saved contours. Crop geometry is fixed for each cell's largest observed extent, with black padding outside the image and no smoothing. SVG trace paths are split at null values; the raw intensity line stays visible beneath the explicitly labelled three-frame mean. All cursors use native times_h, and clicks choose the closest native frame. The selected cell is always the first card; pinned comparison identities are stored only as display state.

## Exit gate

1. Browser changes the shared movie frame when either signal or area trace is clicked, and every card cursor agrees.
2. Pinning and changing selected cell preserve existing IDs and introduce no duplicate cards.
3. A missing mask leaves signal/area gaps and a clear absent-mask label; canvas image smoothing is disabled.

## Known risks

Independent trace scales can exaggerate magnitude comparisons; label each card's units and scales. Do not turn a null measurement into zero. A movie may show raw tissue at a missing-mask time, but must never display an invented mask.
