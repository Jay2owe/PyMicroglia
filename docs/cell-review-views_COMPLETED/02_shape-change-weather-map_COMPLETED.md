# Stage 02: Shape change weather map

<!-- plan-records: {"depends_on": ["01", "01a"], "exit_gate": ["Synthetic browser review shows the expected gained/lost pixels for translation and a changed protrusion.", "Centred translation yields zero gained/lost pixels; native-coordinate translation yields the hand-calculated nonzero counts.", "Selecting a missing frame shows a blank mask and no comparison to an earlier frame; browser reports no page errors."], "files_touched": [{"change": "NEW", "path": "src/pymicroglia/visualisation/review_assets/review.html", "reason": "Dashboard shell and shape-change controls."}, {"change": "NEW", "path": "src/pymicroglia/visualisation/review_assets/review.css", "reason": "Responsive sharp-pixel presentation."}, {"change": "NEW", "path": "src/pymicroglia/visualisation/review_assets/review.js", "reason": "Shared state, native crops, time playback and event routing."}, {"change": "NEW", "path": "src/pymicroglia/visualisation/review_assets/changes.js", "reason": "Exact gain/loss comparisons and rendering."}, {"change": "NEW", "path": "tests/test_cell_review_browser.py", "reason": "Browser checks of aligned and original changes, missing masks and linked frame selection."}], "read_first": ["00_overview.md", "AGENTS.md", "src/pymicroglia/visualisation/overlays.py", "tests/test_cell_motion_trails.py"]} -->

## Why this stage exists

Mask gains and losses are easier to spot as coloured pixels than as competing outlines. The view should let the user distinguish changes in the recording coordinates from changes after centring the same cell.

## Prerequisites

01 and 01a must be completed.

## Read first

- 00_overview.md
- AGENTS.md
- src/pymicroglia/visualisation/overlays.py
- tests/test_cell_motion_trails.py

## Scope

- Add shared native-frame slider, cell selector and play/pause controls.
- Draw cyan gained pixels and amber lost pixels against the previous consecutive observed frame.
- Offer original-coordinate and cell-centred comparisons using integer translations without resampling masks.
- Show raw source crop, current outline, colour legend and missing-comparison explanation.

## Out of scope

- Biological classifications of gained or lost pixels.
- Gap filling or identity assignment.
- Sculpture, cards and barcode are stages 03–05.

## Files touched

| Path | Change | Reason |
|---|---|---|
| src/pymicroglia/visualisation/review_assets/review.html | NEW | Dashboard shell and shape-change controls. |
| src/pymicroglia/visualisation/review_assets/review.css | NEW | Responsive sharp-pixel presentation. |
| src/pymicroglia/visualisation/review_assets/review.js | NEW | Shared state, native crops, time playback and event routing. |
| src/pymicroglia/visualisation/review_assets/changes.js | NEW | Exact gain/loss comparisons and rendering. |
| tests/test_cell_review_browser.py | NEW | Browser checks of aligned and original changes, missing masks and linked frame selection. |

## Implementation sketch

Expand the selected observation's exact runs into integer pixel keys only for the displayed comparison. In centred mode subtract each observation's stored integer anchor; in original mode keep native coordinates. Gained=current minus previous; lost=previous minus current. A pure whole-pixel translation must show zero centred changes and nonzero original changes. If either consecutive observation is absent, show no coloured comparison and do not reach across the gap. Shared state is {cellId, frame, playing, coordinateMode}; each view subscribes to a single update function. Keep the browser rendering test runnable using Playwright without adding a runtime dependency.

## Exit gate

1. Synthetic browser review shows the expected gained/lost pixels for translation and a changed protrusion.
2. Centred translation yields zero gained/lost pixels; native-coordinate translation yields the hand-calculated nonzero counts.
3. Selecting a missing frame shows a blank mask and no comparison to an earlier frame; browser reports no page errors.

## Known risks

Integer centring removes whole-pixel translation, not rotation or subpixel motion; describe the view accurately. Browser image loading must not allow a late frame to paint over a newer selection. Disable image smoothing for every movie canvas.
