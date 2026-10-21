# Stage 03: Rotatable time sculpture

<!-- plan-records: {"depends_on": ["01", "02"], "exit_gate": ["Browser reports exactly the synthetic cell's observed frames as slices, including a visible missing-time gap.", "Drag rotation changes the displayed projection and reset restores the initial view without changing the selected identity or frame.", "Native time height matches the payload cadence and source-frame offset; no errors during cell changes."], "files_touched": [{"change": "NEW", "path": "src/pymicroglia/visualisation/review_assets/sculpture.js", "reason": "Canvas projection and rotatable observed outline stack."}, {"change": "NEW", "path": "src/pymicroglia/visualisation/review_assets/review.html", "reason": "Sculpture pane and coordinate/rotation controls."}, {"change": "NEW", "path": "src/pymicroglia/visualisation/review_assets/review.css", "reason": "Sculpture layout and input feedback."}, {"change": "NEW", "path": "tests/test_cell_review_browser.py", "reason": "Verify observed slice counts, gaps, native time height and rotation."}], "read_first": ["00_overview.md", "AGENTS.md", "src/pymicroglia/visualisation/panels/cell_review.py", "src/pymicroglia/visualisation/follow.py"]} -->

## Why this stage exists

Stacking the saved outlines with time as height exposes movement and shape history at a glance. Rotation should reveal that history without inventing an outline between observations.

## Prerequisites

01, 02

## Read first

- 00_overview.md
- AGENTS.md
- src/pymicroglia/visualisation/panels/cell_review.py
- src/pymicroglia/visualisation/follow.py

## Scope

- Render every observed contour at its native observation time in a browser canvas.
- Add drag rotation, wheel zoom, reset view and recording-coordinate versus centred display.
- Highlight the shared selected frame and use a time gradient with labelled time height and pixel axes.
- Leave absent observations as gaps; include disconnected components and holes.

## Out of scope

- A reconstructed cell surface or invented intermediate masks.
- Three-dimensional tissue coordinates; this is two spatial dimensions plus time.
- Cards and barcode are stages 04 and 05.

## Files touched

| Path | Change | Reason |
|---|---|---|
| src/pymicroglia/visualisation/review_assets/sculpture.js | NEW | Canvas projection and rotatable observed outline stack. |
| src/pymicroglia/visualisation/review_assets/review.html | NEW | Sculpture pane and coordinate/rotation controls. |
| src/pymicroglia/visualisation/review_assets/review.css | NEW | Sculpture layout and input feedback. |
| tests/test_cell_review_browser.py | NEW | Verify observed slice counts, gaps, native time height and rotation. |

## Implementation sketch

Represent each contour as (native x, native y, times_h[frame]); subtract the cell anchor only in centred mode. Fit the drawing volume to the selected identity's observed bounds. Apply a yaw and pitch projection and painter-order outlines by depth. Use a visual time-height scale, labelled hours, without changing source times. Drag changes angles; reset restores a readable oblique view. Current-frame outlines receive a stronger stroke; null observations generate no geometry.

## Exit gate

1. Browser reports exactly the synthetic cell's observed frames as slices, including a visible missing-time gap.
2. Drag rotation changes the displayed projection and reset restores the initial view without changing the selected identity or frame.
3. Native time height matches the payload cadence and source-frame offset; no errors during cell changes.

## Known risks

A crowded stack can obscure contours; supply zoom and coordinate choices while keeping every observation available. Axis labels must say time rather than tissue depth. Canvas redraws must remain responsive for the accepted 99-frame label histories.
