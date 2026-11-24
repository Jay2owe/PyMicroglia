# figure_tables

<!-- agent-work-trail:begin -->
## Current work
Status: draft (does not imply scientific approval).
- deliverable: [README.md](README.md)
- support: [tracking_tower.py](tracking_tower.py)

## Recent changes
- 2026-10-23: Added optional actual-acquisition hours to cell-value maps using the existing occupancy integrator; preserved historical no-clock behavior and other overlap rules. Fourteen focused checks passed.
- 2026-10-23: Added optional exact selected-component paths and explicit edge permissions; existing defaults preserved, ten focused checks passed
<!-- agent-work-trail:end -->

## Tracking tower component paths

`tracking_tower.analyse` accepts optional `path_components` and `allowed_links`.
The former maps `(identity, zero-based label frame)` to selected `(y, x)` pixel
arrays; the latter explicitly permits edges ending at that frame. Both exact
endpoints must exist and every selected pixel must belong to the saved identity.
Without these options the existing whole-label behaviour is unchanged.

Selected geometry affects paths only. Full-field snapshot outlines remain
unchanged. Supply fluorescence for the selected geometry, or missing values;
whole-label photometry must not silently stand in for a smaller selection.
A selection is display evidence, not a new tracking identity or biological fit
eligibility. The caller retains decision provenance and clock-gap exclusions.
