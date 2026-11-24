# Pixel and table measurements

Modules register their inputs, settings and output columns through the shared measurement declarations. The measurement runner aligns inputs, invokes the configured modules, and saves their tables through the common store.

Territory and turnover avoid repeated outline scans; contacts and local background rings use equivalent pixel operations. Current implementation, original-source comparisons, regression checks and regeneration instructions are linked in the [measurement performance report](../../../../development/speed-pass-20261011/measurement-access/README.md).

<!-- agent-work-trail:begin -->
## Current work
Status: complete (does not imply scientific approval).
- deliverable: [territory.py](territory.py)
- deliverable: [surveillance.py](surveillance.py)
- deliverable: [contacts.py](contacts.py)
- deliverable: [intensity.py](intensity.py)

## Recent changes
- 2026-10-22: Territory, turnover, contact and background-ring pixel work optimized; exact original-output comparisons and complete measurement timings retained in the measurement performance report.
<!-- agent-work-trail:end -->
