# Recording and measurement inputs

A movie declaration identifies accepted labels, raw images and tracking exports.
Measurements retain accepted cell identities and label-to-source frame mapping.
Frame interval and spatial calibration must describe the original recording.

`pymicroglia.measure.load_config(path)` accepts movies, enabled modules,
per-module settings, conditions, windows, contrasts, metric groups, channels,
object sets and explicitly granted side tables. Validation rejects unknown
settings and ambiguous joins. Relative paths resolve against the configuration.

Keep recording, cell, biological sample and condition identifiers distinct.
A cell is not automatically an independent biological replicate. Pooled inputs
retain recording identifiers, separating repeated numeric cell identities.
Workflow requests resolve metric groups to recorded column lists.
