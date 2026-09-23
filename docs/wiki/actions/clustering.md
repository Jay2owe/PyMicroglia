# Group whole-cell fingerprints

The `cluster` action groups whole-cell summaries using the validated
`clustering_options` mapping. A whole-cell fingerprint describes an identity
over its observed recording; frame-level states describe individual observations.

Keep training, validation and sample groups explicit. A learned group describes
the supplied features, not an established cell type. Trajectory comparisons
retain declared time-warping settings. Changing a scientific setting or seed
changes provenance; changing a colour does not refit a model.
