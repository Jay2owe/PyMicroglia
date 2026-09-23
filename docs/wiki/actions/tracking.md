# Tracking boundary

`pymicroglia.tracking` runs the frozen Motion engine shipped in the wheel.
`tracking.status()` reports whether its entry point resolves. The U-Net mask
and registered photons first become six pinned Motion inputs; no fallback
tracking rule is substituted.

The contract records accepted labels, identity ownership, tracking tables,
source images, frame mapping and fingerprints. Existing verified outputs can
be measured without rerunning tracking. Motion retains its assignment objective,
remembered identities, review TIFFs and issue annotations. Measurements read
the original unmasked photons, not Motion's scaled counts.

After tracking, `cell_eligibility` can make separate identity-preserving label
views for analysis, videos and images. Its defaults reject an internal absence
longer than four hours or a missing fraction of at least 0.5 from analysis only.
It does not write back into Motion or change identity numbers.

`tracked_cell_video` and `tracked_cell_image` draw configurable per-identity
outside boundaries over original photons. Their contrast, temporal/spatial
smoothing, outline width, opacity, colours, timestamps and playback are display
settings only.
