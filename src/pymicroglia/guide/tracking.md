# Tracking boundary

`pymicroglia.tracking` runs a frozen Motion engine shipped in the wheel.
`tracking.status()` reports whether its entry point resolves. The handoff
first builds Motion's six pinned inputs from the U-Net mask and registered
photons; no fallback segmentation or tracking rule is substituted.

The contract records accepted labels, identity ownership, tracking tables,
source images, frame mapping and fingerprints. Existing verified outputs can
be measured without rerunning tracking. Motion retains its assignment objective,
remembered identities, review TIFFs and issue annotations. Biological signal
is measured from the original unmasked photons, not Motion's scaled counts.

The automated chain then audits which finished identities may enter each
consumer. By default, an internal gap longer than four hours or a missing
fraction of at least 0.5 excludes a cell from analysis only; videos and still
images keep all identities for review. `eligibility_max_gap_h`,
`eligibility_max_missing_fraction`, and `eligibility_exclude_from` make the
limits and the `analysis`, `videos`, and `images` destinations independent.
Filtered views preserve identity numbers and never feed back into Motion.

`tracked_cell_video` and `tracked_cell_image` draw the accepted per-identity
outside boundaries over original photons. Outline width, opacity and colours,
contrast, display-only smoothing, timestamps, playback speed, and the still
frame are configurable. Use `pymicroglia describe cell_eligibility`,
`pymicroglia describe tracked_cell_video`, or
`pymicroglia describe tracked_cell_image` for the live parameters.
