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

After tracking, `tracking.post_refine.refine_motion` can redraw each tracked
cell's outline with a refiner without changing identities. For LV200 red
(CD68-mCherry) use `method="lv200_red"`: it outlines the whole cell rather than
the tracking core, then `tracking.outline_morph` replaces one-frame failures
(the refiner kept the tracking outline, or the outline fell below half or rose
past twice its neighbours' area) with a morph of the good outlines either side.
Measure the returned `measurement_labels` with
`measure.expanded_channels.measure(..., options={"background_statistic":
"median"})`: each outline grown 3 px, minus the median of a 4-12 px local ring
times the grown area.

`tracked_cell_video` and `tracked_cell_image` draw the accepted per-identity
outside boundaries over original photons. Outline width, opacity and colours,
contrast, display-only smoothing, timestamps, playback speed, and the still
frame are configurable. Use `pymicroglia describe cell_eligibility`,
`pymicroglia describe tracked_cell_video`, or
`pymicroglia describe tracked_cell_image` for the live parameters.

## Correcting tracks and Jamie's review

After tracking, an agent authorised to correct a recording follows Motion's
`motion_edit/guide/agent-review.md`: audit the finished run, make each
correction as a new editor session (the original labels never change), and
render Before/After/Changes. Jamie reviews the session in the workbench's
track review, `pymicroglia ui --motion-root "<Motion folder>"`, which starts
Motion's editor service with the window; one session also opens with
Motion's `edit_tracks.cmd <session folder>`. The video is an aid, not the
review: measurements use only a session Jamie has reviewed, exported through
the editor's measurement handoff.
