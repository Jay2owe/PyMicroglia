# Lifecycle review and cell films

Lifecycle tables retain accepted starts, ends and intervening events.
Discover event names from `pymicroglia.measure.modules.lifecycle.EVENTS`.
Film rendering does not infer events again.

An intensity-only run has no lifecycle classifications. For post-tracking
refinement, classify starts, ends and divisions from the **original accepted
Motion masks and unclaimed foreground**, then display those saved decisions
with the refined outlines. The explicit `calculate_lifecycle_review` helper
in `pymicroglia.ui.exploration_lifecycle` uses the original, checksum-pinned
tracking attachment or explicitly supplied original inputs. It rejects
differences in identity presence or recording time. It never substitutes
refined masks for missing original inputs.

The review is saved under `AI_Exports/lifecycle_review`, with its input
fingerprints, native method settings and reproducing script. Verified tables
are shared by Events, the track timeline, event markers and native figures
and cell films. Original measurements and the main manifest stay unchanged.
Candidate events retain their candidate status; they are not biological
confirmations. Opening a page only reads the saved review.

The display-only `follow` action accepts a saved run, recording stem,
identities, events, cell limit and span. Choose `recording`, `lifespan` or
`event`; the last makes one clip around each selected event, with
`event_hours` on either side. `dry_run=True` returns frame and caption
plans without writing films. Source frame alignment is retained.

Auto-Organotypic supplies display filtering and video encoding. Frame images
are retained only with `keep_frames=True`. Films do not alter measurements,
segmentation, ownership or event decisions.

The **Events** page in the left PyMicroglia extras sidebar opens a cell-image
grid of all saved lifecycle events, state changes and tracking gap assessments.
Inner tabs select All events or individual event types. Each tile represents
one recorded occurrence, with cell identity, time, status and any missing mask
shown explicitly. All matching previews fill automatically with progress;
opening a tab does not run analysis or encode films. Empty event types show
that there are no saved matching events.

Click a tile to scrub its original frames, play an existing film, or explicitly
save an event video. Back to grid returns to the same filtered gallery.
Display controls set context, speed, crop size and colour; noise filtering is off.
Event filters and options sit in a panel on the right beside the grid or player; the
shared left navigation keeps the same appearance across pages.
Tracking assessments keep their status and do not become confirmed births or
merges. Inspect opens the same cell and frame in the existing linked plots.

For scripts, `event_frames` restricts films to exact saved event frames, while
`frame_range=[first, last]` trims the inclusive native frame range. Saved frame
and time coordinates are retained. New UI films and their reproducing Python
scripts are written to the selected run's `AI_Exports/event_videos` folder.
