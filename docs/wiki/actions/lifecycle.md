# Lifecycle review and cell films

Lifecycle tables retain accepted starts, ends and intervening events.
Discover event names from `pymicroglia.measure.modules.lifecycle.EVENTS`.
Film rendering does not infer events again.

The display-only `follow` action accepts a saved run, recording stem,
identities, events, cell limit and span. Choose `recording`, `lifespan` or
`event`; the last makes one clip around each selected event, with
`event_hours` on either side. `dry_run=True` returns frame and caption
plans without writing films. Source frame alignment is retained.

Auto-Organotypic supplies display filtering and video encoding. Frame images
are retained only with `keep_frames=True`. Films do not alter measurements,
segmentation, ownership or event decisions.
