# Grey cell outlines

Common case: load a recording, open **Grey outlines**, keep cells with a saved
significant rhythm in their original colour, and show the others in grey.
All measurements, identities, traces and statistical populations remain intact.
Exclusions continue to have their own independent rules.

The grey-outline editor reuses the exclusions rule builder. Rules can combine
saved rhythms, missing frames, tracking events, measurements and distances with
AND, OR, NOT and XOR. Missing evidence follows an explicit keep/grey/error policy.
The editor remembers draft and applied settings independently across reloads.
Changing appearance never launches a rhythm analysis.

For an existing fixed-circle comparison, import its per-cell decision CSV into
the grey editor and use **Imported decision passed = Yes**, keeping matching
cells. The CSV must contain `identity` and either `passed` or
`passes_fixed_circle`; an optional `recording` column selects the current
recording. True/false values are required; omitted cells have unknown evidence.
Importing decisions does not recompute or validate the underlying scientific test.

`tracked_cell_image` and `tracked_cell_video` accept `cell_filter`, either a
saved filter JSON filename, its decoded result, or a recipe containing `run`,
`stem`, `expression`, `mode`, `unknown`, optional `analysis_ids` and `decisions`.
Recipes evaluate saved evidence; frozen results retain their recorded decisions.
Every identity in the rendered labels must have a decision. The renderer keeps
source pixels and native labels unchanged and records the grey identities and
filter evidence in its output receipt. `filter_grey_colour` defaults to `#aaaaaa`.

```python
from pymicroglia.video import tracked_cell_video

tracked_cell_video(
    "photons.tif", labels="tracked_labels.tif",
    cell_filter="recording_cell_filter.json",
    filter_grey_colour="#aaaaaa",
)
```

The UI uses the same frozen filter for exports. Download **Save filter JSON**
or **Reproduce in Python** to retain settings and evidence. The latter checks
the native observations and complete decision evidence before returning a result.

Implementation boundary: PyMicroglia evaluates cell criteria; Auto-Organotypic
only receives per-identity colour overrides. Default rendering is unchanged.
Acceptance targets are 1440 × 900 and 800 × 900: grey outlines, selectable grey
cells, unchanged cohort, independent exclusions, restored settings and no jobs.
