# Recording and Batch

## Recording

`Recording` exposes every stored result for one recording and loads each value
only when requested.

Build it from a results folder without opening the source image:

```python
from pymicroglia import Recording

rec = Recording(r"C:\path\to\AI_Exports\one-run-folder")
print(list(rec))
```

Or build it from a recording. PyMicroglia looks one level under the adjacent
`AI_Exports` folder and identifies matching sidecars:

```python
rec = Recording(r"C:\path\to\recording.ome.tif")
```

Common aliases include:

| Attribute | Stored stage |
|---|---|
| `rec.shifts` | `registration` |
| `rec.cosmic_events` | `cosmic_rays_events` |
| `rec.background` | `off_tissue_background` |
| `rec.labels` | `segmentation` |
| `rec.cells` | `segmentation_objects` |
| `rec.trace_objects` | `traces_objects` |
| `rec.decoys` | `decoy_test` |
| `rec.control` | `instrumental_control` |

New stage names remain available even without a friendly alias:

```python
value = rec["new_stage_name"]
provenance = rec.about("new_stage_name")
```

## Batch

`Batch` groups recordings under one export folder and combines compatible
results:

```python
from pymicroglia import Batch

batch = Batch(r"C:\path\to\AI_Exports")
print(batch.summary)
print(batch.cells)
first = batch.recordings[0]
```

Recordings are grouped by the source identity stored in their sidecars, not by
guessing from dated folder names.

## Why the Objects Are Not Pickled

`Recording` and `Batch` are views over files on disk. Reopening them reflects
new or replaced artefacts immediately, so they cannot silently retain values
computed under old settings. The source pixels remain closed unless you ask a
separate pixel API to read them.

## Display-only Results

By default, `Recording` exposes measurement artefacts. Construct it with
`display_only=True` only when you explicitly want display-branch products:

```python
display = Recording(
    r"C:\path\to\AI_Exports\one-run-folder",
    display_only=True,
)
```

Keeping the two views separate prevents a smoothed display array from being
mistaken for measurement input.

## See Also

- [API reference](../api-reference.md)
- [Analysis flow](../concepts/analysis-flow.md)
- [Check a run](../getting-started/check-a-run.md)
