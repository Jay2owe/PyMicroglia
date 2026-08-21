# Results

PyMicroglia writes ordinary files with sidecars instead of requiring a saved
Python session. The result interface rebuilds a current view of those files
when you open it.

| Layer | Use |
|---|---|
| Run record | Who ran what, with which resolved settings, method version, claim, outputs, and duration. |
| Pipeline manifest | Stage order, output folder, summary, review, open questions, and blocked state. |
| Tier A sidecar | Exact source, parameters, method, and upstream key for one derived artefact. |
| `Recording` | Lazy view over the stored results for one source. |
| `Batch` | Combined view over several recordings under one export folder. |

## Start With the Review

For a pipeline run, read `REPORT.md` before loading tables. It distinguishes:

- Notes: information to carry into interpretation.
- Checks: evidence a person should inspect.
- Blockers: unresolved conditions that prevent the stated scientific claim.

Then read `manifest.json` for the machine-readable version of the same run.

## See Also

- [Check a run](../getting-started/check-a-run.md)
- [Recording and Batch](recording-and-batch.md)
- [Artefact store](../concepts/artefact-store.md)
