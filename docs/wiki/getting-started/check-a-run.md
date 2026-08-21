# Check a Run

## Goal

Find what a run wrote, inspect its scientific review, and recover the exact
equivalent script.

## From the Command Line

List recent runs for the current project:

```powershell
pymicroglia runs --limit 10 --table
```

Filter by an action or text from the recording name or claim:

```powershell
pymicroglia runs --action dluc_single_cell --contains "recording-name"
```

Read one full record:

```powershell
pymicroglia result RUN_ID
```

Recover its equivalent Python script:

```powershell
pymicroglia result RUN_ID --script > reproduce_run.py
```

`runs`, `result`, notebooks, and notes require the `kit` extra. Analysis itself
still runs without it, but the command output will say that the run was not
recorded.

## In the Results Folder

Complete pipelines write a run folder under `AI_Exports` by default. Start with:

- `REPORT.md`: decisions, warnings, blockers, evidence, and remedies.
- `manifest.json`: source, settings, stage order, outputs, and timing.
- Tier A artefacts and their JSON sidecars: permanent measurement results.
- Figures and videos: review products; the manifest identifies display-only files.

Do not treat a successful process exit as scientific acceptance. Resolve the
blockers in `REPORT.md` first.

## From Python

```python
from pymicroglia import Recording

recording = Recording(r"C:\path\to\AI_Exports\one-run-folder")
print(list(recording))
print(recording.about("registration"))
cells = recording.cells
traces = recording.traces
```

`Recording` reads result sidecars first and loads an artefact only when you ask
for it. Pointing it at a results folder does not open the source pixels.

## Next

- [Recording and Batch](../results/recording-and-batch.md)
- [Artefact store](../concepts/artefact-store.md)
- [Reanalyse with new parameters](../workflows/reanalyse-with-new-parameters.md)
