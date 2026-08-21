# Reanalyse With New Parameters

## Goal

Change one analysis setting while reusing every exact upstream result that is
still valid.

## 1. Inspect the Previous Run

```powershell
pymicroglia runs --contains "recording-name" --table
pymicroglia result RUN_ID
```

Save the equivalent script when you want a concrete starting point:

```powershell
pymicroglia result RUN_ID --script > previous_run.py
```

## 2. Identify the Stage That Owns the Setting

```powershell
pymicroglia describe segment
pymicroglia describe dluc_single_cell
```

For example, changing a segmentation threshold should not invalidate the
registration or cosmic-ray correction. Changing the registration method does.

## 3. Validate the New Request

```powershell
pymicroglia validate dluc_single_cell `
  source="C:\path\to\recording.ome.tif" `
  prominence=0.08 `
  reuse=True
```

Use the live output from `describe` for the actual parameter name and units.

## 4. Run With Explicit Reuse

```powershell
pymicroglia run dluc_single_cell `
  source="C:\path\to\recording.ome.tif" `
  prominence=0.08 `
  reuse=True `
  if_exists="version" `
  --claim "test whether the lower prominence gate admits valid microglial traces"
```

`reuse=True` does not mean “use something similar.” It means use an exact key
match. A changed parameter misses its owning stage and downstream stages while
unaffected upstream artefacts remain available.

## 5. Compare Reviews and Artefacts

- Keep both run folders with `if_exists="version"`.
- Compare `REPORT.md`, `manifest.json`, and the relevant quality-control figure.
- Use `Recording.about(stage)` to compare stored parameters and method versions.
- If reuse is surprising, call `store.explain(...)` for the stage.

## What Goes Wrong With Overwrite

`if_exists="overwrite"` removes the previous run folder before writing the new
one. Use it only when the old outputs are disposable; it makes visual and
provenance comparison harder.

## See Also

- [Artefact store](../concepts/artefact-store.md)
- [Check a run](../getting-started/check-a-run.md)
- [Pipeline folder policy](../pipelines/README.md#existing-folder-policy)
