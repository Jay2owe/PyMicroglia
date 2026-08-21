# First Analysis

## Goal

Inspect and validate a complete single-cell dLuc analysis before allowing it to
read a large recording.

## 1. Check the Environment

```powershell
pymicroglia doctor
```

Resolve any item under `complaints` before a long run. A missing Fiji connection
only blocks a manual region outline; it does not block unattended analysis.

## 2. Inspect the Pipeline

```powershell
pymicroglia describe dluc_single_cell
```

The response is the live parameter reference. Check these settings in
particular:

| Parameter | Default | Why it matters |
|---|---:|---|
| `t0` | `72.0` hours | Analysis starts after the first 72 hours unless you override it. |
| `channels` | inferred or stored | A wrong channel assignment invalidates everything downstream. |
| `skip_control` | `false` | Rhythm claims require the instrumental control. |
| `skip_videos` | `false` | Videos add time but do not feed measurement. |
| `if_exists` | `"version"` | Existing outputs are kept and a new run folder is created. |
| `reuse` | `true` | Exact stored upstream artefacts are reused. |

## 3. Validate the Request

Validation checks action and parameter names without starting the analysis:

```powershell
pymicroglia validate dluc_single_cell `
  source="C:\path\to\recording.ome.tif" `
  t0=72.0 `
  skip_videos=True
```

## 4. Run It

This command may read a very large stack and run for hours:

```powershell
pymicroglia run dluc_single_cell `
  source="C:\path\to\recording.ome.tif" `
  t0=72.0 `
  --claim "test whether individual microglia retain circadian dLuc rhythms" `
  --request "segment cells, extract traces, and test rhythmicity"
```

The human-written claim is required because this pipeline draws a scientific
conclusion. Mechanical actions such as registration can generate an honest
claim from the file and operation; segmentation, controls, and rhythm testing
cannot.

## What Happens

```text
source recording
    -> choose channels and usable time window
    -> register every channel
    -> remove cosmic-ray events from registered dLuc pixels
    -> segment objects and test area-matched decoys
    -> extract local-ring-subtracted traces
    -> run instrumental controls
    -> test rhythmicity
    -> write review, figures, tables, videos, and provenance
```

Display smoothing branches after measurement and never feeds a reported number.

## Next

- [Check a run](check-a-run.md)
- [Single-cell dLuc pipeline](../pipelines/dluc-single-cell.md)
- [Analysis flow](../concepts/analysis-flow.md)
