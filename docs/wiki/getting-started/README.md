# Getting Started

## Goal

Get from an installed PyMicroglia package to a checked environment, a validated
analysis request, and a run you can find and reproduce later.

## Before You Start

- Use Python 3.10 or newer.
- Work from a TIFF or OME-TIFF time-lapse. VSI conversion is a separate Fiji
  step; PyMicroglia does not silently convert a VSI file during analysis.
- Keep the original recording unchanged. Derived results go beside it under
  `AI_Exports` unless you choose another output folder.
- Start with read-only inspection commands before a long run.

## Steps

| Step | Page | Outcome |
|---|---|---|
| 1 | [Installation](installation.md) | Install the features your analysis needs. |
| 2 | [First analysis](first-analysis.md) | Check health, inspect defaults, validate parameters, and start a recorded run. |
| 3 | [Check a run](check-a-run.md) | Find outputs, review the record, and recover reproducible code. |

## Check It Worked

These commands should report `"ok": true` and zero pending actions:

```powershell
pymicroglia doctor
pymicroglia discover
```

## Next

- [Analysis flow](../concepts/analysis-flow.md)
- [Action index](../actions/README.md)
- [Pipeline index](../pipelines/README.md)
- [Troubleshooting](../troubleshooting/README.md)
